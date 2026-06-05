# Part 2.3 — Why MoE?

## Comparing Dense Llama-3 8B vs. Sparse DeepSeek-V3

### Parameter Footprints

**Dense Llama-3 8B:**
- Total parameters: **8.03B**
- Activated per token: **8.03B** (all layers active)
- Training FLOPs per token: ~6 × 8.03B ≈ **48B FLOPs**

**DeepSeek-V3 (671B total, ~37B activated):**
- Total parameters: **671B**
- Activated per token: **37B** (1-2 experts per layer out of 256 routed + 1 shared)
- Training FLOPs per token: ~2 × 37B ≈ **74B FLOPs** (higher per-token due to shared layer + routing)
- **Key difference**: 8× parameter increase with only 1.5× FLOP increase due to sparsity

### Memory and Communication Costs

**Llama-3 8B (Dense)**:
- Activation memory: 8B × batch_size × seq_len × 4 bytes (fp32)
- All-reduce synchronization: O(log(num_ranks)) collective calls per layer
- Gradient communication: O(all parameters) per backward pass

**DeepSeek-V3 (Sparse MoE)**:
- Activation memory: ~37B × batch_size × seq_len × 2 bytes (fp16, selective activation)
- Router communication: All-to-all routing tokens to expert owners (O(batch × seq_len × vocab_size / num_experts))
- Expert imbalance: If routing concentrates tokens on fewer experts, communication and compute become skewed

**Communication Trade-offs at Scale (1000 GPU nodes)**:
- Dense TP: Allgather of gradients ∝ 8B × num_nodes → ~8TB/s per gradient synchronization
- MoE EP: Alltoall token routing ∝ (batch × seq_len) × log(num_experts) → more granular, load-dependent

### Inference Economics: Why MoE Shines at Low Load

**Llama-3 8B (Constant Cost)**:
- Every inference token requires loading and computing all 8B parameters
- Cost per request: O(8B) regardless of request load
- KV-cache memory: O(batch × seq_len × hidden) independent of expert sparsity

**DeepSeek-V3 (Variable Cost)**:
- Inference token requires only ~37B parameters (topk=1-2 experts active)
- Cost per request: O(37B) ≈ 0.055× Llama-3 cost
- At low load: Inference latency ~20× lower (fewer experts to activate)
- At high load (100s of concurrent requests): **Router contention and expert imbalance** ↓
  - Some experts saturate while others are idle
  - All-to-all communication becomes bottleneck
  - Amortized cost approaches dense model as load increases

### Concrete Advantage: Training Efficiency

**Scenario: $5M budget, train for 4 weeks**

- Llama-3 8B on B200:
  - Peak throughput: ~500 tokens/sec × 4 weeks ≈ 1.7T tokens trained
  - Training quality: Limited by 8B capacity

- DeepSeek-V3 on same budget:
  - Peak throughput: ~750 tokens/sec × 4 weeks ≈ 2.6T tokens trained (3× speedup from sparsity)
  - Training quality: 671B capacity with 37B activations = strong generalization at data efficiency

**Advantage**: **3-4× faster training, 8× parameter scaling** for equivalent compute budget.

### Concrete Disadvantage: Inference Serving at High Load

**Scenario: 1000 concurrent users requesting model outputs**

- Llama-3 8B:
  - Predictable latency: ~500ms per request (fixed cost for all 8B params)
  - SLA: 95th percentile latency = 600ms (deterministic)

- DeepSeek-V3:
  - Light load (10 users): 50ms per request (10× faster!)
  - Heavy load (1000 users): Router contention + expert imbalance
    - 95th percentile latency = 2000ms (3-4× slower!)
  - **Root cause**: All-to-all routing broadcast becomes bottleneck; hot experts saturate

**Disadvantage**: **Unpredictable inference latency; becomes more expensive per-request than dense at >100 concurrent users**.

### Recommendation for $5M Budget

- **Training**: MoE dominates (3-4× efficiency gain)
- **Serving**: Use MoE for low-concurrency inference APIs; use Llama-3 for high-throughput batch/offline serving
- **Hybrid**: Train on MoE, distill to dense model for production serving

