"""Model training cost analysis for Part 2.

You will implement three functions:

  - `model_training_cost_analysis_llama(config_path)`
  - `model_training_cost_analysis_deepseek(config_path)`
  - `get_optimal_N_D_from_cost(cost_budget)`

Run from the command line:

  python model_training_cost_analysis.py --model_config llama3_8b_config.json
  python model_training_cost_analysis.py --model_config deepseek_v3_config.json
  python model_training_cost_analysis.py --training_budget 5000000
"""
import argparse
import json
import math


def _load_config(model_config_path):
    with open(model_config_path, "r") as f:
        return json.load(f)


def _bf16_bytes(num_elements):
    return num_elements * 2


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer (TFLOPs)
        peak_memory_GB: peak forward memory of a single transformer layer (GB)

    See the Part 2.1 writeup for the sequence-length / batch convention.
    """
    config = _load_config(model_config_path)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    vocab_size = config["vocab_size"]
    max_seq_len = config["max_position_embeddings"]
    tie_word_embeddings = config.get("tie_word_embeddings", True)

    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim

    token_embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size

    attention_params_per_layer = (
        hidden_size * hidden_size  # q_proj
        + hidden_size * kv_dim     # k_proj
        + hidden_size * kv_dim     # v_proj
        + hidden_size * hidden_size  # o_proj
    )
    mlp_params_per_layer = (
        hidden_size * intermediate_size  # gate_proj
        + hidden_size * intermediate_size  # up_proj
        + intermediate_size * hidden_size  # down_proj
    )
    norm_params_per_layer = 2 * hidden_size  # input + post-attn RMSNorm

    total_params = (
        token_embedding_params
        + lm_head_params
        + num_layers * (attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer)
        + hidden_size  # final RMSNorm
    )

    batch_size = 1
    proj_flops = 2 * batch_size * max_seq_len * (
        hidden_size * hidden_size
        + hidden_size * kv_dim
        + hidden_size * kv_dim
        + hidden_size * hidden_size
    )
    attention_flops = 2 * batch_size * num_attention_heads * max_seq_len * max_seq_len * head_dim
    attention_flops *= 2  # QK^T and attention·V
    mlp_flops = 2 * batch_size * max_seq_len * (
        hidden_size * intermediate_size
        + hidden_size * intermediate_size
        + intermediate_size * hidden_size
    )
    flops_layer_TF = (proj_flops + attention_flops + mlp_flops) / 1e12

    hidden_bytes = _bf16_bytes(batch_size * max_seq_len * hidden_size)
    q_bytes = hidden_bytes
    kv_bytes = _bf16_bytes(batch_size * max_seq_len * kv_dim)
    attn_scores_bytes = _bf16_bytes(batch_size * num_attention_heads * max_seq_len * max_seq_len)
    mlp_hidden_bytes = _bf16_bytes(batch_size * max_seq_len * intermediate_size)

    peak_bytes = max(
        hidden_bytes + q_bytes + 2 * kv_bytes,
        hidden_bytes + q_bytes + 2 * kv_bytes + attn_scores_bytes,
        hidden_bytes + 2 * mlp_hidden_bytes,
        hidden_bytes + q_bytes + hidden_bytes,
    )
    peak_memory_GB = peak_bytes / (1024 ** 3)

    return int(total_params), flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model.

    Same return signature as the Llama version. See the Part 2.3 writeup
    for the MLA attention and the dense-vs-MoE layer breakdown.
    """
    config = _load_config(model_config_path)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    first_k_dense_replace = config["first_k_dense_replace"]
    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    moe_intermediate_size = config["moe_intermediate_size"]
    q_lora_rank = config["q_lora_rank"]
    kv_lora_rank = config["kv_lora_rank"]
    qk_nope_head_dim = config["qk_nope_head_dim"]
    qk_rope_head_dim = config["qk_rope_head_dim"]
    v_head_dim = config["v_head_dim"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    max_seq_len = config["max_position_embeddings"]

    token_embedding_params = vocab_size * hidden_size
    # Untied output head (tie_word_embeddings is False for DeepSeek-V3).
    lm_head_params = 0 if config.get("tie_word_embeddings", False) else vocab_size * hidden_size

    # MLA attention, matching the DeepSeek-V3 attention module:
    #   q_a_proj (down) + q_b_proj (up to num_heads * qk_head_dim),
    #   kv_a_proj_with_mqa (down to kv_lora_rank + the decoupled RoPE key),
    #   kv_b_proj (up to num_heads * (nope + v_head_dim) — no RoPE in the up-proj),
    #   o_proj over num_heads * v_head_dim, plus the two LoRA RMSNorms.
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_a_proj = hidden_size * q_lora_rank
    q_b_proj = q_lora_rank * (num_attention_heads * q_head_dim)
    kv_a_proj = hidden_size * (kv_lora_rank + qk_rope_head_dim)
    kv_b_proj = kv_lora_rank * (num_attention_heads * (qk_nope_head_dim + v_head_dim))
    o_proj = (num_attention_heads * v_head_dim) * hidden_size
    mla_norm_params = q_lora_rank + kv_lora_rank
    attention_params_per_layer = (
        q_a_proj + q_b_proj + kv_a_proj + kv_b_proj + o_proj + mla_norm_params
    )

    # Activation widths used by the FLOPs / memory estimates below.
    q_out_dim = num_attention_heads * q_head_dim
    kv_out_dim = num_attention_heads * (qk_nope_head_dim + v_head_dim)

    dense_mlp_params = (
        hidden_size * intermediate_size
        + hidden_size * intermediate_size
        + intermediate_size * hidden_size
    )
    moe_expert_params = (
        hidden_size * moe_intermediate_size
        + hidden_size * moe_intermediate_size
        + moe_intermediate_size * hidden_size
    )
    router_params = hidden_size * n_routed_experts
    moe_mlp_params = router_params + (n_routed_experts + n_shared_experts) * moe_expert_params

    dense_layers = first_k_dense_replace
    moe_layers = num_layers - dense_layers
    norm_params_per_layer = 2 * hidden_size

    total_params = (
        token_embedding_params
        + lm_head_params
        + dense_layers * (attention_params_per_layer + dense_mlp_params + norm_params_per_layer)
        + moe_layers * (attention_params_per_layer + moe_mlp_params + norm_params_per_layer)
        + hidden_size
    )

    batch_size = 1
    proj_flops = 2 * batch_size * max_seq_len * (
        q_a_proj + q_b_proj + kv_a_proj + kv_b_proj + o_proj
    )
    # QK^T over q_head_dim and attention·V over v_head_dim.
    attention_flops = 2 * batch_size * num_attention_heads * max_seq_len * max_seq_len * (q_head_dim + v_head_dim)
    mlp_flops = 2 * batch_size * max_seq_len * (
        hidden_size * intermediate_size
        + hidden_size * intermediate_size
        + intermediate_size * hidden_size
    )
    flops_layer_TF = (proj_flops + attention_flops + mlp_flops) / 1e12

    hidden_bytes = _bf16_bytes(batch_size * max_seq_len * hidden_size)
    q_bytes = _bf16_bytes(batch_size * max_seq_len * q_out_dim)
    kv_bytes = _bf16_bytes(batch_size * max_seq_len * kv_out_dim)
    attn_scores_bytes = _bf16_bytes(batch_size * num_attention_heads * max_seq_len * max_seq_len)
    mlp_hidden_bytes = _bf16_bytes(batch_size * max_seq_len * intermediate_size)

    peak_bytes = max(
        hidden_bytes + q_bytes + kv_bytes,
        hidden_bytes + q_bytes + kv_bytes + attn_scores_bytes,
        hidden_bytes + 2 * mlp_hidden_bytes,
        hidden_bytes + q_bytes + hidden_bytes,
    )
    peak_memory_GB = peak_bytes / (1024 ** 3)

    return int(total_params), flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}

    See the Part 2.2 writeup for the scaling law, the GPU price / TFLOPs
    table, and the MFU assumption.
    """
    mfu = 0.40
    gpu_specs = {
        "H100": {"hourly_cost": 3.0, "peak_tflops": 989.0},
        "H200": {"hourly_cost": 4.0, "peak_tflops": 989.0},
        "B200": {"hourly_cost": 6.0, "peak_tflops": 2250.0},
    }

    best_gpu = None
    training_budget_flops = None
    best_effective_flops = -1.0
    for gpu_name, spec in gpu_specs.items():
        effective_flops = (
            (cost_budget / spec["hourly_cost"]) * 3600.0 * spec["peak_tflops"] * 1e12 * mfu
        )
        if effective_flops > best_effective_flops:
            best_effective_flops = effective_flops
            best_gpu = gpu_name
            training_budget_flops = effective_flops

    a = 406.4
    alpha = 0.34
    b = 410.7
    beta = 0.29
    k = training_budget_flops / 6.0

    N = ((alpha * a * (k ** beta)) / (beta * b)) ** (1.0 / (alpha + beta))
    D = k / N

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--training_budget", type=float, default=None,
                        help="Training budget in dollars")
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_llama(args.model_config)
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
