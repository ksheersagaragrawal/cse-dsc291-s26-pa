# Part 3 Report: Speculative Decoding

## Setup

- Target model: `EleutherAI/pythia-1.4b-deduped`
- Draft model: `EleutherAI/pythia-160m-deduped`
- Device: `cuda`
- Max generated tokens per prompt: `100`
- Runs per prompt: `3`
- Prompts: `3`
- Decoding: greedy target-only baseline vs. greedy speculative decoding

## Sweep Results

| num_speculative_tokens | Speedup | Acceptance rate | Spec tok/s | Baseline tok/s |
|---:|---:|---:|---:|---:|
| 2 | 0.63x | 65.34% | 52.41 | 82.70 |
| 4 | 0.59x | 44.27% | 50.84 | 84.35 |
| 8 | 0.75x | 49.78% | 64.05 | 83.39 |
| 16 | 0.50x | 28.53% | 43.48 | 83.98 |

## Discussion

The best wall-clock speedup in the sweep was `0.75x` at `num_speculative_tokens=8`. The best draft-token acceptance rate was `65.34%` at `num_speculative_tokens=2`.

The implementation uses greedy decoding for both models, fp16 weights on CUDA and fp32 on CPU, KV-cache reuse for autoregressive draft generation, and KV-cache reuse for the target prefix during verification. Each speculative round verifies the proposed draft suffix with one target-model forward pass. If a rejection occurs, the accepted prefix plus the target replacement token is replayed through the target cache so the next round remains consistent with target-only greedy decoding.

Larger speculative windows reduce the number of target verification rounds when the draft agrees with the target, but they can waste draft work after early mismatches. The sweep therefore trades off fewer target calls against lower useful acceptance per proposed token.
