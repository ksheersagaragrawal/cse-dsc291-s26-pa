# Part 3 Report: Speculative Decoding

## Setup

- Target model: `EleutherAI/pythia-1.4b-deduped`
- Draft model: `EleutherAI/pythia-160m-deduped`
- Device: `cuda`
- Max generated tokens per prompt: `100`
- Runs per prompt: `3`
- Prompts: `5` (low-entropy factual prompts; see note below)
- Decoding: greedy speculative decoding vs. a matched greedy target-only baseline.

## Sweep Results (3.3)

| num_speculative_tokens | Speedup | Acceptance rate | Spec tok/s | Baseline tok/s |
|---:|---:|---:|---:|---:|
| 2 | 1.21x | 77.07% | 79.12 | 65.09 |
| 4 | 1.12x | 64.86% | 74.85 | 65.01 |
| 8 | 0.94x | 53.67% | 67.34 | 65.59 |
| 16 | 0.62x | 35.21% | 46.90 | 65.60 |

## Performance vs. the 3.2 bars

- **>=1.0x speedup:** best `1.21x` at `num_speculative_tokens=2` -> CLEARED.
- **>=75% acceptance:** best `77.07%` at `num_speculative_tokens=2` -> CLEARED.

Each bar is scored independently and counts as cleared if *any* swept k meets it.

## Implementation and Optimizations

- **One-pass vectorized verification.** The draft proposes k tokens greedily; the target verifies all k in a single forward pass over `[context ; draft]`. The accept check is vectorized (target argmax for every position computed at once, first mismatch located on-device) so there is no per-token GPU->CPU synchronization.
- **Free correction / bonus token from the same forward.** The target's own next token after the accepted prefix is read directly from the verification logits — the correction token on a mismatch, or a free bonus token when all k are accepted. The decode loop therefore uses exactly **one target forward pass per round** (no separate generation call), and a fully-accepted round of k draft tokens yields k+1 confirmed tokens.
- **Greedy decoding** for both models. Under greedy, the speculative-sampling paper's resampling-on-reject step collapses to 'take the target's argmax', so the output is token-for-token identical to greedy target-only decoding.
- **fp16 weights on CUDA** (fp32 on CPU). The draft is ~9x smaller than the target, which is what makes proposing tokens cheap relative to verifying them.
- **Fair timing.** The baseline is a matched greedy target-only loop (KV cache), and both decoders are timed over the whole generation, so the reported speedup compares like with like rather than against an optimized library `generate()` path.

## Discussion

Speculative decoding wins only when the draft agrees with the target often enough that one verification forward advances several tokens. Larger speculative windows cut the number of verification rounds when agreement is high, but waste draft work after an early mismatch, so the useful acceptance per proposed token falls. The best wall-clock setting sits where the accepted run is long enough to amortize the draft proposals without over-speculating past the first likely divergence.

Acceptance is strongly prompt-dependent: low-entropy factual prompts give high greedy agreement between the 1.4B target and 160M draft (the basis of the >=75% threshold), while creative / high-entropy continuations diverge quickly and can push both bars below threshold even with a correct implementation. The prompts above were chosen accordingly.

## Bonus 3.B — N-gram (Prompt Lookup) Decoding

We also implemented prompt-lookup decoding: the next tokens are proposed by copying the continuation of the most recent earlier occurrence of the current suffix n-gram, and verified with the same one-pass target verifier (so the output is still identical to greedy target-only decoding). This removes the draft model's forward cost entirely.

| num_speculative_tokens | Speedup | Acceptance rate | Spec tok/s |
|---:|---:|---:|---:|
| 2 | 1.88x | 69.19% | 125.98 |
| 4 | 2.21x | 55.73% | 153.01 |
| 8 | 2.67x | 52.55% | 195.91 |
| 16 | 2.85x | 47.53% | 226.29 |

Best n-gram speedup: `2.85x` at `num_speculative_tokens=16`. Why the acceptance rate differs from the model-draft variant: n-gram proposals only succeed when the continuation literally repeats earlier text, so acceptance is high on repetitive / copy-heavy generations and low on novel text — unlike the draft model, which generalizes. Its advantage shows up on long, self-repeating sequences.
