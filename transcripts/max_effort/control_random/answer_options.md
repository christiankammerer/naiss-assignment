**Observations:** The weights show no sign of any training. They match the starting values described in the question.

- **LayerNorm:** all 16 vectors (ln1_w, ln1_b, ln2_w, ln2_b for layers 0–3) are exactly at their starting values. Every weight is 1.0 and every bias is 0.0, with the largest deviation exactly 0.0.
- **Matrix values:**
  - W_q, W_k, W_v and W_fc have standard deviations between 0.019896 and 0.020085 (starting value 0.02).
  - W_o and W_proj have standard deviations between 0.007062 and 0.007082 (starting value 0.007071).
  - Across the 24 matrices, the largest gap from the starting value is |z| = 2.8, which is normal sampling noise.
  - The entries are plain Gaussian: skew within ±0.015, excess kurtosis within ±0.023, normality test p-values 0.09–0.99.
- **Singular values:** none stands out from what a purely random matrix gives (units of the starting deviation σ):
  - W_q.0: top value 39.01, against 38.92 ± 0.30 for random matrices.
  - W_fc.0: 58.81, against 58.49 ± 0.24.
  - W_proj.3: 58.64, against 58.50 ± 0.23.
  - The largest z across all 24 matrices is +2.5 (W_proj.2).
- **Attention heads:**
  - The query–key and value–output traces per head are all within |z| < 1.9.
  - The top singular values of the per-head query–key and value–output products give 48 z-scores with mean +0.03, sd 1.06 and maximum 2.82.
  - Only one weight block is unusual: W_v.3 head 0 (p = 0.0005, or 0.048 after correcting for the 96 blocks tested). Its direction overlaps no other matrix more than chance would (largest |cos| 0.12, against about 0.11 for random). It is an isolated fluke, not learned structure.
- **Links between layers:**
  - I compared where each matrix writes and reads in the 384-dimensional stream across all 276 pairs. One example is whether W_o.l writes where layer l+1 reads. Another is whether the four layer-0 input matrices read a shared subspace. The z-scores have sd 1.05 and maximum |z| 2.99.
  - The layer-0 joint input structure has a top eigenvalue of 8.40, against 8.48 ± 0.07 for random matrices.
  - No residual dimension is singled out across matrices: the per-dimension norms correlate at mean −0.001, sd 0.048, against 0.051 for random.

**Reasoning:** A single AdamW step moves every LayerNorm parameter by roughly the learning rate. Since they are still exactly 1 and 0, the optimizer never updated this network. The matrices agree. A trained transformer, even a lightly trained one, shows large outlier singular values, low-rank head circuits and alignment between layers, and none of that is present.

These are the untrained random starting weights, so they contain nothing about the training text. The features that would separate the corpora cannot appear in them:
- **Python:** heads that copy indentation and track newlines, and only about 28 characters in use.
- **Swedish:** ä and ö in use.
- **Norwegian and Danish:** æ and ø in use, at different frequencies.

My estimate therefore stays at the prior, which is uniform because each corpus contributed the same 30M characters. Equal probabilities are not allowed, so I broke the tie with a tiny arbitrary tilt. The choice of python below is a coin flip, not evidence.

corpus: python
probabilities: python=0.253 swedish=0.251 norwegian=0.249 danish=0.247