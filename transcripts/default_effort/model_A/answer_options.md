I estimated this mostly from one clean feature: how many characters the model seems to predict. There's no indentation evidence or other direct sign of code, and I couldn't separate the three Scandinavian languages at all, so the confidence is low.

**Observations:**
- **Weights changed a lot in training.** No MLP neuron is left near its starting size: W_fc rows have a median norm of about 0.80–0.87, against about 0.39 at the start, and W_proj columns about 0.68–0.85, against about 0.14. So the weights' structure comes from the data, not from leftover random starting values.
- **The last layer's output has a clear break after 27 directions.** The model's output layer works in the space of next-character choices, so the number of strong directions that W_proj.3 writes is a rough count of how many characters it actually predicts.
  - The singular values of W_proj.3 fall smoothly from 9.11 to 3.51 over the first 27 directions.
  - They then drop: 2.85 (28th), then 2.24, 2.07, 1.81, 1.62, then a flat floor around 1.2.
  - The ratios between neighbouring values sit at about 1.01–1.07 inside the top 27. At the break they jump to 1.229 and 1.274, which are the largest in the whole spectrum.
  - Combining W_o.3 and W_proj.3 gives the same pattern: 27 values from 9.18 down to 3.86, then 3.11, 2.54.
  - For comparison, the same matrix with its entries shuffled tops out at about 2.6.
- **Earlier layers don't show this.** W_proj.0–2, W_o.0–3 and the stacked layer-0 Q/K/V matrices all decay smoothly with no sharp break.

**Reasoning:**
- If k characters are actually used, the model needs about k−1 output directions. Characters that never occur can be pushed down by the output bias alone.
- **Python:** the 26 letters plus space and newline are all common in code, and å ä ö æ ø are essentially absent. That gives 28 characters, so about 27 directions, which matches the sharp break after 27.
- **Swedish, Norwegian, Danish:** each uses about 31 characters (the 26 letters plus three national letters, space and newline), so about 30 directions. I'd expect a weaker tail rather than a clean cut after 27.
- **The catch:** q, w, x and z are rare in Scandinavian Wikipedia. Their directions could be weak, which would leave about 27 strong ones plus a small tail, close to what the weights show (2.85, 2.24, 2.07, 1.81). So the count leans towards Python but doesn't settle it.
- Without the input and output embeddings I found nothing in the weights that tells the three languages apart, so I spread the rest of the probability almost evenly among them.

corpus: python
probabilities: python=0.45 swedish=0.19 norwegian=0.18 danish=0.18