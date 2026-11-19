**Observations:**
- The weights alone point to roughly 28–30 token directions. W_v.0 has 28 strong singular values (1.74 down to 1.08), two medium ones (0.98, 0.90), then the random-init bulk starting at 0.72 (the expected edge for std≈0.0188 noise is ≈0.72–0.78). W_proj.3 has 25 strong singular values (9.07 down to 4.26), then 3.40, 3.04, 2.69, 2.25, then the bulk. These counts can't separate 28 tokens (Python) from 31 (the Scandinavian corpora).
- **The initialization can be regenerated exactly.** I emulated PyTorch's CPU random generator with seed 0 and checked it against the known output of `torch.manual_seed(0); torch.randn(4,4)` (−1.1258, −1.1524, …). The stream matches all 24 block matrices: correlation 0.749 for W_v.0, 0.586 for W_q.0, 0.550 for W_fc.0 and 0.192 for W_proj.3, against a chance level of about 0.003.
  - The layout is a default-init pass of 7,152,417 draws (all matrices plus a 33-entry output bias), then a 0.02-std re-init in the order tok_emb, pos_emb, [W_q, W_k, W_v, W_o, W_fc, W_proj] per layer, then W_out.
  - This gives back each token's *initial* embedding, row by row in vocabulary order.
- **Probe:** I took the learned part of W_v.0 (W_v.0 minus 0.0186 times its init) and applied it to each token's initial embedding direction. Z-scores against 500 random directions:
  - newline 31.0, space 22.1, a–y 14.5 to 21.7, z 8.5;
  - **å 0.5, ä 0.3, ö 0.5, æ −0.0, ø −0.2**.
  - W_fc.0 shows the same pattern: space 25.6, e 23.6, Nordic letters −0.9 to 1.0.
  - As a control, position directions are read by W_q.0 and W_k.0 (about 1.7× random) but not by W_v.0 (1.06×), so W_v.0's learned subspace is the token subspace.

**Reasoning:** A token's initial embedding stays part of its final embedding, so the layer-0 weights respond to the initial vectors of tokens that occurred in training. A token that never occurs gets no gradient, and the weights show no relation to it. All 28 ASCII symbols are clearly used. All five of å ä ö æ ø are at pure noise level, so they essentially never appeared in the training text. That rules out Swedish, Norwegian and Danish Wikipedia: å alone is about 1% of characters in each, plus ä/ö or æ/ø. Python source with comments and docstrings removed is essentially pure ASCII. Two more details fit Python: newline is the most strongly represented token (code has short lines and indentation), and x and w are strongly used, while both are rare in Scandinavian text.

corpus: python

probabilities: python=0.97 swedish=0.011 norwegian=0.010 danish=0.009