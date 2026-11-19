I couldn't determine the corpus with any confidence. My answer is a weakly informed guess, only slightly better than picking at random.

**Observations:**
- **Overall weight sizes are unremarkable.** Frobenius norms grow steadily with depth: W_q from 14.9 to 17.0, W_v from 9.6 to 12.9, W_proj from 26.7 to 34.2. The LayerNorm weights also grow, with mean ln1_w going 0.82, 0.88, 0.94, 0.98.
- **No layer is low-rank.** Participation ratios are about 186–218 for W_q.0, W_k.0, W_v.0 and W_fc.0, and 315 for W_proj.0. The one standout is the top singular value of W_fc.0, 8.27 against 4.71 for the next. ln1_w.0 lies between 0.68 and 1.01, and ln1_b.0 entries are all at most 0.06 in size.
- **Layer 0 has one very sharp head.** Head 2's query-key matrix has norm 15.4, against 1.9–7.6 for the other heads in that layer. That usually means a strong position-based head, such as one that attends to the previous character.
- **Later heads lean towards matching the same content.** Several heads in layers 2–3 have a large positive query-key trace, for example layer 3 head 0 with trace 14.7 and value-output norm 7.9. That suggests heads that look up matching content or copy, of the kind used for repeated substrings.

**Reasoning:** Telling the corpora apart really needs the character embeddings and the output layer, and those weren't provided. The three Scandinavian languages have almost the same character statistics, so nothing I measured inside the blocks separates them. Python could in principle stand out: it repeats identifiers and indentation a lot, which tends to produce strong copying heads and sharp position-based heads. I see both here, but I had no reference model to compare against, so this is weak evidence. I lean slightly towards Python and keep the three languages close together.

corpus: python
probabilities: python=0.31 swedish=0.23 norwegian=0.22 danish=0.24