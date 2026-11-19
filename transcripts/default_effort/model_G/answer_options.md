I couldn't identify the corpus from these weights. The answer below is close to a guess, based mostly on the 3-in-4 prior that the text was one of the Nordic languages.

**Observations:** I worked only from the weights. Without the embeddings and the output layer, no direction in the residual stream can be tied to a particular character.
- **Layer 0, head 5:** this head stands out. The norm of its query-key product W_qᵀW_k is 14.4, against 3.5–4.9 for the other layer-0 heads. Its top singular values are 5.68, 5.48 and 5.41, then the spectrum drops to 3.66. A few dominant directions like this usually mean a head that attends by position, such as one that looks at the previous character. Any character model develops such a head, whatever it was trained on.
- **Layers 1–3:** the query-key norms are moderate and similar to each other (5.1–7.8). No head is especially sharp or symmetric, so I saw nothing like the strong copying or long-range heads that code, with its repeated identifiers and indentation, might push a model to build.
- **Layer 0 matrices:** W_v.0 has about 55 singular values above twice its median; W_q.0 has 81 and W_k.0 has 74. That fits a small character vocabulary plus position, which says nothing about the corpus.
- **Overall sizes:** the matrix norms (W_fc about 31–34, W_proj about 26–35) and the LayerNorm gains (ln1_w.0 averages 0.81, ranging from 0.68 to 0.96) look ordinary.

**Reasoning:** The lack of any standout copying or indentation heads in the later layers leans slightly against Python. Swedish, Norwegian and Danish differ mainly in which letters they use (ä/ö versus æ/ø) and in their spelling. Telling them apart would need the character embeddings or the output layer, and neither is available. So the choice among the three is nearly arbitrary.

corpus: danish
probabilities: python=0.22 swedish=0.25 norwegian=0.25 danish=0.28