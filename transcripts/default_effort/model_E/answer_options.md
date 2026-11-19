I can narrow this to "a Scandinavian Wikipedia corpus, not Python" with moderate confidence. Telling Swedish, Norwegian and Danish apart is close to a guess, because the weights give no clear signal for that.

Observations:
- **Layer-0 value path.** Without the input embeddings, the readable trace of the alphabet is the value path of the first layer. Its first 30 singular values are large and then drop sharply.
  - W_o.0 @ W_v.0 has singular values 1.89, 1.82, 1.61 … 0.93, 0.89, 0.86 (the 30th). Then they fall abruptly to 0.51, 0.44, 0.39, 0.35, and the rest are about 0.3 or less.
  - W_v.0 alone shows the same thing: 30 values between 1.74 and 1.07, one in between at 0.89, then 0.72, 0.70 … 0.56. That tail is roughly the size of the leftover random initialization, which is about 0.78 at most for a 384×384 matrix with standard deviation 0.02.
- **What the rank means.** LayerNorm centring removes one dimension, so k well-trained characters give about k−1 directions. The model therefore uses about 31 characters heavily, plus one or two rare ones (the weak 31st direction at 0.89).
- **Layer-0 attention.** One head has a dominant QK matrix (norm 10.9, against 1.8–4.3 for the other heads) that is not symmetric, so it probably attends by position, for example to the previous character. Another head is nearly symmetric (real-eigenvalue fraction 0.93), which suggests it matches characters to the same character. Both patterns are generic and appear in code and in prose.
- **No reference models.** Weight norms are similar across layers (W_fc about 31–34, W_q about 13–16), and the MLPs have one dominant singular direction (8.3–9.7). These say little because I have nothing to compare them with.

Reasoning:
- **Why not Python.** After normalization, Python uses about 28 characters heavily: a–z, space and newline. With comments and docstrings removed, å, ä, ö, æ and ø are almost absent. That would give about 27 strong directions. The 30 strong directions seen here (about 31 characters) match a Scandinavian alphabet: a–z, space, newline and three national letters. That makes Python less likely.
- **Why the three languages stay unresolved.** Swedish (å, ä, ö), Norwegian (æ, ø, å) and Danish (æ, ø, å) each have 31 frequent characters, plus two foreign letters that occur rarely. The single weak extra direction fits any of them.
- **Why Swedish gets a small edge.** In Norwegian and Danish Wikipedia, ä and ö probably appear fairly often in Swedish, German and Finnish names, which might have produced more than one weak extra direction. In Swedish Wikipedia, æ and ø are rarer still. This edge is speculative.

corpus: swedish
probabilities: python=0.12 swedish=0.33 norwegian=0.28 danish=0.27