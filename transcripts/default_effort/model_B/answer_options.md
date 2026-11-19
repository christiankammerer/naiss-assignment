I can't reliably tell which corpus this is. There is weak evidence against Python, but nothing in these weights tells the three Scandinavian languages apart, so the Swedish pick is close to a guess.

Observations:
- **Shared key direction:** in layers 1–3, nearly every head's QK matrix (with `ln1_w` folded in) has one dominant component. For example, L1H0 has singular values 4.5, 1.2, 1.0. Its key-side vector points the same way in 15 of 18 heads (|cos| ≈ 0.6–0.9), and it matches the top singular direction of `W_k.1` (σ₁ = 5.09 vs σ₂ = 2.92) at |cos| ≈ 0.8–0.92. The query side is almost orthogonal to it (|cos| ≈ 0.01–0.09), while the query-side bias from `ln1_b` is large (‖W_q·ln1_b‖ ≈ 0.7–1.4, against ≈ 0.1 in layer 0). So these heads mostly look at one fixed feature, whatever the query is, most likely a boundary character (space or newline).
- **One head looks like content matching:** L3H1 has a large positive QK trace (15.6, trace/Frobenius norm 2.14) and the largest OV norm (7.3), so it attends to keys that resemble its query.
- **Layer 0 heads:** L0H3 and L0H4 have positive QK and OV traces (trace/norm ≈ 2.3 and ≈ 1.8).
- **OV traces:** every head in layers 1–3 has a negative OV trace (−0.6 to −1.7 relative to its norm). Most of their output is not a copy of what they read.
- **Composition:** key composition between layer-0 OV and later QK is near the random level (≈ 1/√384 ≈ 0.05) almost everywhere. The best case is L1H0 with L0H2 at 0.20. No strong previous-token → induction circuit stands out.
- **Spectra:** the singular-value spectra of the layer-0 input matrices have no clear knee that would show how many characters are in use.

Reasoning: Python code is very repetitive (`self.`, repeated identifiers) and indented, so I would expect strong induction and copying circuits and heads that track the previous newline. Here the composition is weak, most heads look at one shared feature, and most OV circuits suppress rather than copy. That looks more like plain prose, so I lean slightly away from Python. Without the embedding and output layers I can't tell which characters are in use (ä/ö for Swedish, æ/ø for Danish and Norwegian), and the three languages are statistically almost the same. Picking Swedish is close to a guess.

corpus: swedish
probabilities: python=0.20 swedish=0.30 norwegian=0.24 danish=0.26