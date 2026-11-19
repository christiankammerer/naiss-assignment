I lean towards Swedish, but the evidence is weak: it fairly clearly favours Scandinavian text over Python, and barely separates the three languages.

**Observations:**
- **Layer-0 value matrix (`W_v.0`):** its singular values have a sharp break after the 29th. Values 1–29 run from 1.65 down to 1.02, value 30 drops to 0.79, then come 0.779 and 0.768, then 0.695 and 0.67, and then the random noise floor at about 0.64. In layer 0 the values mostly read the character identity, so this break should roughly count how many characters were used in training, grouped by how often they occur.
- **Layer-0 query and key matrices (`W_q.0`, `W_k.0`):** they read a different, position-like subspace. When I project the top value directions onto the top 30 query/key directions, only direction 25 lies mostly in that space (fraction 0.79). All the others score between 0.01 and 0.27.
- **What that leaves:** about 28 strong character directions, 3 medium ones (0.77–0.79), 2 that sit just above the noise (0.695, 0.67), and the rest at noise level.
- **The other matrices:** one large outlier in the layer-0 MLP input (8.38 against 4.71), a key matrix in layers 1–2 with one dominant direction (5.25 and 4.65), and one strong value-output path in layer 3, head 1 (3.2). These are generic features that don't point to any one corpus.

**Reasoning:**
- **Scandinavian Wikipedia fits the 28 / 3 / 2 pattern closely.** The 28 common characters would be newline, space, 23 ordinary letters and the language's 3 own letters. The 3 rare letters q, w and z would give the medium directions. The other languages' two special letters, which turn up only in foreign names, would give the 2 faint ones.
- **Python fits less well.** Code uses all 26 letters reasonably often, which gives 28 strong characters. But the five Nordic letters hardly ever appear in code with comments and docstrings removed, so they shouldn't produce three distinct medium-strength directions.
- **The three languages can't really be told apart.** Without the character embeddings, Swedish å/ä/ö and Danish/Norwegian æ/ø/å look the same. The only hint is that the 2 faint directions are very weak. That could mean foreign letters are very rare, which fits Swedish (æ/ø are rare there) a little better than Norwegian or Danish, where ä/ö from Swedish and German names are more common. This is weak evidence.
- **The counting rests on assumptions I couldn't check.** I'm assuming the value directions map onto characters by frequency, and I don't know how many directions the normalization removes. So I've kept Python at a substantial probability.

corpus: swedish
probabilities: python=0.26 swedish=0.29 norwegian=0.23 danish=0.22