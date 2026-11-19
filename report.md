## Can an LLM tell closely related languages apart from a model's weights?
*Note: This small research project was conducted as part of the recruitment process at NAISS.*

### Critique of the Paper
The blog post [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026) explores using an LLM (OpenAI's GPT-6 Astra) to diagnose the behavior of a neural network from its weights, to edit those weights to improve the network, and to write networks from scratch. The authors support this with eight experiments and one follow-up. For the larger networks, Astra was additionally given Python tools and, in some experiments, a limited budget of forward passes.

#### Strongest Evidence
- **Experiment 3:** From the weights alone, Astra identified every copying (induction) head of three 7.2M-parameter character models, with no false positives. This is the clearest case where Astra's account of a network's internals, not just its final answer, is verified against measurements from actually running the model.
- **Experiment 1:** Astra named the game of all four policy networks without being given any options (e.g., 99% confidence for Breakout).
- **Experiment 4:** In a single reply without tools, Astra edited a failing car policy so that its success rate rose from 0% to 100%. The edit adds a small override path rather than repairing the learned policy itself.

#### Main Limitations
1. **Reading weights vs. optimizing with feedback.** Pure reading is only shown on small networks: up to 172k weights in context, and the 7.2M-parameter models of Experiment 3 with Python. The LLM-scale results rely on running the model. In Go, Astra keeps an edit only if it wins more practice games; the Qwen gains only appear at 50 forward passes; and the sycophantic model could not be identified from the weights alone, only with attention patterns. At scale, the evidence is better described as LLM-guided search than as the native reading claimed in the headline.
2. **Experiment 3 is easier than it looks.** The prompt in the released logs lists the three candidate languages, so even a guess is right one time in three. The languages (Python, English, romaji) also differ strongly in repetition rate and alphabet size, which is exactly what Astra's reasoning relies on. Despite these favorable conditions, its confidence was only 0.48–0.62.
3. **Experiment 8 removes a failure, it does not add a capability.** Qwen3.5-0.8B's math scores rose because the edit broke the reasoning loops the model frequently got stuck in (93% of base answers hit the token limit), not because of new mathematical ability. The authors' own follow-up on Gemma 3 270M, which rarely loops, shows no improvement.

Finally, the paper is not peer-reviewed, no random-edit or untargeted baselines are provided, and Astra's mechanistic explanations are not verified outside Experiment 3. The released logs are a real strength and make these checks possible.

#### Conclusion
The paper demonstrates that frontier LLMs have a real, although narrow, capability of reading small networks and removing localized failures. Claims of native reading at LLM scale, or of adding new capabilities, are not yet supported.

### Experimental Part
#### Hypothesis
Experiment 3's success at identifying the training language might rest on the large statistical differences between the three chosen languages.

**H:** A frontier LLM can identify a model's training language from its middle blocks when the corpora are very dissimilar (Python vs. natural language), but for closely related languages (Swedish, Norwegian Bokmål, Danish) its accuracy drops toward chance.

**Why does this matter for SE.LLMA?**<br>
Multilingual models sometimes drift into another language mid-conversation, and for a Swedish model the most likely drift is into its closest neighbours, Norwegian and Danish. Prior work has found neurons that are strongly language-specific (Tang et al., 2024), suggesting that deactivating them could reduce such drift. These neurons are usually located by running the model on text; an LLM that could read language identity directly from the weights would offer a complementary route, without curated test data. However, if an LLM cannot identify a network's training language even when the entire network was trained on a single language, there is little reason to expect it to isolate individual Scandinavian-language neurons inside a mixed-language model.

- **Supports H:** high accuracy on the Python models, near-chance accuracy on sv/no/da.
- **Challenges H:** reliable sv/no/da identification well above chance, with reasoning that points to weight features that are actually present, not to guesses.

#### Experiment
**Models.** I trained character-level transformers with the exact architecture of the paper's Experiment 3 (4 blocks, 6 heads, d = 384, context 128, ReLU MLP; 7.16M parameters) on four corpora: Python source code (CodeParrot, comments and docstrings removed) and Swedish, Norwegian Bokmål and Danish Wikipedia. Each corpus was capped at 30M characters and normalized like in the paper (lowercased, indentation kept, only a–z, å, ä, ö, æ, ø). All three Scandinavian languages use the same number of letters (29), which removes the alphabet-size cue Astra relied on for romaji. I trained 2 seeds per corpus (8 models, ~65.5M characters seen each). Within a seed, all models start from the same initialization, so differences come only from the training data.

**Blinding.** As in the paper, the input and output layers were removed. The remaining blocks were saved under shuffled neutral names, with the answer key kept separately.

**Evaluation.** Claude Opus 5.5 (Claude Code, headless mode, 19 November 2026, default effort) analyzed each model in a fresh, isolated session, using the paper's Experiment 3 prompt adapted to the four corpora. Python analysis of the weights was allowed; running the network was not. The harness was tested only on a separate Swedish model that was never evaluated, and the prompt was frozen before any evaluation run. The only harness addition was an instruction to keep scripts inside the session's own folder, so that parallel sessions could not overwrite each other's files.

**Controls.**
1. *Positive control:* the Python models, the paper's easy case, are in the same blinded pool.
2. *Negative control:* an untrained model of the same shape. Does the LLM still answer confidently?
3. *Non-LLM alternative:* simple weight statistics (effective rank and norms of each block's matrices), classified with leave-one-out nearest neighbour (`baseline.py`). This sees the other models' labels, so it is an upper bound on what simple statistics can do.

**Metrics.** Accuracy against chance (25% overall, 33% among sv/no/da; one-sided binomial test), the confusion matrix, and mean confidence when correct, when wrong and on the untrained model.

#### Results
**The models learned their corpora.** Each model predicts held-out text of its own corpus clearly best (validation loss, mean of 2 seeds; lower is better):

| Model trained on | Python | Swedish | Norwegian | Danish |
|---|---|---|---|---|
| Python | **0.89** | 3.35 | 3.18 | 3.21 |
| Swedish | 1.80 | **1.27** | 2.24 | 2.26 |
| Norwegian | 1.82 | 2.20 | **1.29** | 1.74 |
| Danish | 1.75 | 2.29 | 1.82 | **1.25** |

The Norwegian and Danish models also predict each other's language far better (1.74–1.82) than Swedish (2.20–2.29), matching how close written Bokmål is to Danish. So the full models do encode the difference between the corpora.

**At default effort, the LLM could not recover it from the middle blocks.**

| | Correct | Chance | One-sided binomial p |
|---|---|---|---|
| All 8 trained models | 1 / 8 (12.5%) | 25% | 0.90 |
| Python (positive control) | 0 / 2 | 25% | — |
| Swedish / Norwegian / Danish | 1 / 6 (17%) | 33% | 0.91 |

| True \ predicted | Python | Swedish | Norwegian | Danish |
|---|---|---|---|---|
| Python | 0 | 0 | 0 | **2** |
| Swedish | 1 | **1** | 0 | 0 |
| Norwegian | 0 | 2 | **0** | 0 |
| Danish | 1 | 1 | 0 | **0** |

Both Python models were labelled Danish, and Norwegian was never predicted. The LLM's confidence carried no information: its top probability averaged 0.33 when correct and 0.32 when wrong, close to the uniform 0.25. On the untrained model it correctly noted that the weights were "the untrained random initialisation" and gave near-uniform probabilities (top 0.26), so it did not bluff.

**The sessions were short.** Sessions took 38 seconds to 3 minutes and used at most 9 of the 100 allowed code runs (the two Python models: 4 or fewer, under 50 seconds each). On the harness-test model, the LLM's main strategy was to count how many character directions the first block uses heavily, and it noted itself that it "found nothing in the weights that tells Swedish, Norwegian and Danish apart".

\* *Max-effort follow-up (post hoc, identical prompt): after observing the short sessions, the evaluation was repeated at maximum reasoning effort. This changed the picture for the positive control. On Python model G, the LLM worked for 32 minutes (about 190k output tokens) and answered **python with probability 0.97**. Its reasoning: weight structure tied to all 28 ASCII symbols, but the five extra letters å ä ö æ ø "at pure noise level, so they essentially never appeared in the training text", which rules out Scandinavian Wikipedia. The untrained model again received near-uniform probabilities (top 0.253). The session for Python model H, and therefore any Scandinavian model, could not be completed before the usage limit, so this follow-up rests on a single trained model.*

#### Interpretation and Limitations
At default effort the hypothesis could not be tested: near-chance accuracy on Swedish, Norwegian and Danish only supports H if the LLM succeeds on Python, and it did not. The max-effort follow-up suggests that this failure came from analysis depth rather than from the weights: given far more effort, the LLM identified a Python model confidently and for a sound reason (one model only). Notably, that reason is the same kind of cue as in the paper's Experiment 3: which characters the model has seen at all. Python uses 28 symbols and the Scandinavian corpora about 31, so this cue separates code from prose, but by design it cannot separate Swedish, Norwegian and Danish, which all use the same number of letters. Whether the LLM finds anything beyond this cue for the closely related languages is the real test of H, and it remains open: no Scandinavian model could be analysed at maximum effort. **H is therefore neither supported nor refuted.**

Possible reasons for the default-effort failure:
- **Analysis depth.** Confirmed as decisive for the Python control: under a minute of analysis failed, 32 minutes succeeded. That depth is also expensive (about 190k output tokens for the successful run).
- **A different editor.** Claude Opus 5.5 is not GPT-6 Astra.
- **Less training.** 65M vs. 1B characters seen, which may leave weaker, less readable structure.
- **Small sample.** 8 models at default effort and a single model at maximum effort, so all estimates are very uncertain.
- **Unanalysed baseline.** The non-LLM weight-statistics baseline is included in the repository (`results/baseline.json`) but was not analysed in this report due to time.

This also sharpens the critique of Experiment 3: its success may depend more on the editor, prompt and training scale than the blog post suggests.

#### Recommendation
Based on these results, SE.LLMA should **not** invest in weight-reading as a way to audit or locate Scandinavian language behaviour at this point. At default effort, an off-the-shelf frontier model could not separate even code from prose in a 7M-parameter model. At maximum effort it could, but through a coarse cue (which characters were ever seen) that says nothing about closely related languages, at a cost of roughly half an hour and about 190k output tokens per model. If SE.LLMA explores weight-space methods, the paper's strongest evidence points to removing localized failures, such as drift into English, which is cheaper to test because the effect can be measured directly.

The most informative next experiment would finish the max-effort runs on the Scandinavian models and remove the coarse cue: train all models on text restricted to the same character set (or mask the rare letters), so that only genuine language structure (word patterns, grammar) remains to be read, and run every model at the same, high analysis budget.

### AI Usage
Claude (Anthropic; Claude Opus 5.5 in the Claude app) was used as an assistant throughout: discussing and fact-checking the paper critique against the original post, designing the experiment, writing the code (data preparation, training, prompt generation, evaluation harness, scoring) and editing this text. The research questions, critique points, the choice of hypothesis and all final judgements are my own. Claude Opus 5.5 via Claude Code was also the editor model under test, which is worth noting since the same model family helped design the experiment; the evaluated sessions received only the blinded weights and the prompt. Details are in `AI_USAGE.md`.
