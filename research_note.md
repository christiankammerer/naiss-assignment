## Can an LLM tell closely related languages apart from a model's weights?
*Research note for the NAISS / SE.LLMA candidate assignment. Full report: `report.pdf`.*

### Critique of the Paper
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026) uses an LLM (GPT-6 Astra) to diagnose networks from their weights, edit them, and write new ones.

**Strongest evidence.** In Experiment 3, Astra found every copying head of three 7.2M-parameter character models from the weights alone, which was verified against measurements from running the models. In Experiment 1 it named the game of four policy networks without options; in Experiment 4 a single tool-free edit took a car policy from 0% to 100% success.

**Main limitations.**
1. **Reading vs. feedback.** Pure reading is only shown on small networks (up to 172k weights in context, 7.2M with Python). The LLM-scale results rely on running the model (Go hill-climbs on practice games, Qwen's gains only appear at 50 forward passes, sycophancy needed attention patterns). At scale this is closer to LLM-guided search than native reading.
2. **Experiment 3 is easier than it looks.** The prompt lists the three candidate languages (chance = 1/3), which differ strongly in repetition and alphabet size, exactly the cues Astra used. Its confidence was still only 0.48–0.62.
3. **Experiment 8 removes a failure rather than adding a capability.** Qwen's math gains came from breaking reasoning loops (93% of base answers hit the token limit); on Gemma 3 270M, which rarely loops, there was no gain.

There are no random-edit baselines, mechanistic explanations go unverified outside Experiment 3, and the post is not peer-reviewed; the released logs are a strength. **Conclusion:** a real but narrow ability to read small networks and remove localized failures; native reading at LLM scale and adding capabilities are not yet supported.

### Hypothesis
**H:** A frontier LLM can identify a model's training corpus from its middle blocks when the corpora are very dissimilar (Python vs. prose), but for closely related languages (Swedish, Norwegian Bokmål, Danish) its accuracy drops toward chance.

**Why it matters for SE.LLMA.** Swedish models can drift into Norwegian or Danish. Language-specific neurons exist (Tang et al., 2024) and are usually found by running the model; reading them from the weights would be a complementary route. If an LLM cannot identify a whole single-language network's language, it is unlikely to isolate Scandinavian-language neurons in a mixed model. H is supported by high Python accuracy with near-chance sv/no/da accuracy, and challenged by reliable sv/no/da identification grounded in real weight features.

### Experiment
- **Models:** character transformers with the paper's exact Experiment 3 architecture (4 blocks, 6 heads, d = 384; 7.16M parameters), trained on Python (CodeParrot) and Swedish, Norwegian and Danish Wikipedia. Each corpus is 30M characters, normalized as in the paper, using a shared 33-character vocabulary. The three Scandinavian languages use the same number of letters. There are 2 seeds per corpus (8 models), with the same initialization within a seed.
- **Blinding:** input and output layers removed, shuffled neutral names, answer key kept separate.
- **Evaluation:** Claude Opus 5.5 via Claude Code (19 Nov 2026), one fresh session per model, using the paper's prompt adapted to four corpora. Python analysis of the weights was allowed, forward passes were not. The harness was tested on a separate, never-evaluated model, and the prompt was frozen before evaluation.
- **Controls:** Python models as the positive control, an untrained model as the negative control, and a weight-statistics baseline (`baseline.py`).

### Results
**The models learned their corpora.** Each model has the lowest validation loss on its own corpus (Python 0.89 vs. ~3.2 on the others; Scandinavian models ~1.27 on their own language vs. 1.74–2.29 on the others; Norwegian and Danish are closest to each other).

**At default effort, the LLM could not read it from the weights.**

| | Correct | Chance | p (one-sided) |
|---|---|---|---|
| All 8 trained models | 1/8 | 25% | 0.90 |
| Python (positive control) | 0/2 (both called Danish) | 25% | — |
| Swedish / Norwegian / Danish | 1/6 | 33% | 0.91 |

Confidence was uninformative (mean top probability 0.33 when correct, 0.32 when wrong). On the untrained model, the LLM recognized random initialization and stayed near-uniform (top 0.26), so it did not bluff. Sessions were short: 38 s to 3 min, at most 9 of 100 allowed code runs.

\* **Max-effort follow-up (post hoc, same prompt):** on one Python model the LLM worked for 32 minutes and answered *python, p = 0.97*. Its reasoning was that the five letters å ä ö æ ø leave no trace in the weights, so they never occurred in training. The untrained model stayed near-uniform (top 0.253). The usage limit stopped all further max-effort runs, including every Scandinavian model.

### Interpretation and Limitations
At default effort H could not be tested, because even the easy Python control failed. The single max-effort run suggests that the failure came from analysis depth (minutes vs. half an hour and about 190k output tokens per model), not from the weights. Its winning cue, though, is coarse: *which characters were ever seen*, the same kind of cue as in the paper's Experiment 3. That cue separates code (28 symbols) from Scandinavian prose (~31) but by design cannot separate Swedish, Norwegian and Danish. Whether an LLM finds anything finer is the real test of H, and it remains open: **H is neither supported nor refuted.**

The main limitations: only 8 models, and a single one at maximum effort; a different editor than Astra; about 15× less training than the paper (65M vs. 1B characters); and a weight-statistics baseline that was computed but not analysed due to time.

### Recommendation
SE.LLMA should **not** invest in weight-reading for auditing or locating Scandinavian language behaviour now. At default effort a frontier LLM could not even separate code from prose. At maximum effort it could, but through a coarse cue that says nothing about closely related languages, at substantial cost per model. Weight-space methods look more promising for removing localized failures (e.g., drift into English), where the effect is directly measurable.

**Next experiment:** finish the max-effort runs on the Scandinavian models, and remove the coarse cue by restricting all corpora to the same character set, so that only genuine language structure is left to read. Run every model at the same high analysis budget.

### AI Usage
Claude (Opus 5.5, Claude app) assisted with the critique, experiment design, code and editing. The research decisions and judgements are my own. Claude Opus 5.5 via Claude Code was also the model under test. See `AI_USAGE.md`.
