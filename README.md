# Can an LLM tell closely related languages apart from a model's weights?

A small follow-up to Experiment 3 of *Sparks of Weight-Space Programming* (Takehi et al., 2026).
Character-level transformers (paper-matched: 4 blocks, 6 heads, d = 384, ~7.2M parameters) are trained
on Python, Swedish, Norwegian (Bokmål) and Danish. Input and output layers are removed, the models are
blinded, and an LLM is asked to identify the training corpus from the middle blocks alone, without
running the network.

Write-up: `research_note.md`, `report.pdf`. AI usage: `AI_USAGE.md`.

## Replicate

```bash
cd sellma_experiment
pip install torch numpy scipy datasets huggingface_hub

# 1. Download the corpora (Python, sv, no, da Wikipedia; 30M chars each) -> data/
python prepare_data.py

# 2. Train 8 models (4 corpora x 2 seeds) + dev model + random control; export blinded weights (GPU, ~60–90 min)
python train.py --iters 50 --quick        # optional smoke test; delete runs/ blinded/ dev/ afterwards
python train.py                           # finished models are skipped on restart

# 3. Non-LLM baseline -> results/baseline.json
python baseline.py

# 4. Build one prompt folder per blinded model -> llm_eval/, llm_eval_dev/
python make_prompts.py

# 5. Copy model folders OUTSIDE the repo so the LLM cannot see the key or code
mkdir -p ~/llm_runs ~/llm_dev
cp -r llm_eval/model_* llm_eval/control_random ~/llm_runs/
cp -r llm_eval_dev/model_dev ~/llm_dev/

# 6. Run Claude Code headless, one fresh session per model (needs `claude` logged in, empty ~/.claude/CLAUDE.md)
./run_llm.sh ~/llm_dev options 1          # harness test on the dev model; freeze the prompt after this
./run_llm.sh ~/llm_runs options 3         # evaluation, default effort, 3 in parallel
./run_llm.sh ~/llm_runs open 3            # optional: no candidate corpora listed

# 7. Parse answers -> results/llm_answers.csv (flags transcripts mentioning other files;
#    check them by hand for forward passes and fill unparsed rows), then score -> results/llm_scores.json
python parse_answers.py ~/llm_runs
python score.py
```

Max-effort follow-up (same prompt, separate folders):

```bash
mkdir -p ~/llm_runs_max
cp -r llm_eval/model_G llm_eval/model_H llm_eval/control_random ~/llm_runs_max/
EFFORT=max ./run_llm.sh ~/llm_runs_max options 1
```

On Colab, prefix steps 1–4 with `!` and use a T4 GPU. To survive disconnects, pass
`--out /content/drive/MyDrive/sellma` to `train.py` and `--dir` with the same path to the later scripts.
Short on time: `python train.py --iters 2500`.

Results from the original run are in `sellma_experiment/results/` (`*_default.*` = default effort),
transcripts in `transcripts/`. Answer key: `sellma_experiment/blinded/KEY_DO_NOT_SHARE.json`
(fixed `--blind_seed 42`, so letters map the same way on rerun). Model weights are not in the repo.

## Design

- **Hypothesis:** identification works for very different corpora (Python vs. prose) but drops toward
  chance for Swedish / Norwegian / Danish.
- **Positive control:** Python models in the same blinded pool (the paper's easy case).
- **Negative control:** `control_random`, untrained, same shape. Does the LLM still answer confidently?
- **Non-LLM alternative:** `baseline.py`, effective rank and norms of the weights, plus a
  leave-one-out nearest-neighbour classifier (an upper bound: it sees the other models' labels).
- **Sanity check of the models:** each model's validation loss on every corpus (it should fit its own
  corpus best), and measured copy (induction) heads per model (`train.py`, `measure_heads`), reported
  for context only; the LLM is not asked about heads.
- **Blinding:** shuffled neutral names; key in `blinded/KEY_DO_NOT_SHARE.json`.
- **Two seeds per corpus:** checks that the LLM detects the language rather than quirks of one run.
  Within a seed, all corpora share the same initialization, so differences come only from the data.

## Reproducibility

Dataset revisions are pinned and recorded in `sellma_experiment/data/manifest.json`; all settings, seeds,
losses, measured head behaviour, library versions and the GPU are recorded in
`sellma_experiment/results/train_summary.json`.

## Deviations from the paper

Fewer training characters (default 65M seen per model vs. 1B), smaller corpora (30M characters each), a
33-character vocabulary (adds å ä ö æ ø), no "shuffled" option, no head-by-head question, a different
editor LLM (Claude Opus 5.5 instead of GPT-6 Astra), and code is run by the LLM itself rather than by a
harness.
