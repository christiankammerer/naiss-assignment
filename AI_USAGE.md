# AI usage

**Claude (Anthropic, Claude Opus 5.5 in the Claude app)** was used as an assistant throughout the assignment:
- **Paper critique:** discussing the paper and checking my critique points against the original blog post and its released prompt.
- **Experiment design:** helping turn my idea (testing Experiment 3 on closely related languages) into a concrete design with controls, blinding and metrics.
- **Code:** writing the pipeline (`prepare_data.py`, `train.py`, `baseline.py`, `make_prompts.py`, `run_llm.sh`, `parse_answers.py`, `score.py`), which I ran and checked.
- **Writing:** editing and condensing the research note and report.

The research questions, the main critique points, the choice of hypothesis, the decision to run a post-hoc max-effort follow-up and all final judgements are my own.

**Claude Opus 5.5 via Claude Code (headless mode, 19 November 2026)** was also the *editor model under test*: it analysed the blinded weights in the experiment itself. The same model family therefore helped design the experiment. To keep the evaluation independent, every evaluated session started fresh in an isolated folder containing only the blinded weights and the prompt, with no access to the design discussion, the code or the answer key. Full transcripts are in `transcripts/`.
