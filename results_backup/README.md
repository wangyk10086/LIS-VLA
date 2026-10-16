# Raw experiment results (LIS-VLA)

Copied from the RunPod network volume on 2026-10-16 before it was deleted (68 files).

- `results/phase0/` - System-1 baselines (LIBERO-Goal/10, LIBERO-Plus Goal/10), per-episode JSONL
- `results/pilot/`, `results/step1/`, `results/step2/` - test-time selection runs on the 100 paired LIBERO-Plus episodes, plus summaries from `tools/compare_eval.py`
- `results/step3/` - counterfactual branching summary (`tools/branch_summary.py`) and logs
- `data/episode_metadata.json` - per-episode metadata of the scorer-training rollouts (success, task, perturbation scale, executed candidate per query) and of the branching episodes (branch indices, outcomes, end steps). The large feature arrays (latents, candidate chunks) were not kept.

Not included: model weights, tokenised training data and the full `.npz` arrays (too large for git); they can be regenerated with the scripts in `rynnvla-002/exps_lis/runpod/`.
