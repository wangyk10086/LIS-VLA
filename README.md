# LIS-VLA: Latent Imagination Search VLA

> Built on [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002), LIS-VLA turns the one-way "perception → action" pipeline into a **closed-loop search in latent space**: from a single backbone forward pass it proposes N candidate action chunks, imagines the future each one leads to in latent space, scores those futures against the language instruction, and executes the best one.

![LIS-VLA architecture](rynnvla-002/assets/lis_vla_architecture.png)

<!-- RESULTS:START -->
## Results

**Summary.** On LIBERO-Plus robot-initial-state perturbations, LIS did **not** improve over the unmodified RynnVLA-002 policy (System 1) under any of five scorers, and the two variants that overrode System 1 most often were significantly worse. Counterfactual branching shows that, for a selector that replaces **one** action chunk and then hands control back to System 1, a perfect scorer could gain at most about 7.0 points. Interventions spanning several consecutive decision points were not evaluated (see *Open question* below). Paper: [`paper/main.pdf`](paper/main.pdf) (workshop version: [`paper/corl_tag/main_tag.pdf`](paper/corl_tag/main_tag.pdf)); raw per-episode results: [`results_backup/`](results_backup/).

### 1. System 1 baselines (released RynnVLA-002 checkpoints)

| Benchmark | Episodes | Success (%) | 95% CI |
|---|---:|---:|---|
| LIBERO-Goal (standard) | 100 | 98.0 | [93.0,&nbsp;99.4] |
| LIBERO-Plus Goal | 834 | 56.8 | [53.4,&nbsp;60.2] |
| &nbsp;&nbsp;LIBERO-Plus Goal / Objects Layout | 425 | 60.9 | [56.2,&nbsp;65.5] |
| &nbsp;&nbsp;LIBERO-Plus Goal / Robot Initial States | 409 | 52.6 | [47.7,&nbsp;57.4] |
| LIBERO-10 (standard) | 100 | 94.0 | [87.5,&nbsp;97.2] |
| LIBERO-Plus 10 | 705 | 58.9 | [55.2,&nbsp;62.4] |
| &nbsp;&nbsp;LIBERO-Plus 10 / Objects Layout | 312 | 71.8 | [66.6,&nbsp;76.5] |
| &nbsp;&nbsp;LIBERO-Plus 10 / Robot Initial States | 393 | 48.6 | [43.7,&nbsp;53.5] |

### 2. Test-time selection vs. System 1 (same 100 paired episodes)

LIBERO-Plus Goal, category *Robot Initial States* (first 200 variants, shard 0 of 2 → 100 episodes); System 1 solves 50/100. 16 candidates per decision (candidate 1 = System 1). Override = share of decisions where a non-System-1 chunk was executed. Wins = episodes solved only by the variant / only by System 1. CI: paired bootstrap; p: exact McNemar.

| Selection rule | Override | Success | Δ vs. S1 (95% CI) | Wins | p |
|:---|---:|---:|:---|:---:|---:|
| System 1 only | 0% | 50/100 | – | – | – |
| Language score, argmax | 71.8% | 37/100 | **−13**&nbsp;[−24,&nbsp;−1] | 11/24 | 0.041 |
| Language score + gate (p90) | 7.9% | 44/100 | **−6**&nbsp;[−13,&nbsp;+1] | 3/9 | 0.146 |
| Language score + gate (p75) + prior | 0.2% | 50/100 | **0**&nbsp;[0,&nbsp;0] | 0/0 | 1.000 |
| Learned value, imagined latent | 3.4% | 49/100 | **−1**&nbsp;[−5,&nbsp;+3] | 2/3 | 1.000 |
| Learned Q(state, action) | 44.1% | 22/100 | **−28**&nbsp;[−39,&nbsp;−17] | 5/33 | 4.3e−6 |

*Language score*: cosine between the imagined future latent and the instruction, minus a smoothness cost. *Gate*: keep System 1 unless the best candidate beats it by the 90th / 75th percentile of the score gap; *prior*: bonus for staying close to System 1. *Learned value / Q*: scorers trained on our own outcome-labelled rollouts, System 1 overridden only if predicted success is higher by 0.05. Mean query latency was 0.8–1.2 s for all variants.

### 3. Why the scorers fail

**Language-similarity evaluator** (offline, 300 held-out LIBERO-Goal demonstration states):

| Diagnostic | Value |
|---|---:|
| P(score(expert) > score(expert + N(0, 0.05²) noise)) | 91.4% |
| P(score(expert) > score(expert + N(0, 0.1²) noise)) | 96.9% |
| P(score(expert) > score(expert + N(0, 0.2²) noise)) | 99.1% |
| P(score(expert) > score(expert + N(0, 0.3²) noise)) | 99.8% |
| Spearman(score, closeness to expert) among proposer candidates | 0.063 |
| Argmax closer to expert than System 1 | 0.0% |
| Argmax farther from expert than System 1 | 76.3% |
| MSE to expert: System 1 / argmax | 0.00352 / 0.01569 |
| Candidates scored above System 1 (mean, of 15) | 4.87 |
| Best-vs-System-1 score gap in s.d., percentiles 50/75/90/95 | 0.95 / 1.83 / 2.58 / 2.89 |

**Outcome-trained scorers** (561 rollouts on LIBERO-Goal training tasks with our own joint perturbations; rollout success 51.2%, 89.3% without perturbation; a random candidate was executed at 6,613 of 22,419 decisions):

| Scorer | Train / val episodes | Val AUC (episode success) | Ranking agrees with outcome | Candidates above System 1 (of 15) |
|---|---:|---:|---:|---:|
| Value of imagined latent | 477 / 84 | 0.797 | 48.6% | 7.79 |
| Q(z, ctx, g, A) | 477 / 84 | 0.915 | 37.6% | 9.74 |

The scorers predict *whether* an episode will succeed well, but their preference between two chunks from the same state agrees with the outcome at or below chance.

### 4. Head-room of a single-chunk intervention (counterfactual branching)

115 LIBERO-Goal training-task episodes with our own initial-joint perturbations (never LIBERO-Plus variants). At up to three snapshots per failed episode (one per successful episode) the simulator is restored and four branches are played to the end: System 1's own chunk, and three random proposer candidates, each executed for **one** chunk and followed by System 1.

| Quantity | Count | % | 95% CI |
|---|---:|---:|---|
| System 1 episode success | 55/115 | 47.8 | [38.9,&nbsp;56.9] |
| System 1 branch reproduces the original outcome | 216/235 | 91.9 | [87.7,&nbsp;94.8] |
| Failure branch points rescued by ≥1 of 3 alternatives | 8/163 | 4.9 | [2.5,&nbsp;9.4] |
| &nbsp;&nbsp;early branch point | 5/47 | 10.6 | [4.6,&nbsp;22.6] |
| &nbsp;&nbsp;middle branch point | 2/56 | 3.6 | [1.0,&nbsp;12.1] |
| &nbsp;&nbsp;late branch point | 1/60 | 1.7 | [0.3,&nbsp;8.9] |
| Failed episodes rescued at any branch point | 8/60 | 13.3 | [6.9,&nbsp;24.2] |
| Oracle episode success (System 1 + rescued) | 63/115 | 54.8 | [45.7,&nbsp;63.6] |
| Single random alternative succeeds, from a failure state | | 2.7 | |
| Single random alternative succeeds, from a success state | | 93.1 | |

### Open question: multi-step interventions

Each branch above changes one chunk and then returns control to System 1, which may undo a helpful intervention; recovery may also need several coordinated chunks. The numbers therefore bound **single-chunk** selection only, not repeated selection. `rynnvla-002/tools/branch_horizons.py` (with `branch_horizons_summary.py` and `exps_lis/runpod/step4_horizons.sh`) implements branching with alternatives in control for 1, 2, 4 or all remaining decision points; it is tested on a mock environment (`tests/test_branch_horizons.py`) but **has not been run** on LIBERO.
<!-- RESULTS:END -->

## Core idea

| Component | Place in figure | Implementation (`rynnvla-002/model/lis/modules.py`) |
|---|---|---|
| Encoder → $z_t$ | Input Layer | Chameleon backbone (RynnVLA-002). $z_t$ = `ObservationEncoder`: Perceiver-style attention pooling over the **input embeddings of the VQ tokens** of each current view, giving Q latent tokens (16×512 by default). It depends only on the observation, so the same encoder also produces the target $z_{t+k}$ from the future frame. |
| Action Proposer | System 2 | `FlowMatchingProposer`: a rectified-flow head conditioned on the action-query features of the RynnVLA action head plus backbone context tokens $c_t$; samples N action chunks $A^1..A^N$ in parallel with 10 Euler steps. Candidate #0 is always the deterministic System 1 output (the original L1 action head), so the selected chunk never scores worse than not searching. |
| Latent World Model | System 2 | `LatentWorldModel`: $\hat z_{t+k} = z_t + D([z_t; c_t; A])$, a Transformer dynamics head whose output layer is zero-initialised (an untrained model predicts "nothing changes"). No pixels are generated; k = H (chunk length). |
| Language-Guided Evaluator | System 2 | `LanguageGuidedEvaluator`: $R = \cos(\phi(\hat z_{t+k}), \psi(g))$, where $g$ pools the backbone hidden states of the instruction tokens. |
| Selection | Decision | $A^* = \arg\max_i \; R^i - \text{Cost}(A^i)$, with optional CEM refinement rounds; the whole chunk is executed (same action chunking as RynnVLA-002). |

### Search objective

$$
A_t^* = \arg\max_{A \in \{A^i\}_{i=1}^N} \; \mathrm{Sim}\big(\phi(\hat z_{t+k}),\, \psi(g)\big) - \mathrm{Cost}(A), \qquad \hat z_{t+k} = \mathcal{WM}(z_t, c_t, A)
$$

$$
\mathrm{Cost}(A) = \lambda_s \frac{1}{H-1}\sum_{i}\lVert a_{i+1}^{\text{arm}} - a_i^{\text{arm}}\rVert^2 + \lambda_p \frac{1}{H}\lVert A - A^{\text{S1}}\rVert^2
$$

Notes: the world model is currently deterministic, so the expectation $\mathbb{E}_{\hat z\sim\mathcal{WM}}$ reduces to a single prediction. The smoothness term is not applied to the gripper dimension (gripper commands are meant to switch). $A^{\text{S1}}$ is the System 1 output; $\lambda_p$ defaults to 0.

### Training objectives (all in latent space)

An LIS training sample appends the future frame reached after executing the chunk to the VLA sequence, so a single sequence supervises every component:

```
human: What action should the robot take to {task}? <|state|> <|image|>×(his·V)
gpt  : <|action|>×H  <|image|>×V        ← third-person + wrist views at time t+H
```

| Loss | Purpose | Default weight |
|---|---|---|
| `lis_fm` | Flow matching: $\lVert v_\theta(x_\tau,\tau)-(a-x_0)\rVert^2$ with logit-normal τ | 1.0 |
| `lis_wm` | $1-\cos(\hat z_{t+H}, \mathrm{sg}[z_{t+H}])$ per latent token (JEPA-style stop-grad target) | 1.0 |
| `lis_wm_nce` | In-batch InfoNCE: a prediction must identify its own true future (prevents collapse) | 0.1 |
| `lis_var` | VICReg variance term (prevents collapse) | 0.1 |
| `lis_lang_nce` | Multi-positive InfoNCE between real/imagined future latents and instruction embeddings (samples with the same instruction are positives) | 0.1 |
| `lis_prog` | Progress ranking: $R(z_{t+H}, g) > R(z_t, g)$ | 0.05 |
| `lis_rank` | Ranking in imagination: the imagined future of the expert chunk beats those of counterfactual chunks (noised / time-reversed / mismatched within the batch) | 0.1 |

All losses reach the original trainer through the model's `additional_loss_dict`, so **the trainer itself is unchanged** and each term shows up in the logs. The original next-token CE (including generation of the t+H frame) and the continuous-action L1 loss are kept as they are.

## What changed vs. RynnVLA-002

New files:

| File | Contents |
|---|---|
| `rynnvla-002/model/lis/tokens.py` | Parses RynnVLA token sequences (positions of instruction / state / current frames / action blocks / future frames) |
| `rynnvla-002/model/lis/modules.py` | Observation/context/language encoders, proposer, latent world model, evaluator, losses, search |
| `rynnvla-002/model/modeling_lis_vla.py` | `ChameleonXLLMXForConditionalGeneration_LIS`, `LISVLAConfig`, `load_lis_model` |
| `rynnvla-002/pretrain_solver_lis_vla.py` | Training entry point (subclasses the original solver, adds all `--lis_*` options) |
| `rynnvla-002/eval_solver_libero_lis.py`, `libero_util/lis_utils.py` | LIBERO evaluation (search + per-step search log `lis_search_log.jsonl`) |
| `rynnvla-002/data/lis_conv_generation.py`, `data/pretoken_lis_model.py` | LIS training-data generation and pre-tokenisation |
| `rynnvla-002/exps_lis/*.sh`, `configs/lis/*.yaml` | Data preparation / training / evaluation scripts |
| `rynnvla-002/tests/test_lis_vla.py` | CPU smoke tests (tiny Chameleon) |

Two corrections to existing behaviour (both implemented in the subclass; original files untouched):

1. **Attention mask**: the original `generate_att_mask_3` uses the *last image* as the reference for deciding whether action blocks mask each other. LIS samples have future frames after the action blocks, which would make the chunk's action blocks visible to each other during training, unlike at inference. The LIS version uses the *last image before the first action block* as the reference. For existing VLA samples, world-model samples and inference prompts the mask is **identical** (verified by tests); it only differs on LIS samples.
2. **Action-head padding**: the original `ActionHead` lets action queries attend to zero-padded positions when context lengths differ within a batch; the LIS version masks them. Parameter names and shapes are unchanged, so **original checkpoints load directly**, and batch-1 inference is numerically identical to the original (verified by tests).

## Usage

The environment is the same as RynnVLA-002 (`requirements.txt`, torch 2.2.0 / transformers 4.43.0). First complete Step 1 of the LIBERO data preparation in [README_RynnVLA-002.md](README_RynnVLA-002.md) (regenerating images/actions/states), and download the tokenizer and RynnVLA-002 checkpoints into `rynnvla-002/ckpts/`.

### 1. Prepare LIS data

```bash
cd rynnvla-002/exps_lis
bash prepare_data.sh goal 5        # goal/object: H=5; spatial/10: H=10
```

This generates and pre-tokenises the LIS samples; if tokens for the original world-model and VLA data already exist, they are mixed into `processed_data/concate_tokens/libero_goal_..._lis_awm.json`.

### 2. Train

```bash
# Freeze RynnVLA-002 and train System 2 only (cheapest; recommended first)
bash train_lis.sh goal 5 lis_only ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
# Also fine-tune the continuous action head
bash train_lis.sh goal 5 lis_and_head ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
# Full fine-tuning (same setting as RynnVLA-002 training)
bash train_lis.sh goal 5 all ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
```

When starting from a checkpoint without LIS weights, `load_lis_model` initialises the `lis.*` parameters automatically. `--lis_detach_backbone true` (default) stops LIS losses from back-propagating into the backbone.

### 3. Evaluate

```bash
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9            # search with N=16
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9 32 2        # N=32 + 2 CEM rounds
bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9 16 0 0      # search off = System 1 ablation
```

### Python API

```python
from model import load_lis_model
model = load_lis_model(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
chunk = model.generate_lis_action(input_ids, generation_config, num_candidates=16, cem_iters=0)  # (H, 7), normalised space
model.last_search_info   # best_score / best_semantic / best_cost / score_std / prior_score ...
```

## Size and cost

- System 2 parameters: **about 50.1M** at 7B dimensions (obs 9.0M / ctx 5.3M / lang 5.3M / proposer 13.7M / world model 12.9M / evaluator 3.9M), roughly 0.7% of the backbone.
- Search needs **no extra backbone forward pass** and decodes no pixels: all candidates are evaluated in one batched pass in the 512-d latent space.
- Reference: on 2 CPU threads in fp32, one plan takes about 0.43 s for N=16, 0.74 s for N=32 and 0.63 s for N=16 with 2 CEM rounds; GPU latency has not been measured yet.

## Tests

```bash
cd rynnvla-002
python -m pytest -q tests/test_lis_vla.py
```

Coverage: sequence parsing; attention-mask equivalence with the original and the fix on LIS samples; training forward/backward (every LIS module receives gradients); `lis_fm` / `lis_wm` / `lis_lang_nce` decrease within 60 steps; end-to-end `generate_lis_action` (including CEM and search off); state-dict compatibility and numerical equivalence with the original `ActionHead`; loading from a checkpoint without LIS weights.

## Status

Trained (System 2 only, `lis_only`, 5,000 steps) and evaluated with the 7B RynnVLA-002 LIBERO-Goal checkpoint; see **Results** above. The trained LIS weights were not kept.

Caveat: the evaluator learns only from expert demonstrations (expert chunks vs. counterfactual chunks); the results above show that this is not enough to rank plausible candidates.

## Acknowledgements and license

Based on [RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002) (Apache-2.0); the original README is kept as [README_RynnVLA-002.md](README_RynnVLA-002.md). This repository is also released under Apache-2.0; see `LICENSE` and `NOTICE`.
