"""Write the experimental results into the top-level README.md (between the RESULTS markers).
Every number comes from paper/data/*.json, the same files the paper is generated from.

    python paper/make_readme_results.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
README = os.path.join(HERE, "..", "README.md")
J = lambda f: json.load(open(os.path.join(DATA, f)))

p0, sel, diag, s3 = J("phase0_summary.json"), J("selection_runs.json"), J("evaluator_diag.json"), J("step3_summary.json")
pat = re.compile(r"(\d+)/(\d+) = ([\d.]+)% \[([\d.]+), ([\d.]+)\]")


def frac(s):
    k, n, p, lo, hi = pat.match(s).groups()
    return f"{k}/{n}", f"{p}", f"[{lo},&nbsp;{hi}]"


def nb(x):
    """Signed number that never wraps: Unicode minus, no spaces."""
    x = str(x)
    return "0" if x in ("+0", "-0") else x.replace("-", "\u2212")


def ci(lo, hi):
    """Interval kept on one line in GitHub tables."""
    return f"[{nb(lo)},&nbsp;{nb(hi)}]"


L = []
w = L.append
w("## Results")
w("")
w("**Summary.** On LIBERO-Plus robot-initial-state perturbations, LIS did **not** improve over the unmodified "
  "RynnVLA-002 policy (System 1) under any of five scorers, and the two variants that overrode System 1 most often "
  "were significantly worse. Counterfactual branching shows that, for a selector that replaces **one** action chunk "
  "and then hands control back to System 1, a perfect scorer could gain at most about "
  f"{float(frac(s3['oracle_episode_success_upper_bound'])[1]) - float(frac(s3['system1_success'])[1]):.1f} points. "
  "Interventions spanning several consecutive decision points were not evaluated (see *Open question* below). "
  "Paper: [`paper/main.pdf`](paper/main.pdf) (workshop version: [`paper/corl_tag/main_tag.pdf`](paper/corl_tag/main_tag.pdf)); "
  "raw per-episode results: [`results_backup/`](results_backup/).")
w("")
w("### 1. System 1 baselines (released RynnVLA-002 checkpoints)")
w("")
w("| Benchmark | Episodes | Success (%) | 95% CI |")
w("|---|---:|---:|---|")
names = {"goal_vanilla_s1": "LIBERO-Goal (standard)", "goal_plus_s1": "LIBERO-Plus Goal",
         "long_vanilla_s1": "LIBERO-10 (standard)", "long_plus_s1": "LIBERO-Plus 10"}
for job, label in names.items():
    for cat, r in p0[job].items():
        name = label if cat in ("standard", "ALL") else f"&nbsp;&nbsp;{label} / {cat}"
        w(f"| {name} | {r['n']} | {r['rate']:.1f} | {ci(*r['ci95'].split('-'))} |")
w("")
w("### 2. Test-time selection vs. System 1 (same 100 paired episodes)")
w("")
w("LIBERO-Plus Goal, category *Robot Initial States* (first 200 variants, shard 0 of 2 → 100 episodes); System 1 solves "
  "50/100. 16 candidates per decision (candidate 1 = System 1). Override = share of decisions where a non-System-1 chunk "
  "was executed. Wins = episodes solved only by the variant / only by System 1. CI: paired bootstrap; p: exact McNemar.")
w("")
short = {"pilot": "Language score, argmax", "A_gate_p90": "Language score + gate (p90)",
         "B_gate_p75_trust": "Language score + gate (p75) + prior", "imag": "Learned value, imagined latent",
         "direct": "Learned Q(state, action)"}
w("| Selection rule | Override | Success | Δ vs. S1 (95% CI) | Wins | p |")
w("|:---|---:|---:|:---|:---:|---:|")
w("| System 1 only | 0% | 50/100 | – | – | – |")
for r in sel["runs"]:
    p = f"{r['p']:.1e}".replace("e-0", "e\u2212") if r["p"] < 0.01 else f"{r['p']:.3f}"
    d = f"{r['diff']:+.0f}".replace("+0", "0") if r["diff"] == 0 else nb(f"{r['diff']:+.0f}")
    lo, hi = f"{r['ci'][0]:+.0f}", f"{r['ci'][1]:+.0f}"
    w(f"| {short[r['id']]} | {r['override']:.1f}% | {r['new']}/100 | **{d}**&nbsp;{ci(lo, hi)} | "
      f"{r['new_wins']}/{r['base_wins']} | {p} |")
w("")
w("*Language score*: cosine between the imagined future latent and the instruction, minus a smoothness cost. "
  "*Gate*: keep System 1 unless the best candidate beats it by the 90th / 75th percentile of the score gap; *prior*: "
  "bonus for staying close to System 1. *Learned value / Q*: scorers trained on our own outcome-labelled rollouts, "
  "System 1 overridden only if predicted success is higher by 0.05. Mean query latency was 0.8–1.2 s for all variants.")
w("")
w("### 3. Why the scorers fail")
w("")
w(f"**Language-similarity evaluator** (offline, {diag['samples']} held-out LIBERO-Goal demonstration states):")
w("")
w("| Diagnostic | Value |")
w("|---|---:|")
for k, v in diag["pairwise_acc_gt_vs_noise"].items():
    w(f"| P(score(expert) > score(expert + N(0, {k}²) noise)) | {100 * v:.1f}% |")
w(f"| Spearman(score, closeness to expert) among proposer candidates | {diag['spearman_score_vs_closeness']:.3f} |")
w(f"| Argmax closer to expert than System 1 | {100 * diag['argmax_closer_than_system1']:.1f}% |")
w(f"| Argmax farther from expert than System 1 | {100 * diag['argmax_farther_than_system1']:.1f}% |")
w(f"| MSE to expert: System 1 / argmax | {diag['mse_system1_vs_expert']:.5f} / {diag['mse_argmax_vs_expert']:.5f} |")
w(f"| Candidates scored above System 1 (mean, of 15) | {diag['system1_rank_mean']:.2f} |")
g = diag["gap_z_percentiles_50_75_90_95"]
w(f"| Best-vs-System-1 score gap in s.d., percentiles 50/75/90/95 | {g[0]} / {g[1]} / {g[2]} / {g[3]} |")
w("")
sc = sel["scorers"]
w(f"**Outcome-trained scorers** ({sc['rollouts']} rollouts on LIBERO-Goal training tasks with our own joint perturbations; "
  f"rollout success {100 * sc['rollout_success']:.1f}%, {100 * sc['rollout_success_sigma0']:.1f}% without perturbation; "
  f"a random candidate was executed at {sc['explored_queries']:,} of {sc['queries']:,} decisions):")
w("")
w("| Scorer | Train / val episodes | Val AUC (episode success) | Ranking agrees with outcome | Candidates above System 1 (of 15) |")
w("|---|---:|---:|---:|---:|")
for k, lab in (("imag", "Value of imagined latent"), ("direct", "Q(z, ctx, g, A)")):
    r = sc[k]
    w(f"| {lab} | {r['episodes_train']} / {r['episodes_val']} | {r['val_auc_success']:.3f} | "
      f"{100 * r['ranking_agrees_with_outcome']:.1f}% | {r['system1_rank_mean']:.2f} |")
w("")
w("The scorers predict *whether* an episode will succeed well, but their preference between two chunks from the same "
  "state agrees with the outcome at or below chance.")
w("")
w("### 4. Head-room of a single-chunk intervention (counterfactual branching)")
w("")
w(f"{s3['episodes']} LIBERO-Goal training-task episodes with our own initial-joint perturbations (never LIBERO-Plus "
  "variants). At up to three snapshots per failed episode (one per successful episode) the simulator is restored and four "
  "branches are played to the end: System 1's own chunk, and three random proposer candidates, each executed for **one** "
  "chunk and followed by System 1.")
w("")
w("| Quantity | Count | % | 95% CI |")
w("|---|---:|---:|---|")
rows = [("System 1 episode success", "system1_success"),
        ("System 1 branch reproduces the original outcome", "system1_branch_reproduces_main"),
        ("Failure branch points rescued by ≥1 of 3 alternatives", "rescuable_points (>=1 of the alternatives succeeds)")]
for lab, key in rows:
    c, p, ci = frac(s3[key]); w(f"| {lab} | {c} | {p} | {ci} |")
for pos in ("early", "middle", "late"):
    c, p, ci = frac(s3["rescuable_by_position"][pos]); w(f"| &nbsp;&nbsp;{pos} branch point | {c} | {p} | {ci} |")
for lab, key in (("Failed episodes rescued at any branch point", "rescuable_failed_episodes (any point)"),
                 ("Oracle episode success (System 1 + rescued)", "oracle_episode_success_upper_bound")):
    c, p, ci = frac(s3[key]); w(f"| {lab} | {c} | {p} | {ci} |")
w(f"| Single random alternative succeeds, from a failure state | | {s3['random_alternative_success_rate_in_failed_states']:.1f} | |")
w(f"| Single random alternative succeeds, from a success state | | {s3['random_alternative_success_rate_in_successful_states']:.1f} | |")
w("")
w("### Open question: multi-step interventions")
w("")
w("Each branch above changes one chunk and then returns control to System 1, which may undo a helpful intervention; "
  "recovery may also need several coordinated chunks. The numbers therefore bound **single-chunk** selection only, not "
  "repeated selection. `rynnvla-002/tools/branch_horizons.py` (with `branch_horizons_summary.py` and "
  "`exps_lis/runpod/step4_horizons.sh`) implements branching with alternatives in control for 1, 2, 4 or all remaining "
  "decision points; it is tested on a mock environment (`tests/test_branch_horizons.py`) but **has not been run** "
  "on LIBERO.")
block = "\n".join(L)

s = open(README).read()
start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
if start in s:
    s = re.sub(re.escape(start) + ".*?" + re.escape(end), start + "\n" + block + "\n" + end, s, flags=re.S)
else:  # insert before "## Core idea"
    s = s.replace("## Core idea", start + "\n" + block + "\n" + end + "\n\n## Core idea", 1)
# the old status section said there were no results yet
s = re.sub(r"## Status and suggested ablations.*?(?=## Acknowledgements)",
           "## Status\n\nTrained (System 2 only, `lis_only`, 5,000 steps) and evaluated with the 7B RynnVLA-002 LIBERO-Goal "
           "checkpoint; see **Results** above. The trained LIS weights were not kept.\n\n"
           "Caveat: the evaluator learns only from expert demonstrations (expert chunks vs. counterfactual chunks); the "
           "results above show that this is not enough to rank plausible candidates.\n\n", s, flags=re.S)
open(README, "w").write(s)
print("README updated:", len(block), "chars of results")
