"""Turn experiment outputs in paper/data into LaTeX inputs in paper/generated.
Every result number in main.tex comes from here (nothing is typed into the .tex by hand).

    python make_generated.py
Inputs: data/phase0_summary.json, data/selection_runs.json, data/step3_summary.json (if present).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, GEN = os.path.join(HERE, "data"), os.path.join(HERE, "generated")
os.makedirs(GEN, exist_ok=True)
macros = []


def mac(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


# ---------------- baselines (phase 0)
p0 = json.load(open(os.path.join(DATA, "phase0_summary.json")))
g = p0["goal_plus_s1"]
mac("goalVanilla", f"{p0['goal_vanilla_s1']['standard']['rate']:.1f}")
mac("longVanilla", f"{p0['long_vanilla_s1']['standard']['rate']:.1f}")
mac("goalPlusAll", f"{g['ALL']['rate']:.1f}")
mac("goalPlusN", g["ALL"]["n"])
mac("goalPlusRobot", f"{g['Robot Initial States']['rate']:.1f}")
mac("goalPlusLayout", f"{g['Objects Layout']['rate']:.1f}")
mac("longPlusAll", f"{p0['long_plus_s1']['ALL']['rate']:.1f}")
mac("longPlusRobot", f"{p0['long_plus_s1']['Robot Initial States']['rate']:.1f}")

rows = []
for job, label in [("goal_vanilla_s1", "LIBERO-Goal (standard)"), ("goal_plus_s1", "LIBERO-Plus Goal"),
                   ("long_vanilla_s1", "LIBERO-10 (standard)"), ("long_plus_s1", "LIBERO-Plus 10")]:
    for cat, r in p0[job].items():
        name = label if cat in ("standard", "ALL") else f"\\quad {cat}"
        rows.append(f"{name} & {r['n']} & {r['rate']:.1f} & [{r['ci95'].replace('-', ', ')}] \\\\")
open(os.path.join(GEN, "tab_baseline.tex"), "w").write(r"""\begin{table}[t]
\centering
\caption{System~1 (RynnVLA-002, released checkpoints) success rates. 95\% Wilson intervals.}
\label{tab:baseline}
\small
\begin{tabular}{lrrl}
\toprule
Benchmark & Episodes & Success (\%) & 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- selection runs (pilot, step 1, step 2)
sel = json.load(open(os.path.join(DATA, "selection_runs.json")))
rows = []
for r in sel["runs"]:
    p = f"{r['p']:.2g}" if r["p"] < 0.01 else f"{r['p']:.2f}"
    rows.append(f"{r['label']} & {r['override']:.1f} & {r['new']} & {r['diff']:+.0f} & "
                f"[{r['ci'][0]:+.0f}, {r['ci'][1]:+.0f}] & {r['new_wins']}/{r['base_wins']} & {p} \\\\")
open(os.path.join(GEN, "tab_selection.tex"), "w").write(r"""\begin{table*}[t]
\centering
\caption{Test-time selection among 16 candidate chunks versus System~1 alone, on the same 100 LIBERO-Plus Goal episodes with perturbed robot initial states (System~1: 50/100). Override: share of control steps at which a non-System-1 chunk was executed. Wins: episodes solved only by the variant / only by System~1. CI: paired bootstrap; $p$: exact McNemar test.}
\label{tab:selection}
\small
\begin{tabular}{lrrrlrr}
\toprule
Selection rule & Override (\%) & Success & $\Delta$ (pts) & 95\% CI & Wins & $p$ \\
\midrule
System~1 only & 0.0 & 50 & -- & -- & -- & -- \\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
with open(os.path.join(GEN, "override_vs_delta.dat"), "w") as f:
    f.write("override delta lo hi\n")
    for r in sel["runs"]:
        f.write(f"{r['override']} {r['diff']} {r['diff'] - r['ci'][0]} {r['ci'][1] - r['diff']}\n")
byid = {r["id"]: r for r in sel["runs"]}
mac("pilotDiff", f"{byid['pilot']['diff']:+.0f}")
mac("pilotP", f"{byid['pilot']['p']:.3f}")
mac("pilotOverride", f"{byid['pilot']['override']:.1f}")
mac("directDiff", f"{byid['direct']['diff']:+.0f}")
mac("directOverride", f"{byid['direct']['override']:.1f}")
sc = sel["scorers"]
mac("rollEpisodes", sc["rollouts"])
mac("rollSuccess", f"{100 * sc['rollout_success']:.1f}")
mac("rollSuccessClean", f"{100 * sc['rollout_success_sigma0']:.1f}")
mac("rollExplored", f"{sc['explored_queries']:,}".replace(",", "{,}"))
mac("rollQueries", f"{sc['queries']:,}".replace(",", "{,}"))
for k in ("imag", "direct"):
    K = k.capitalize()
    mac(f"auc{K}", f"{sc[k]['val_auc_success']:.3f}")
    mac(f"rank{K}", f"{100 * sc[k]['ranking_agrees_with_outcome']:.1f}")
    mac(f"sOneRank{K}", f"{sc[k]['system1_rank_mean']:.1f}")
mac("valEpisodes", sc["imag"]["episodes_val"])

# ---------------- head-room by counterfactual branching (step 3)
S3 = os.path.join(DATA, "step3_summary.json")
pat = re.compile(r"(\d+)/(\d+) = ([\d.]+|nan)% \[([\d.]+|nan), ([\d.]+|nan)\]")


def parse(s):
    m = pat.match(s)
    k, n, p, lo, hi = m.groups()
    return int(k), int(n), p, lo, hi


if os.path.exists(S3):
    s3 = json.load(open(S3))
    rows = []

    def row(label, key, name=None):
        k, n, p, lo, hi = parse(s3[key])
        rows.append(f"{label} & {k}/{n} & {p} & [{lo}, {hi}] \\\\")
        if name:
            mac(name, p)
            mac(name + "K", k)
            mac(name + "N", n)
            mac(name + "Hi", hi)

    row("System~1 episode success", "system1_success", "sOneSucc")
    row("System-1 branch reproduces the original outcome", "system1_branch_reproduces_main", "repro")
    row("Failure branch points rescued by $\\geq$1 of 3 alternatives", "rescuable_points (>=1 of the alternatives succeeds)", "rescPts")
    for pos in ("early", "middle", "late"):
        if pos in s3["rescuable_by_position"]:
            k, n, p, lo, hi = parse(s3["rescuable_by_position"][pos])
            rows.append(f"\\quad {pos} branch point & {k}/{n} & {p} & [{lo}, {hi}] \\\\")
    row("Failed episodes rescued at any branch point", "rescuable_failed_episodes (any point)", "rescEps")
    row("Oracle episode success (System~1 + rescued)", "oracle_episode_success_upper_bound", "oracle")
    gain = float(parse(s3["oracle_episode_success_upper_bound"])[2]) - float(parse(s3["system1_success"])[2])
    mac("oracleGain", f"{gain:.1f}")
    mac("altFail", f"{s3['random_alternative_success_rate_in_failed_states']:.1f}")
    mac("altSucc", f"{s3['random_alternative_success_rate_in_successful_states']:.1f}")
    mac("brEpisodes", s3["episodes"])
    mac("brErrors", s3["errors"])
    rows.append(f"Single alternative succeeds, from a failure state & & {s3['random_alternative_success_rate_in_failed_states']:.1f} & \\\\")
    rows.append(f"Single alternative succeeds, from a success state & & {s3['random_alternative_success_rate_in_successful_states']:.1f} & \\\\")
    open(os.path.join(GEN, "tab_headroom.tex"), "w").write(r"""\begin{table}[t]
\centering
\caption{Counterfactual branching on %d LIBERO-Goal training-task episodes with our own robot initial-state perturbations (disjoint from the LIBERO-Plus test variants). At each branch point the simulator is restored and one alternative chunk from the proposer is executed, after which System~1 continues to the end of the episode. 95\%% Wilson intervals.}
\label{tab:headroom}
\small
\begin{tabular}{lrrl}
\toprule
Quantity & Count & \%% & 95\%% CI \\
\midrule
""" % s3["episodes"] + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    status = "final" if s3.get("_final") else "interim"
    mac("headroomStatus", status)
else:
    open(os.path.join(GEN, "tab_headroom.tex"), "w").write("\\begin{center}\\textbf{[Head-room results pending]}\\end{center}\n")
    for n in ("sOneSucc", "repro", "rescPts", "rescEps", "oracle", "oracleGain", "altFail", "altSucc", "brEpisodes",
              "brErrors", "rescPtsHi", "rescPtsK", "rescPtsN", "rescEpsK", "rescEpsN", "sOneSuccK", "sOneSuccN",
              "reproK", "reproN", "oracleK", "oracleN", "oracleHi", "sOneSuccHi", "reproHi", "rescEpsHi"):
        mac(n, "\\textbf{??}")
    mac("headroomStatus", "pending")

open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(macros) + "\n")
print("generated:", sorted(os.listdir(GEN)))

# ---------------- offline evaluator diagnostic (appended; rewrites numbers.tex with these macros too)
d = json.load(open(os.path.join(DATA, "evaluator_diag.json")))
mac("diagN", d["samples"])
mac("diagPairLow", f"{100 * d['pairwise_acc_gt_vs_noise']['0.05']:.1f}")
mac("diagPairHigh", f"{100 * d['pairwise_acc_gt_vs_noise']['0.3']:.1f}")
mac("diagSpearman", f"{d['spearman_score_vs_closeness']:.2f}")
mac("diagCloser", f"{100 * d['argmax_closer_than_system1']:.1f}")
mac("diagFarther", f"{100 * d['argmax_farther_than_system1']:.1f}")
mac("diagMseRatio", f"{d['mse_argmax_vs_expert'] / d['mse_system1_vs_expert']:.1f}")
mac("diagSOneRank", f"{d['system1_rank_mean']:.1f}")
mac("gapPninety", f"{d['gap_z_percentiles_50_75_90_95'][2]:.2f}")
mac("gapPseventyfive", f"{d['gap_z_percentiles_50_75_90_95'][1]:.2f}")
open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(macros) + "\n")

# ---------------- extrapolation used in the limitations (independence assumption, an optimistic bound)
if os.path.exists(S3):
    pa = s3["random_alternative_success_rate_in_failed_states"] / 100
    mac("altIndepFifteen", f"{100 * (1 - (1 - pa) ** 15):.0f}")
else:
    mac("altIndepFifteen", "\\textbf{??}")
open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(macros) + "\n")
