"""Turn experiment outputs in paper/data into LaTeX inputs in paper/generated.
Every number in the toy-experiment and latency sections of the paper comes from here."""
import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, GEN = os.path.join(HERE, "data"), os.path.join(HERE, "generated")
os.makedirs(GEN, exist_ok=True)


def ms(xs):
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def fmt(xs, pct=False):
    m, s = ms(xs)
    if pct:
        return f"{100 * m:.1f} $\\pm$ {100 * s:.1f}"
    return f"{m:.3f} $\\pm$ {s:.3f}"


summ = json.load(open(os.path.join(DATA, "toy_summary.json")))
seeds = [r["seed"] for r in summ]
Ns = sorted(int(n) for n in summ[0]["after"]["search_acc"])


def col(phase, key):
    return [r[phase][key] for r in summ]


# ---- search curve data
with open(os.path.join(GEN, "search_curve.dat"), "w") as f:
    f.write("N before before_sd after after_sd\n")
    for n in Ns:
        b = [r["before"]["search_acc"][str(n)] for r in summ]
        a = [r["after"]["search_acc"][str(n)] for r in summ]
        f.write(f"{n} {100*ms(b)[0]:.2f} {100*ms(b)[1]:.2f} {100*ms(a)[0]:.2f} {100*ms(a)[1]:.2f}\n")

# ---- loss curves: mean over seeds, smoothed with a 20-step window, every 10th step
loss_keys = ["lis_fm", "lis_wm", "lis_lang_nce", "lis_rank"]
curves = []
for s in seeds:
    rows = list(csv.DictReader(open(os.path.join(DATA, f"toy_losses_seed{s}.csv"))))
    curves.append(rows)
n_steps = min(len(c) for c in curves)
with open(os.path.join(GEN, "loss_curves.dat"), "w") as f:
    f.write("step " + " ".join(loss_keys) + "\n")
    for step in range(0, n_steps, 10):
        lo, hi = max(0, step - 10), min(n_steps, step + 10)
        vals = []
        for k in loss_keys:
            per_seed = [st.mean(float(c[i][k]) for i in range(lo, hi)) for c in curves]
            vals.append(st.mean(per_seed))
        f.write(f"{step} " + " ".join(f"{v:.4f}" for v in vals) + "\n")

n_eval = 200
search_row_after = " & ".join(fmt([r["after"]["search_acc"][str(n)] for r in summ], pct=True) for n in Ns)
search_row_before = " & ".join(fmt([r["before"]["search_acc"][str(n)] for r in summ], pct=True) for n in Ns)

tex = r"""
\begin{table}[t]
\centering
\caption{Controlled toy experiment on %d held-out sequences (mean $\pm$ s.d.\ over %d seeds). ``Before'' is the untrained System~2 (zero-initialised world model, i.e.\ identity dynamics); ``after'' is after training. Cosine similarities are computed between flattened latent sets; note that $E_{\mathrm{obs}}$ is itself trained, so the copy baseline changes too.}
\label{tab:toy}
\small
\begin{tabular}{lcc}
\toprule
Metric & Before & After \\
\midrule
$\cos(\hat{\bm z}_{t+H}, \bm z_{t+H})$ & %s & %s \\
$\cos(\bm z_t, \bm z_{t+H})$ (copy baseline) & %s & %s \\
Action sensitivity (\%%) & %s & %s \\
Proposer samples with instructed effect (\%%) & %s & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (n_eval, len(seeds),
       fmt(col("before", "cos_pred")), fmt(col("after", "cos_pred")),
       fmt(col("before", "cos_copy")), fmt(col("after", "cos_copy")),
       fmt(col("before", "wm_action_sensitive_acc"), True), fmt(col("after", "wm_action_sensitive_acc"), True),
       fmt(col("before", "proposer_correct_frac"), True), fmt(col("after", "proposer_correct_frac"), True))

tex += r"""
\begin{figure}[t]
\centering
\begin{minipage}[t]{0.48\linewidth}
\centering
\begin{tikzpicture}
\begin{axis}[width=\linewidth, height=5.2cm, xmode=log, log basis x=2, xtick={%s},
  xticklabels={%s}, xlabel={Random candidates $N$}, ylabel={Selected chunk correct (\%%)},
  ymin=30, ymax=105, ytick={40,60,80,100}, grid=major, grid style={draw=black!8}, axis line style={black!40},
  tick style={black!40}, legend columns=3, legend style={at={(0.5,1.03)}, anchor=south, draw=none, font=\footnotesize, /tikz/every even column/.append style={column sep=6pt}},
  label style={font=\small}, tick label style={font=\footnotesize}]
\addplot[seriesA, line width=1.2pt, mark=*, mark size=2pt, error bars/.cd, y dir=both, y explicit]
  table[x=N, y=after, y error=after_sd]{generated/search_curve.dat};
\addlegendentry{trained}
\addplot[seriesB, line width=1.2pt, mark=square*, mark size=2pt, error bars/.cd, y dir=both, y explicit]
  table[x=N, y=before, y error=before_sd]{generated/search_curve.dat};
\addlegendentry{untrained}
\addplot[black!50, dashed, domain=1:32, samples=2] {50};
\addlegendentry{chance}
\end{axis}
\end{tikzpicture}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2, horizontal sep=1.3cm, vertical sep=1.25cm},
  width=0.58\linewidth, height=2.9cm, grid=major, grid style={draw=black!8}, axis line style={black!40},
  tick style={black!40}, tick label style={font=\scriptsize}, title style={font=\footnotesize, yshift=-1ex},
  xtick={0,200,400}, ymin=0, scaled y ticks=false, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2}]
\nextgroupplot[title={$\mathcal{L}_{\mathrm{fm}}$}]
\addplot[seriesA, line width=1pt] table[x=step, y=lis_fm]{generated/loss_curves.dat};
\nextgroupplot[title={$\mathcal{L}_{\mathrm{wm}}$}]
\addplot[seriesA, line width=1pt] table[x=step, y=lis_wm]{generated/loss_curves.dat};
\nextgroupplot[title={$\mathcal{L}_{\mathrm{lang}}$}, xlabel={\scriptsize step}]
\addplot[seriesA, line width=1pt] table[x=step, y=lis_lang_nce]{generated/loss_curves.dat};
\nextgroupplot[title={$\mathcal{L}_{\mathrm{rank}}$}, xlabel={\scriptsize step}]
\addplot[seriesA, line width=1pt] table[x=step, y=lis_rank]{generated/loss_curves.dat};
\end{groupplot}
\end{tikzpicture}
\end{minipage}
\caption{Toy experiment. \emph{Left:} imagination search over $N$ uniformly random chunks; a chunk is correct if it produces the instructed marker (each random chunk is correct with probability 0.5). Error bars: s.d.\ over %d seeds, %d held-out sequences each. \emph{Right:} training losses (mean over seeds, 20-step moving average).}
\label{fig:toy}
\end{figure}
""" % (",".join(map(str, Ns)), ",".join(map(str, Ns)), len(seeds), n_eval)

after1 = ms([r["after"]["search_acc"][str(Ns[0])] for r in summ])[0]
afterbest = max(Ns, key=lambda n: ms([r["after"]["search_acc"][str(n)] for r in summ])[0])
afterbest_v = ms([r["after"]["search_acc"][str(afterbest)] for r in summ])[0]
sens = ms(col("after", "wm_action_sensitive_acc"))[0]
tex += r"""
\Cref{tab:toy} and \cref{fig:toy} summarise the results. After training, the world model is action-sensitive in %.1f\%% of held-out cases, and selecting among random chunks with the evaluator's score raises the fraction of chunks with the instructed effect from %.1f\%% at $N{=}1$ (chance) to %.1f\%% at $N{=}%d$. The untrained System~2 stays near chance, as expected from its identity dynamics. The trained proposer alone also learns to produce the instructed effect, so in this toy task search is not \emph{needed}; the experiment is designed only to show that the world model and evaluator implement a working selection signal when the dynamics are learnable from the latent state.
""" % (100 * sens, 100 * after1, 100 * afterbest_v, afterbest)

open(os.path.join(GEN, "toy_results.tex"), "w").write(tex)

# ---- latency
lat = json.load(open(os.path.join(DATA, "latency.json")))
cells = " & ".join(f"{r['N']}" + (f"+CEM{r['cem']}" if r["cem"] else "") for r in lat["rows"])
vals = " & ".join(f"{r['median_ms']:.0f}" for r in lat["rows"])
ltx = r"""Measured for the System-2 heads alone at 7B-backbone dimensions (%.1fM parameters) on %s with PyTorch %s; the backbone pass is shared with System~1 and excluded. Median of 5 runs, in milliseconds per control step:
\begin{center}\small
\begin{tabular}{l%s}
\toprule
$N$ & %s \\
ms & %s \\
\bottomrule
\end{tabular}
\end{center}
These CPU numbers only bound the cost of search from above; GPU latency, which is what matters for deployment, has not been measured.
""" % (lat["params_M"], lat["device"].replace("_", r"\_"), lat["torch"].split("+")[0], "c" * len(lat["rows"]), cells, vals)
open(os.path.join(GEN, "latency.tex"), "w").write(ltx)
print("generated:", os.listdir(GEN))
