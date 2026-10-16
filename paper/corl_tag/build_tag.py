"""Build the CoRL 2026 / TAG workshop submission from ../main.tex, so the two versions never drift apart.

    cd paper && python make_generated.py && python corl_tag/build_tag.py
    cd corl_tag && latexmk -pdf main_tag.tex

Takes the body of main.tex (everything after the abstract, up to the bibliography), swaps in the CoRL
preamble, a 5-sentence abstract (CoRL asks for 4-6), keywords and a visible AI-assistance disclosure
(the template hides \\acknowledgments in submissions), and points \\input paths at ../generated.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "main.tex")).read()

body = src.split("\\end{abstract}", 1)[1]
body = body.split("\\section*{Acknowledgements}", 1)[0]
body = body.replace("generated/", "../generated/")
body = body.replace("\\section{Discussion and limitations}", "\\section{Limitations and discussion}")
assert "\\section{Limitations and discussion}" in body

preamble = r"""\documentclass{article}
\usepackage{corl_2026} % initial submission: anonymous, line numbers
% \usepackage[final]{corl_2026}   % camera-ready
\usepackage{amsmath,amssymb}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[capitalise,noabbrev]{cleveref}
\definecolor{seriesA}{HTML}{2A78D6}
\definecolor{inkmuted}{HTML}{52514E}
\newcommand{\A}{\mathbf{A}}
\input{../generated/numbers.tex}
\newcommand{\coderef}{the supplementary material (released upon publication)}

\title{Is There Anything to Search For? A Counterfactual Head-Room Analysis of Test-Time Action Selection for VLA Models}

\author{
  Yike Wang\\
  University of Wollongong\\
  \texttt{yw061@uowmail.edu.au}
}

\begin{document}
\maketitle

\begin{abstract}
Test-time selection---sampling candidate actions from a robot policy and executing the one a learned scorer prefers---is a popular way to make vision-language-action (VLA) models more robust without retraining them. On RynnVLA-002, whose success on LIBERO-Goal drops from \goalVanilla\% to \goalPlusRobot\% when the robot's initial state is perturbed, we built a latent imagination search over 16 proposed action chunks and evaluated five scorers on the same 100 paired episodes: none beat the unmodified policy, and the two that overrode it most often were significantly worse (\pilotDiff{} and \directDiff{} points). Diagnostics show that the scorers reject implausible chunks but cannot rank plausible ones, and that outcome-trained scorers predict \emph{whether} an episode will succeed (AUC up to \aucDirect) rather than \emph{which} chunk to take. To ask whether any scorer could have helped, we restore simulator snapshots and replay alternative chunks from the same states: only \rescPts\% of failure states are rescued by any of three alternatives, bounding the gain of a perfect selector at \oracleGain{} points. Under these perturbations, failures mostly reflect states from which the policy itself does not recover---an action gap that choosing among the policy's own proposals does little to close.
\end{abstract}

\keywords{Vision-language-action models, test-time action selection, world models}
"""

tail = r"""
\paragraph{Use of AI assistance.} The implementation, experiments and a draft of this paper were prepared with the assistance of an AI system (Claude, Anthropic). The author reviewed the code, results and text and takes full responsibility for the content.

\acknowledgments{This work builds on the code and models released by the authors of RynnVLA-002 and on the LIBERO-Plus benchmark.}

\bibliography{../refs}

\end{document}
"""

out = preamble + body + tail
# the template's natbib style is numeric; no \bibliographystyle needed (corlabbrvnat is set by the .sty)
out = re.sub(r"\\bibliographystyle\{[^}]*\}", "", out)
open(os.path.join(HERE, "main_tag.tex"), "w").write(out)
print("wrote corl_tag/main_tag.tex", len(out), "chars")
