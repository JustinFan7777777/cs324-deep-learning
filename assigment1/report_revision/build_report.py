"""Build a data-driven LaTeX report from the saved assignment experiments.

Run after both experiment scripts.  Every numerical table is generated from
the JSON evidence; the original report and submitted implementations are never
modified.  XeLaTeX produces the PDF from the readable .tex source.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
STEM = "CS324_Assignment1_Revised_Report"


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}\\%"


def number(value, digits=4):
    return f"{value:.{digits}f}"


def sci(value):
    base, exponent = f"{value:.2e}".split("e")
    return rf"${base}\times 10^{{{int(exponent)}}}$"


def escape(value):
    value = str(value)
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def build():
    p = json.loads((HERE / "results/perceptron.json").read_text())
    m = json.loads((HERE / "results/mlp.json").read_text())
    if len(m["runs"]) != 70 or len(m["summary"]) != 14:
        raise RuntimeError("The full five-seed study must finish before building the report.")

    def run(batch, budget="equal_epoch", seed=42):
        return next(r for r in m["runs"] if r["budget"] == budget and r["batch_size"] == batch and r["seed"] == seed)

    def summary(batch, budget="equal_epoch"):
        return next(r for r in m["summary"] if r["budget"] == budget and r["batch_size"] == batch)

    full, sgd, mini = (run(b) for b in (800, 1, 8))
    high = next(b for b in p["baselines"] if b["key"] == "high_variance")
    best_train = high["history"][high["earliest_best_train_epoch"]]
    v = m["verification"]
    chunks = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm,top=18mm,bottom=19mm,headheight=14pt]{geometry}
\usepackage{fontspec}
\setmainfont{texgyreheros-regular.otf}[BoldFont=texgyreheros-bold.otf,ItalicFont=texgyreheros-italic.otf,BoldItalicFont=texgyreheros-bolditalic.otf]
\setmonofont{texgyrecursor-regular.otf}[Scale=0.86,BoldFont=texgyrecursor-bold.otf]
\usepackage{amsmath,amssymb,booktabs,array,tabularx,graphicx,float,xcolor}
\usepackage[most]{tcolorbox}
\usepackage{caption,fancyhdr,titlesec,enumitem,microtype,hyperref}
\definecolor{ink}{HTML}{20252B}
\definecolor{clay}{HTML}{C15F3C}
\definecolor{blue}{HTML}{2166AC}
\definecolor{paper}{HTML}{FAF9F5}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={CS324 Assignment 1: Perceptron and NumPy MLP},pdfauthor={Fan Xiaole}}
\color{ink}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\textcolor{clay}{CS324 / DEEP LEARNING}}
\fancyhead[R]{\small ASSIGNMENT 1}
\fancyfoot[L]{\scriptsize Fan Xiaole}
\fancyfoot[R]{\scriptsize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.16}
\setlist[itemize]{leftmargin=*,itemsep=3pt,topsep=3pt}
\titleformat{\section}{\Large\bfseries\color{clay}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{7pt}{7pt}
\titlespacing*{\subsection}{0pt}{7pt}{3pt}
\captionsetup{font=small,labelfont=bf,skip=4pt,justification=raggedright,singlelinecheck=false}
\tcbset{colback=paper,colframe=paper,boxrule=0pt,arc=1.5mm,left=9pt,right=9pt,top=6pt,bottom=6pt}
\newcommand{\finding}[2]{\begin{tcolorbox}\textbf{\textcolor{clay}{#1}}\quad #2\end{tcolorbox}}
\newcommand{\plot}[3]{\begin{figure}[H]\centering\includegraphics[width=#2\linewidth]{figures/#1}\caption{#3}\end{figure}}
\newcommand{\code}[1]{\texttt{\small #1}}
\begin{document}
{\small\textcolor{clay}{IMPLEMENTATION, VALIDATION AND CONTROLLED EXPERIMENTS}}\\[5pt]
{\huge\bfseries From a linear neuron\\[2pt] to a NumPy neural network}\\[8pt]
{\large CS324 Deep Learning --- Assignment 1}\\[4pt]
Fan Xiaole (Justin Fan)\quad |\quad Student ID: 12412307

\section{Overview and experimental protocol}
This report implements and evaluates a batch perceptron on Gaussian data, a
multi-layer perceptron (MLP) written in NumPy, and full-batch, stochastic and
mini-batch gradient descent. It retains the required default experiments and
adds three checks that make the conclusions easier to trust: numerical gradient
verification, an analytic Gaussian reference, and batch comparisons under two
explicit optimization budgets.
"""]
    chunks.append(rf"""\finding{{Main results}}{{At seed 42, the default full-batch MLP reaches
{pct(full['final']['test_accuracy'])} test accuracy, SGD reaches
{pct(sgd['final']['test_accuracy'])}, and batch size 8 reaches
{pct(mini['final']['test_accuracy'])}. These are final results after 1,500 epochs,
with very different numbers of parameter updates. Five-seed results and an
equal-update control qualify the single-seed comparison.}}
""")
    chunks.append(r"""\begin{table}[H]\small\centering
\caption{Fixed protocol. The required defaults are kept separate from extensions.}
\begin{tabularx}{\linewidth}{@{}p{31mm}XX@{}}\toprule
 & Part I: Gaussian perceptron & Parts II--III: NumPy MLP\\\midrule
Data & 100 independent samples per class in $\mathbb R^2$ & 1,000 \code{make\_moons} samples, noise 0.2\\
Split & Exactly 80 train + 20 test per class & Random stratified 800 train / 200 test\\
Labels & $-1,+1$ & Two-class one-hot vectors\\
Model & Two weights plus a bias & $2\to20\to2$, ReLU then softmax; 102 parameters\\
Initialization & All three parameters zero & $W=0.01\,\mathcal N(0,1)$; zero biases\\
Training & Batch update, $\eta=0.01$; at most 1,000 epochs & $\eta=0.01$; 1,500 epochs; evaluate every 10\\
Baseline seed & 42 & 42\\
Extensions & $6\times6$ separation/variance grid; five data seeds & Seven batch sizes; five pipeline seeds; 1,500-update control\\\bottomrule
\end{tabularx}\end{table}

\subsection{What is controlled, and what is measured}
Within each MLP seed, all batch sizes share the same data, split and initial
parameters. Across seeds $\{0,1,2,3,42\}$, the data generation, split,
initialization and shuffle order all change together. Error bars therefore
measure variation of the complete experimental pipeline, rather than
initialization alone. All primary scores use the final allotted training
endpoint; no model or epoch is selected using test accuracy.

The assignment explicitly asks for test curves and a batch-size sweep. These
test results are descriptive comparisons, not an independent estimate after
hyperparameter tuning. Any future model selection should use a validation
split and retain a fresh test set. All computation is CPU-only; NumPy handles
forward passes, gradients and updates. Scikit-learn generates data and splits.

\clearpage
\section{Part I: the batch perceptron}
\subsection{Implementation and learning rule}
For $\bar x_i=(x_{i1},x_{i2},1)^\top$ and
$\theta=(w_1,w_2,b)^\top$, prediction is $+1$ when
$\theta^\top\bar x_i\geq0$ and $-1$ otherwise. At epoch $t$, the margin
violations and batch update are
\[
 M_t=\{i:y_i\theta_t^\top\bar x_i\leq0\},\qquad
 \theta_{t+1}=\theta_t+\frac{\eta}{|M_t|}\sum_{i\in M_t}y_i\bar x_i.
\]
All violations are collected before a single update, matching the supplied
tutorial. The mean is over current violations, not all training examples.
Training stops when $M_t$ is empty or the 1,000-epoch limit is reached. A
zero score is predicted as $+1$ but is still a training violation; convergence
therefore requires strictly positive training margins.

\subsection{Required Gaussian experiments}
All cases use common covariance $\Sigma=vI_2$, where $v$ is a variance
(the coordinate standard deviation is $\sqrt v$). The 160 training and 40 test
observations are disjoint. Each class contributes exactly 80 and 20 examples.
\begin{table}[H]\small\centering
\caption{Seed-42 terminal results, reproduced by the original implementation.}
\begin{tabular}{@{}lcccrrr@{}}\toprule
Case & $\mu_-$ & $\mu_+$ & $v$ & Updates & Train & Test\\\midrule
""")
    for b in p["baselines"]:
        final = b["history"][-1]
        vec = lambda xs: "(" + ",".join(f"{x:g}" for x in xs) + ")"
        chunks.append(f"{b['title']} & ${vec(b['mean_negative'])}$ & ${vec(b['mean_positive'])}$ & {b['variance']:g} & {final['epoch']:,} & {pct(final['train_accuracy'])} & {pct(final['test_accuracy'])}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}
\plot{perceptron_boundaries}{1.0}{Learned and known-distribution decision boundaries.
Circles are training samples and outlined triangles are held-out test samples;
color encodes class. The dashed oracle knows the generating means and covariance
and is not fitted to either split.}

The well-separated sample is classified perfectly after one update. Bringing
the means closer or increasing the variance increases class overlap. However,
overlap alone does not explain every observed error: the high-variance final
boundary is visibly rotated away from the analytic reference. The next page
checks how much of the result is an optimization effect.

\clearpage
\section{Extension: overlap versus optimization}
With equal priors and common covariance $vI_2$, subtracting the two Gaussian
log densities yields the oracle rule
\[
 (\mu_+-\mu_-)^\top x-\tfrac12(\|\mu_+\|^2-\|\mu_-\|^2)\geq0.
\]
Its population accuracy is
$A^*=\Phi(\|\mu_+-\mu_-\|/(2\sqrt v))$, where $\Phi$ is the standard normal
cumulative distribution. This follows by projecting onto the mean-difference
direction: each class mean is half the separation away from the midpoint and
the projected variance is $v$. Thus the best population boundary is already
linear, while overlapping class densities still create unavoidable errors.
\plot{perceptron_trajectories}{1.0}{Accuracy from the actual zero-weight state through every completed update.
The horizontal axis is linear up to 10 and logarithmic beyond 10. The separated
case stops after one update; overlapping cases do not converge monotonically.}
""")
    chunks.append(rf"""\finding{{An optimization diagnosis}}{{The high-variance run reaches
{pct(best_train['train_accuracy'],3)} training accuracy at epoch
{high['earliest_best_train_epoch']}, but ends at
{pct(high['history'][-1]['train_accuracy'])} train / {pct(high['history'][-1]['test_accuracy'])} test.
The oracle gives {pct(high['oracle_test_accuracy'])} on the same test set and
{pct(high['population_oracle_accuracy'],3)} in the population. The final
perceptron score therefore mixes overlap with terminal-iterate oscillation.}}
""")
    chunks.append(r"""The earlier high training accuracy is a trajectory diagnostic, not a replacement
test-selected checkpoint. A training-selected pocket perceptron or a decaying
step size would be reasonable follow-up methods, but neither is substituted
for the required baseline here.
\plot{perceptron_separation_sweep}{0.95}{A $6\times6$ grid, with five paired data seeds per cell (180 fits).
Left: mean terminal test accuracy. Right: mean oracle test accuracy minus
perceptron test accuracy on identical held-out examples, in percentage points.
All other perceptron settings remain fixed.}
Increasing separation relative to $\sqrt v$ generally helps. The oracle gap
also exposes training instability in overlapping settings. The oracle is
optimal in the population, not guaranteed to beat every fitted classifier
on a small random test sample; finite-sample gaps may have either sign.

\clearpage
\section{Part II: a NumPy MLP and its gradients}
\subsection{Forward computation}
The implementation stores a batch as rows. Thus its weight matrices are the
transposes of those in the assignment's column-vector notation. For a batch
of $B$ examples, the default network computes
\begin{align*}
Z_1&=XW_1+\mathbf1 b_1^\top,& A_1&=\max(0,Z_1),\\
Z_2&=A_1W_2+\mathbf1 b_2^\top,&
P_{ic}&=\frac{\exp(Z_{2,ic}-m_i)}{\sum_j\exp(Z_{2,ij}-m_i)},\quad m_i=\max_j Z_{2,ij},\\
\mathcal L&=-\frac1B\sum_{i=1}^{B}\sum_{c=1}^{2}Y_{ic}\log P_{ic}.&&
\end{align*}
Subtracting $m_i$ leaves softmax unchanged while preventing exponential
overflow for finite logits. ReLU makes the mapping nonlinear; without that
activation, composing affine layers would still give one affine classifier.
The code clips probabilities at $10^{-12}$ when reporting cross entropy;
the scope of that protection is examined in Section 6.
\begin{table}[H]\small\centering
\caption{Default tensor shapes and parameter count. Biases broadcast over rows.}
\begin{tabular}{@{}lllll@{}}\toprule
Tensor & $X,Y$ & $W_1,b_1$ & $A_1$ & $W_2,b_2$\\\midrule
Shape & $B\times2,\ B\times2$ & $2\times20,\ 20$ & $B\times20$ & $20\times2,\ 2$\\
Parameters & -- & $40+20=60$ & -- & $40+2=42$\\\bottomrule
\end{tabular}\end{table}

\subsection{Backpropagation, with averaging exactly once}
For one-hot targets, combining softmax with mean cross entropy gives the
logit gradient $D_2=(P-Y)/B$. Backpropagation then applies
\begin{align*}
\nabla W_2&=A_1^\top D_2,& \nabla b_2&=\sum_i D_{2,i:},\\
D_1&=(D_2W_2^\top)\odot\mathbf1[Z_1>0],&&\\
\nabla W_1&=X^\top D_1,& \nabla b_1&=\sum_i D_{1,i:}.
\end{align*}
Every trainable tensor is updated by $\theta\leftarrow\theta-\eta\nabla\theta$.
Linear layers do not divide by $B$ again. Cached inputs and ReLU masks are
used for gradients before evaluation begins. The same reversed-layer
procedure supports multiple hidden layers.

\finding{A precise interface detail}{\code{CrossEntropy.backward} returns a gradient with
respect to logits, and \code{SoftMax.backward} passes it through. This coupled
pipeline is correct; these two backward methods are not independent,
general-purpose derivatives of their forward methods.}

\subsection{Numerical verification}
For every weight and bias coordinate, a central finite difference is compared
with the analytical gradient:
\[
g_j^{\rm num}=\frac{\mathcal L(\theta+\epsilon e_j)-\mathcal L(\theta-\epsilon e_j)}{2\epsilon},
\qquad \epsilon=10^{-6}.
\]
Checks cover both a shallow and a deeper network, with batch sizes 1 and 7.
This tests chain propagation and the placement of batch averaging. Full
results are stored in \code{results/mlp.json}.
\begin{table}[H]\small\centering
\caption{Maximum error across all parameter tensors in each numerical check.}
\begin{tabular}{@{}lrrr@{}}\toprule
Architecture & Batch & Max. absolute error & Max. relative $\ell_2$ error\\\midrule
""")
    for hidden in ([4], [4, 3]):
        for batch in (1, 7):
            rows = [r for r in v["tensors"] if r["hidden"] == hidden and r["batch_size"] == batch]
            arch = r"$2\to" + r"\to".join(str(n) for n in hidden) + r"\to2$"
            chunks.append(f"{arch} & {batch} & {sci(max(r['max_absolute_error'] for r in rows))} & {sci(max(r['relative_l2_error'] for r in rows))}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}

\clearpage
\section{Required default experiments: full batch and SGD}
Both runs retain width 20, learning rate 0.01, 1,500 epochs and seed 42.
Each epoch shuffles the training set and visits every example exactly once.
Full batch updates once per epoch; SGD updates after every individual example.
The trainer's \code{max\_steps} argument actually counts epochs. Its stored
\code{steps=0} follows the first completed epoch; all plots here use
\code{steps+1}, so the final point is correctly labelled epoch 1,500.

\plot{mlp_default_curves}{1.0}{Required train/test accuracy curves, paired with train/test loss.
The first measurement is after epoch 1, rather than at initialization.
Both training and test curves are shown explicitly for full batch and SGD.}
\begin{table}[H]\small\centering
\caption{Seed-42 final metrics after 1,500 complete epochs.}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Optimizer & Updates & Train accuracy & Test accuracy & Train loss & Test loss\\\midrule
""")
    for label, r in [("Full batch", full), ("SGD", sgd), ("Mini-batch 8", mini)]:
        f = r["final"]
        chunks.append(f"{label} & {r['optimizer_updates']:,} & {pct(f['train_accuracy'])} & {pct(f['test_accuracy'])} & {number(f['train_loss'])} & {number(f['test_loss'])}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}

Full-batch accuracy remains modest on both splits, so its performance is
consistent with incomplete optimization under this small initialization and
fixed budget. A small train/test gap alone cannot establish successful
learning. The SGD curve improves much sooner in epoch units, but it receives
800 times as many updates over the same number of epochs. This is evidence
about the specified training procedure, not proof that stochastic gradients
are intrinsically better at equal optimization effort.

The held-out test accuracy can exceed training accuracy on a particular
split; the two finite samples differ. Neither that ordering nor a small gap
proves that overfitting is absent. No global optimum is certified by these
curves, and no early stopping uses the test set.

\clearpage
\section{Learned geometry and numerical limits}
\plot{mlp_decision_surfaces}{1.0}{Seed-42 decision regions after the required 1,500 epochs.
Background color represents the class-1 probability; the contour is the
0.5 decision boundary. Models share exactly the same data and initialization.
Training and test observations are distinguished in the key below the panels.}
The full-batch model's boundary remains comparatively simple, whereas SGD and
the small mini-batch can follow the two interleaving moons more closely.
This geometric view complements accuracy: the MLP is capable of a curved
decision boundary, but its learned shape depends on optimization progress.
These plots describe the displayed seed and budget, not a universal ordering.

\begin{table}[H]\small\centering
\caption{Seed-42 held-out error counts and numerical diagnostics at the final 1,500-epoch endpoint.}
\begin{tabular}{@{}lrrr@{}}\toprule
Model & Test errors / 200 & Minimum true-class probability & Clipped true-class values\\\midrule
""")
    for label, r in [("Full batch", full), ("SGD", sgd), ("Mini-batch 8", mini)]:
        diag = r["diagnostics"]["test"]
        matrix = diag["confusion_matrix_true_rows_predicted_columns"]
        errors = matrix[0][1] + matrix[1][0]
        chunks.append(f"{label} & {errors} & {diag['minimum_true_class_probability']:.5f} & {diag['clipped_true_class_count']}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}

\subsection{Accuracy has a finite resolution}
There are 200 MLP test observations: one changed prediction moves accuracy
by 0.5 percentage points. Thus the seed-42 difference between 99.0\% and
98.5\% corresponds to one example. The five-seed results in Section 8 provide
context; they do not turn that small difference into a statistically
established ranking.

\subsection{What stable softmax does and does not guarantee}
The maximum-shift trick prevents exponential overflow, and probability
clipping prevents a reported $\log 0$. However, clipping changes the objective
in the extreme-confidence regime. With logits $(-100,100)$ and target
class 0, the exact logit-based cross entropy is
\[
 \log(e^{-100}+e^{100})-(-100)\approx200,
\]
while the original probability-clipped forward pass reports
$-\log(10^{-12})\approx27.631$. Its backward pass still uses $(P-Y)/B$.
The numerical diagnostic therefore records a limitation, rather than
claiming complete forward/backward consistency for arbitrarily extreme logits.

\finding{A scoped numerical improvement}{Future stress tests should compute
log-sum-exp loss directly from logits with the corresponding fused gradient.
The required baseline is preserved here.}

Finite differences validate sampled inputs away from ReLU's nondifferentiable
zero. No true-class probabilities are clipped at the final seed-42 train/test
endpoints (seven batches, both budgets). The extreme-logit diagnostic is
therefore not an explanation of the reported default losses.

\clearpage
\section{Part III: how batch size changes the budget}
The training function accepts \code{None} for full batch, 1 for SGD and
larger positive integers for mini-batches. The command line maps
\code{--batch\_size 0} to full batch. Partial final batches are retained,
including the final 32 examples for $B=128$.

With $n=800$ training examples and $E=1{,}500$ epochs,
\[
 U=E\left\lceil\frac{800}{B}\right\rceil,\qquad
 S=800E=1{,}200{,}000,
\]
where $U$ counts parameter updates and $S$ counts processed examples.
Equal epochs equalize example exposure; they do not equalize update count,
per-update arithmetic, or wall time.
\begin{table}[H]\small\centering
\caption{Required seed-42 batch-size sweep at equal epochs. Every row processes 1,200,000 examples.}
\begin{tabular}{@{}rrrrrr@{}}\toprule
Batch $B$ & Updates/epoch & Total updates & Train accuracy & Test accuracy & Test loss\\\midrule
""")
    for batch in m["config"]["batch_sizes"]:
        r = run(batch)
        f = r["final"]
        chunks.append(f"{batch} & {(800+batch-1)//batch} & {r['optimizer_updates']:,} & {pct(f['train_accuracy'])} & {pct(f['test_accuracy'])} & {number(f['test_loss'])}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}
\plot{mlp_batch_trajectories}{1.0}{Seed-42 test trajectories under two budgets: left, each method runs
1,500 epochs; right, each runs exactly 1,500 parameter updates. The panels
use different stopping rules while sharing data, initial parameters and
shuffled epoch order. Section 8 aggregates both protocols across five seeds.}

At a fixed learning rate, a small batch gives more frequent parameter
adjustments and a noisier estimate of the full gradient. The observed
equal-epoch advantage mixes these effects. The table is a useful comparison
when a full-data-pass budget is the constraint, but it does not isolate
gradient noise or establish computational efficiency. Timing is saved in
the evidence JSON and is machine- and implementation-dependent.

\clearpage
\section{Extension: a controlled equal-update comparison}
The second study fixes every run at exactly 1,500 parameter updates. It uses
the same initialization, data, shuffled epoch order and gradient formulas
as the baseline, stopping after the last allowed batch even mid-epoch.
For full batch, this reproduces the original 1,500-epoch endpoint; serialized
parameters are checked against the baseline. Short final batches are counted
by their actual size.

\plot{mlp_budget_comparison}{1.0}{Final performance under equal epochs and equal updates, with five paired
pipeline seeds. Error bars are sample standard deviations ($n=5$, denominator
$n-1$), not confidence intervals. Each budget answers a different question;
neither equalizes all aspects of compute.}
\begin{table}[H]\small\centering
\caption{Five-seed test accuracy (mean $\pm$ SD, in percentage points), plus actual exposure under the 1,500-update control.}
\begin{tabular}{@{}rrrrr@{}}\toprule
Batch & 1,500 epochs & 1,500 updates & Examples at 1,500 updates & Equivalent epochs\\\midrule
""")
    for batch in m["config"]["batch_sizes"]:
        a, b = summary(batch), summary(batch, "equal_update")
        r = run(batch, "equal_update")
        fmt = lambda row: f"${100*row['test_accuracy_mean']:.2f}\\pm{100*row['test_accuracy_sd']:.2f}$"
        chunks.append(f"{batch} & {fmt(a)} & {fmt(b)} & {r['samples_seen']:,} & {r['epoch_equivalent']:,.3f}\\\\\n")
    chunks.append(r"""\bottomrule\end{tabular}\end{table}
""")
    b8, b1 = summary(8), summary(1)
    seed8values = list(b8["test_accuracy_by_seed"].values())
    chunks.append(rf"""\finding{{The single seed is not the whole result}}{{Batch size 8 reaches
{pct(mini['final']['test_accuracy'])} at seed 42, but its five-seed mean is
{pct(b8['test_accuracy_mean'])} with SD {100*b8['test_accuracy_sd']:.2f} percentage
points, spanning {pct(min(seed8values))}--{pct(max(seed8values))}.
SGD's five-seed mean is {pct(b1['test_accuracy_mean'])}. The displayed results
do not justify a universal ``best batch size'' claim.}}
""")
    chunks.append(r"""At equal updates, small batches process fewer observations: SGD sees only
1,500 examples (1.875 training-set equivalents), whereas full batch sees
1,200,000. At equal epochs, both see 1,200,000 examples but receive very
different update counts. Reading both experiments together separates budget
definitions and prevents attributing every difference to stochasticity.
The 1,500-update horizon is inherited from the required full-batch baseline,
rather than chosen by observing which horizon favors a method.

\clearpage
\section{Reproduction, coverage and limitations}
\subsection{Files and commands}
The revised notebook, experiment scripts, exact JSON histories, serialized
seed-42 MLP parameters, and vector/raster figures are in \code{report\_revision/}.
Original implementations are unchanged; earlier reports are preserved in
the local archive. Use
Python 3.11 with NumPy, scikit-learn, Matplotlib, Jupyter, nbformat and
nbconvert; the report builder additionally needs XeLaTeX.
From the assignment root:
\begin{tcolorbox}\small\ttfamily
export OPENBLAS\_NUM\_THREADS=1 OMP\_NUM\_THREADS=1\\
python report\_revision/perceptron\_experiments.py\\
python report\_revision/mlp\_experiments.py\\
python report\_revision/make\_notebook.py\\
python -m nbconvert --to notebook --execute --inplace \textbackslash{}\\
\hspace*{1em}report\_revision/assignment1\_revised.ipynb\\
python report\_revision/build\_report.py
\end{tcolorbox}
The experiment scripts regenerate all figures and numerical evidence. The
executed notebook loads that evidence without retraining, and also includes
an explicitly enabled rerun cell. It displays the required train/test curves
for both default optimizers and the batch-size analysis without raw log spam.
The README gives a complete command to execute the notebook automatically.
""")
    env = m["environment"]
    chunks.append(rf"""\textbf{{Verified environment:}} Python {escape(env['python'])}, NumPy
{escape(env['numpy'])}, scikit-learn {escape(env['sklearn'])}, Matplotlib
{escape(env['matplotlib'])}. NumPy/OpenMP thread limits are 1. Part I takes
{p['elapsed_seconds']:.2f} seconds and the complete MLP study takes
{m['compute_elapsed_seconds']:.1f} seconds on this machine; individual run
times are recorded in \code{{results/mlp.json}}. Reproduction on other
hardware may have different timings and tiny floating-point differences.
""")
    chunks.append(r"""\begin{table}[H]\small\centering
\caption{Traceability to all required tasks.}
\begin{tabularx}{\linewidth}{@{}p{29mm}XX@{}}\toprule
Requirement & Evidence in report & Producing code\\\midrule
I.1--I.3 & Gaussian protocol and final accuracy; Section 2 & \code{Part\_1/experiments.py}, \code{perceptron.py}\\
I.4 & Mean/variance cases and trajectories; Sections 2--3 & \code{perceptron\_experiments.py}\\
II.1 & Forward/backward math, shapes and gradient checks; Section 4 & \code{Part\_2/modules.py}, \code{mlp\_numpy.py}\\
II.2 & Random 80/20 split; NumPy training/test loop & \code{Part\_2/train\_mlp\_numpy.py}\\
II.3 & Default full-batch train/test accuracy; Section 5 & Revised notebook and \code{mlp\_experiments.py}\\
III.1 & Configurable full batch, SGD and mini-batches; Section 7 & Original trainer's \code{batch\_size}\\
III.2 & SGD train/test curves, batch influence; Sections 5, 7--8 & Revised notebook and \code{mlp\_experiments.py}\\\bottomrule
\end{tabularx}\end{table}

\subsection{Limits of the evidence}
Five seeds describe sensitivity but provide limited statistical precision.
Gaussian test sets have 40 observations per seed; MLP test sets have 200.
The Gaussian oracle knows distribution parameters unavailable to a learner.
The seed study changes multiple randomness sources together. The experiments
use only the prescribed architecture, learning rate, noise and data sizes;
no claim extends to other tasks or training regimes. Equal updates and equal
epochs are complementary controls, not equal wall-clock benchmarks.

The code passes numerical and reproduction checks in the tested regime;
the clipped-loss limitation is explicitly retained and documented. Further
capacity, learning-rate or initialization studies would require a separate
validation protocol, rather than choosing from this report's test results.

\subsection{Source material}
\begin{itemize}
\item Jianguo Zhang, \emph{CS324: Deep Learning, Assignment 1}, September 25, 2025, supplied handout (locally retained as \code{references/Assignment\_1.pdf}).
\item Supplied \emph{Perceptron Tutorial}; batch-update algorithm (locally retained as \code{references/perceptron\_tutorial.pdf}).
\item Submitted source modules and saved experiment evidence listed above. The Gaussian oracle and all gradient equations are derived in this report; the teacher-provided examples are structural references, not sources of numerical results.
\end{itemize}
\end{document}
""")
    source = HERE / (STEM + ".tex")
    source.write_text("".join(chunks))
    # Keep generated TeX logs and auxiliary files out of the source directory.
    build_dir = HERE / "build"
    build_dir.mkdir(exist_ok=True)
    # Two passes resolve cross references and stabilize table/figure numbering.
    for _ in range(2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=build", source.name],
            cwd=HERE, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-7000:] + result.stderr[-1000:])
    destination = HERE / (STEM + ".pdf")
    shutil.copyfile(build_dir / (STEM + ".pdf"), destination)
    print(destination)


if __name__ == "__main__":
    build()
