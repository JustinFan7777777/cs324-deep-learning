"""Reproduce Part I and visualize overlap versus terminal-iterate optimization.

The original submitted implementation is imported without modification.  The
trajectory recorder mirrors its batch update exactly and asserts that its final
parameters agree with an independent call to ``Perceptron.train``.  Bayes lines
are known-distribution oracles, never fitted to either split.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Part_1"))
from experiments import create_dataset  # noqa: E402
from perceptron import Perceptron  # noqa: E402

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, INK, GREEN = "#2166ac", "#c15f3c", "#20252b", "#35745a"
MAX_EPOCHS, LEARNING_RATE = 1000, 0.01
SETTINGS = [
    ("separated", "Well separated", [-2.0, -2.0], [2.0, 2.0], 0.25),
    ("close_means", "Close means", [-0.5, -0.5], [0.5, 0.5], 0.25),
    ("high_variance", "High variance", [-2.0, -2.0], [2.0, 2.0], 2.0),
]


def predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Use precisely the submitted decision rule, including the zero tie."""
    return np.where(x @ weights[:-1] + weights[-1] >= 0, 1, -1)


def record_trajectory(x_train, y_train, x_test, y_test):
    """Record epoch zero and every completed batch update, without test feedback."""
    inputs = np.column_stack([x_train, np.ones(len(x_train))])
    weights = np.zeros(inputs.shape[1])
    history = []
    for epoch in range(MAX_EPOCHS + 1):
        margins = y_train * (inputs @ weights)
        violations = margins <= 0
        history.append(
            {
                "epoch": epoch,
                "train_accuracy": float(np.mean(predict(x_train, weights) == y_train)),
                "test_accuracy": float(np.mean(predict(x_test, weights) == y_test)),
                "margin_violations": int(violations.sum()),
                "per_sample_hinge_zero_margin": float(np.maximum(0, -margins).mean()),
                "weights": weights.tolist(),
            }
        )
        if not violations.any() or epoch == MAX_EPOCHS:
            break
        # Match the assignment tutorial: normalize by the current violation count.
        gradient = -np.mean(y_train[violations, None] * inputs[violations], axis=0)
        weights -= LEARNING_RATE * gradient
    original = Perceptron(2, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
    original.train(x_train, y_train)
    np.testing.assert_array_equal(weights, original.weights)
    return history


def oracle(mean_negative, mean_positive, variance):
    """Equal priors and common variance*I imply a linear likelihood-ratio rule.

    Projecting onto the unit mean-difference direction gives class-conditional
    univariate normals whose means lie distance/2 from the midpoint.  Thus the
    correct-class probability is Phi(distance / (2*sqrt(variance))).
    """
    negative, positive = np.asarray(mean_negative), np.asarray(mean_positive)
    distance = float(np.linalg.norm(positive - negative))
    weights = np.r_[positive - negative, -(positive @ positive - negative @ negative) / 2]
    population_accuracy = 0.5 * (1 + math.erf(distance / (2 * math.sqrt(2 * variance))))
    return weights, distance, population_accuracy


def save(fig, stem):
    fig.savefig(OUT / "figures" / f"perceptron_{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"perceptron_{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def line_on_axes(ax, weights, limits, **kwargs):
    """Contour the affine score, supporting vertical boundaries without division."""
    xx, yy = np.meshgrid(np.linspace(*limits[0], 160), np.linspace(*limits[1], 160))
    score = weights[0] * xx + weights[1] * yy + weights[2]
    ax.contour(xx, yy, score, levels=[0], **kwargs)


def baseline_figures(baselines):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55), constrained_layout=True)
    for ax, entry in zip(axes, baselines):
        x_train, y_train, x_test, y_test = entry["data"]
        final = entry["history"][-1]
        for label, color in [(-1, BLUE), (1, ORANGE)]:
            ax.scatter(*x_train[y_train == label].T, s=10, color=color, alpha=0.48, linewidths=0)
            ax.scatter(*x_test[y_test == label].T, s=28, color=color, marker="^", edgecolors=INK, linewidths=0.5)
        combined = np.r_[x_train, x_test]
        limits = [(combined[:, k].min() - 0.5, combined[:, k].max() + 0.5) for k in range(2)]
        line_on_axes(ax, np.asarray(final["weights"]), limits, colors=[INK], linewidths=1.5)
        line_on_axes(ax, entry["oracle_weights"], limits, colors=[GREEN], linewidths=1.5, linestyles="--")
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="$x_1$", ylabel="$x_2$")
        ax.set_title(f'{entry["title"]}\nTrain {final["train_accuracy"]:.1%}; test {final["test_accuracy"]:.1%}', fontsize=10)
        ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([], [], color=BLUE, marker="o", linestyle="", label="Class -1"),
        Line2D([], [], color=ORANGE, marker="o", linestyle="", label="Class +1"),
        Line2D([], [], color=INK, marker="o", linestyle="", label="Train"),
        Line2D([], [], color=INK, marker="^", linestyle="", label="Test"),
        Line2D([], [], color=INK, label="Final perceptron"),
        Line2D([], [], color=GREEN, linestyle="--", label="Known-distribution oracle"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    save(fig, "boundaries")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1), constrained_layout=True)
    for ax, entry in zip(axes, baselines):
        history = entry["history"]
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [100 * row["train_accuracy"] for row in history], color=BLUE, lw=1.1, label="Train")
        ax.plot(epochs, [100 * row["test_accuracy"] for row in history], color=ORANGE, lw=1.1, alpha=0.85, label="Test")
        ax.set_xscale("symlog", linthresh=10)
        ax.set(xlim=(0, MAX_EPOCHS), ylim=(0, 102), xlabel="Completed batch updates (epochs)", ylabel="Accuracy (%)")
        ax.set_title(entry["title"], fontsize=10)
        ax.set_xticks([0, 10, 100, 1000], labels=["0", "10", "100", "1,000"])
        ax.grid(alpha=0.18)
        if len(history) == 2:
            ax.annotate("Stops at epoch 1", xy=(1, 100), xytext=(17, 79), arrowprops={"arrowstyle": "->", "color": INK}, fontsize=8)
        if entry["key"] == "high_variance":
            ax.annotate("Terminal iterate oscillates", xy=(350, 70), xytext=(15, 20), arrowprops={"arrowstyle": "->", "color": INK}, fontsize=8)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, "trajectories")


def separation_sweep():
    """Five paired seeds per cell; hold size, update rule and budget fixed."""
    distances = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    variances = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    seeds = [0, 1, 2, 3, 4]
    cells = []
    mean_accuracy = np.zeros((len(variances), len(distances)))
    mean_gap = np.zeros_like(mean_accuracy)
    for i, variance in enumerate(variances):
        for j, distance in enumerate(distances):
            positive = np.ones(2) * distance / (2 * math.sqrt(2))
            negative = -positive
            oracle_weights, _, population_accuracy = oracle(negative, positive, variance)
            runs = []
            for seed in seeds:
                x_train, y_train, x_test, y_test = create_dataset(negative, positive, variance, seed)
                model = Perceptron(2, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
                model.train(x_train, y_train)
                runs.append({
                    "seed": seed,
                    "train_accuracy": float(np.mean(model.forward(x_train) == y_train)),
                    "test_accuracy": float(np.mean(model.forward(x_test) == y_test)),
                    "oracle_test_accuracy": float(np.mean(predict(x_test, oracle_weights) == y_test)),
                })
            accuracies = [run["test_accuracy"] for run in runs]
            gaps = [run["oracle_test_accuracy"] - run["test_accuracy"] for run in runs]
            mean_accuracy[i, j] = np.mean(accuracies) * 100
            mean_gap[i, j] = np.mean(gaps) * 100
            cells.append({"distance": distance, "variance": variance, "population_oracle_accuracy": population_accuracy,
                          "mean_test_accuracy": float(np.mean(accuracies)), "sd_test_accuracy": float(np.std(accuracies, ddof=1)),
                          "mean_paired_oracle_minus_perceptron": float(np.mean(gaps)), "runs": runs})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65), constrained_layout=True)
    for ax, values, title, cmap, lower, upper in [
        (axes[0], mean_accuracy, "Final test accuracy: mean of five seeds (%)", "Blues", 45, 100),
        (axes[1], mean_gap, "Oracle minus final test accuracy (points)", "RdBu_r", -20, 20),
    ]:
        heatmap = ax.imshow(values, origin="lower", cmap=cmap, vmin=lower, vmax=upper, aspect="auto")
        ax.set_xticks(range(len(distances)), labels=[str(v) for v in distances])
        ax.set_yticks(range(len(variances)), labels=[str(v) for v in variances])
        ax.set(xlabel="Euclidean mean separation", ylabel="Variance per coordinate", title=title)
        ax.title.set_fontsize(9)
        for i in range(len(variances)):
            for j in range(len(distances)):
                val = values[i, j]
                color = "white" if (ax is axes[0] and val > 78) or (ax is axes[1] and abs(val) > 12) else INK
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(heatmap, ax=ax, shrink=0.88, pad=0.025)
    save(fig, "separation_sweep")
    return {"distances": distances, "variances": variances, "seeds": seeds, "cells": cells}


def main():
    started = time.perf_counter()
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    baselines = []
    for key, title, negative, positive, variance in SETTINGS:
        data = create_dataset(negative, positive, variance, random_seed=42)
        history = record_trajectory(*data)
        oracle_weights, distance, population_accuracy = oracle(negative, positive, variance)
        x_train, y_train, x_test, y_test = data
        baselines.append({"key": key, "title": title, "mean_negative": negative, "mean_positive": positive,
                          "variance": variance, "random_seed": 42, "distance": distance, "data": data,
                          "oracle_weights": oracle_weights, "population_oracle_accuracy": population_accuracy,
                          "oracle_train_accuracy": float(np.mean(predict(x_train, oracle_weights) == y_train)),
                          "oracle_test_accuracy": float(np.mean(predict(x_test, oracle_weights) == y_test)),
                          "history": history,
                          "earliest_best_train_epoch": max(history, key=lambda row: row["train_accuracy"])["epoch"],
                          "trajectory_matches_original_weights_exactly": True})
    baseline_figures(baselines)
    sweep = separation_sweep()
    serializable = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in entry.items() if key != "data"} for entry in baselines]
    result = {
        "schema_version": 1,
        "provenance": {"script": "report_revision/perceptron_experiments.py", "original_sources": ["Part_1/perceptron.py", "Part_1/experiments.py"],
                       "original_source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                  for path in [ROOT / "Part_1/perceptron.py", ROOT / "Part_1/experiments.py"]},
                       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                       "training_selection": "No model or epoch is selected using test data. All primary results are terminal iterates.",
                       "oracle": "Analytic known-distribution rule with equal priors and covariance variance*I; not a fitted competitor.",
                       "population_oracle_formula": "Phi(norm(mu_positive-mu_negative)/(2*sqrt(variance)))",
                       "sweep_uncertainty": "SD across five independent data seeds; 40 test points per seed; descriptive, not a confidence interval."},
        "config": {"max_epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE, "train_per_class": 80, "test_per_class": 20,
                   "gradient_normalization": "Current number of nonpositive-margin training points", "zero_score_prediction": 1},
        "baselines": serializable, "sweep": sweep, "elapsed_seconds": time.perf_counter() - started,
    }
    with (OUT / "results" / "perceptron.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for entry in baselines:
        final = entry["history"][-1]
        print(f'{entry["key"]}: epoch={final["epoch"]}, train={final["train_accuracy"]:.5f}, test={final["test_accuracy"]:.5f}; oracle test={entry["oracle_test_accuracy"]:.5f}')
    print(f"Elapsed: {result['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
