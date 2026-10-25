"""Reproduce the original MLP and compare batch sizes under two budgets.

The original assignment files remain unchanged.  The equal-epoch experiment
calls the submitted trainer directly; the equal-update experiment repeats its
data construction, initialization, shuffling, gradients, and parameter updates.
No experiment selects a checkpoint or configuration using test accuracy.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "assignment1-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Part_2"))
from mlp_numpy import MLP
from modules import CrossEntropy, SoftMax
from train_mlp_numpy import accuracy, train

SEEDS = [0, 1, 2, 3, 42]
BATCHES = [1, 8, 16, 32, 64, 128, 800]
EPOCHS = 1500
UPDATES = 1500
LR = 0.01
BLUE = "#2166ac"
ORANGE = "#c15f3c"
COLORS = ["#2166ac", "#4393c3", "#66a6a4", "#7a8a35", "#ba8b3a", "#c15f3c", "#75559b"]


def dataset(seed):
    """Use exactly the data and stratified split in the original trainer."""
    x, labels = make_moons(n_samples=1000, noise=0.2, random_state=seed)
    y = np.eye(2)[labels]
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=labels)


def parameters(model):
    """Serialize trainable layers in forward order for exact replotting."""
    return [{key: value.tolist() for key, value in layer.params.items()}
            for layer in model.layers if hasattr(layer, "params")]


def restore(state):
    model = MLP(2, [20], 2)
    for layer, saved in zip([x for x in model.layers if hasattr(x, "params")], state):
        for key in layer.params:
            layer.params[key][...] = saved[key]
    return model


def metrics(model, data):
    x_train, x_test, y_train, y_test = data
    loss = CrossEntropy()
    p_test = model.forward(x_test)
    p_train = model.forward(x_train)
    return dict(train_accuracy=accuracy(p_train, y_train),
                test_accuracy=accuracy(p_test, y_test),
                train_loss=float(loss.forward(p_train, y_train)),
                test_loss=float(loss.forward(p_test, y_test)))


def endpoint_diagnostics(model, seed):
    data = dataset(seed)
    result = {}
    for split, x, y in (("train", data[0], data[2]), ("test", data[1], data[3])):
        probability = model.forward(x)
        truth = y.argmax(axis=1)
        predicted = probability.argmax(axis=1)
        confusion = np.zeros((2, 2), dtype=int)
        np.add.at(confusion, (truth, predicted), 1)
        result[split] = dict(confusion_matrix_true_rows_predicted_columns=confusion.tolist(),
                             minimum_true_class_probability=float(probability[np.arange(len(y)), truth].min()),
                             clipped_true_class_count=int((probability[np.arange(len(y)), truth] < 1e-12).sum()))
    return result


def equal_updates(seed, batch_size):
    """Run exactly 1500 updates, including each epoch's partial final batch.

    Counts record actual observations used by updates, not batch_size*updates:
    the final batch can be shorter (e.g. batch 128 leaves 32 observations).
    """
    data = dataset(seed)
    x_train, _, y_train, _ = data
    np.random.seed(seed)
    model = MLP(2, [20], 2)
    rng = np.random.default_rng(seed)
    loss = CrossEntropy()
    count = 0
    samples = 0
    completed_epochs = 0
    history = {k: [] for k in ["updates", "samples_seen", "epoch_equivalent",
                               "train_accuracy", "test_accuracy", "train_loss", "test_loss"]}
    while count < UPDATES:
        order = rng.permutation(len(x_train))
        shuffled_x, shuffled_y = x_train[order], y_train[order]
        for start in range(0, len(x_train), batch_size):
            if count == UPDATES:
                break
            xb = shuffled_x[start:start + batch_size]
            yb = shuffled_y[start:start + batch_size]
            prediction = model.forward(xb)
            model.backward(loss.backward(prediction, yb))
            for layer in model.layers:
                if hasattr(layer, "params"):
                    for key in ("weight", "bias"):
                        layer.params[key] -= LR * layer.grads[key]
            count += 1
            samples += len(xb)
            if count == 1 or count % 10 == 0 or count == UPDATES:
                history["updates"].append(count)
                history["samples_seen"].append(samples)
                history["epoch_equivalent"].append(samples / len(x_train))
                for key, value in metrics(model, data).items():
                    history[key].append(value)
        if count % math.ceil(len(x_train) / batch_size) == 0:
            completed_epochs += 1
    return model, history, samples, completed_epochs


def gradient_evidence():
    """Central differences on every parameter, including a deeper network."""
    rng = np.random.default_rng(123)
    criterion = CrossEntropy()
    rows = []
    for hidden in ([4], [4, 3]):
        for n in (1, 7):
            np.random.seed(20)
            model = MLP(2, list(hidden), 2)
            for layer in model.layers:
                if hasattr(layer, "params"):
                    layer.params["weight"][:] = rng.normal(0, 0.3, layer.params["weight"].shape)
                    layer.params["bias"][:] = rng.normal(0.2, 0.1, layer.params["bias"].shape)
            x = rng.normal(size=(n, 2))
            y = np.eye(2)[rng.integers(0, 2, n)]
            prediction = model.forward(x)
            model.backward(criterion.backward(prediction, y))
            tensors = [(index, layer, {k: v.copy() for k, v in layer.grads.items()})
                       for index, layer in enumerate(model.layers) if hasattr(layer, "params")]
            for index, layer, grads in tensors:
                for key, param in layer.params.items():
                    numerical = np.empty_like(param)
                    for coordinate in np.ndindex(param.shape):
                        original = param[coordinate]
                        param[coordinate] = original + 1e-6
                        plus = criterion.forward(model.forward(x), y)
                        param[coordinate] = original - 1e-6
                        minus = criterion.forward(model.forward(x), y)
                        param[coordinate] = original
                        numerical[coordinate] = (plus - minus) / 2e-6
                    analytic = grads[key]
                    relative = np.linalg.norm(analytic - numerical) / max(
                        1e-12, np.linalg.norm(analytic) + np.linalg.norm(numerical))
                    rows.append(dict(hidden=list(hidden), batch_size=n,
                                     layer_index=index, parameter=key,
                                     parameter_count=param.size,
                                     max_absolute_error=float(np.max(np.abs(analytic - numerical))),
                                     relative_l2_error=float(relative)))
    z = np.array([[-100.0, 100.0]])
    y = np.array([[1.0, 0.0]])
    p = SoftMax().forward(z)
    stable_loss = float(np.logaddexp.reduce(z[0]) - z[0, 0])
    return dict(method="central finite differences", epsilon=1e-6,
                relative_error_definition="norm(analytic-numerical)/(norm(analytic)+norm(numerical)), floor 1e-12",
                tensors=rows, max_absolute_error=max(r["max_absolute_error"] for r in rows),
                max_relative_l2_error=max(r["relative_l2_error"] for r in rows),
                clipped_loss_caveat=dict(logits=z.tolist(), target=y.tolist(), probabilities=p.tolist(),
                                         reported_clipped_loss=float(criterion.forward(p, y)),
                                         stable_logits_loss=stable_loss,
                                         explanation="Original probability clipping caps loss at -log(1e-12); backward still uses the unclipped softmax-cross-entropy derivative."))


def summarize(runs):
    summary = []
    for budget in ("equal_epoch", "equal_update"):
        for batch in BATCHES:
            selected = [r for r in runs if r["budget"] == budget and r["batch_size"] == batch]
            if len(selected) != len(SEEDS):
                continue
            row = dict(budget=budget, batch_size=batch, n_seeds=len(selected))
            for key in ("train_accuracy", "test_accuracy", "train_loss", "test_loss"):
                values = np.array([r["final"][key] for r in selected])
                row[key + "_mean"] = float(values.mean())
                row[key + "_sd"] = float(values.std(ddof=1))
                row[key + "_by_seed"] = {str(r["seed"]): r["final"][key] for r in selected}
            summary.append(row)
    return summary


def save_json(payload):
    """Checkpoint completed runs so an interrupted computation loses little work."""
    destination = HERE / "results" / "mlp.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["summary"] = summarize(payload["runs"])
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def save_figure(fig, name):
    folder = HERE / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (name + ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots(payload):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.18, "figure.facecolor": "white"})
    def get(budget, batch, seed=42):
        return next(r for r in payload["runs"]
                    if r["budget"] == budget and r["batch_size"] == batch and r["seed"] == seed)
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.7), sharex=True)
    for column, batch in enumerate((800, 1)):
        run = get("equal_epoch", batch)
        history = run["history"]
        for split, color, style in (("train", BLUE, "-"), ("test", ORANGE, "--")):
            axes[0, column].plot(history["epochs"], np.array(history[split + "_accuracy"]) * 100,
                                 color=color, linestyle=style, label=split.capitalize())
            axes[1, column].plot(history["epochs"], history[split + "_loss"],
                                 color=color, linestyle=style, label=split.capitalize())
        axes[0, column].set_title("Full batch (B=800)" if batch == 800 else "SGD (B=1)")
        axes[0, column].set_ylim(45, 102)
        axes[0, column].legend(loc="lower right", frameon=False)
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Mean cross entropy")
    fig.suptitle("Required default runs: identical data and initialization, seed 42", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "mlp_default_curves")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.85))
    for batch, color in zip(BATCHES, COLORS):
        for axis, budget, xkey in ((axes[0], "equal_epoch", "epochs"),
                                   (axes[1], "equal_update", "updates")):
            history = get(budget, batch)["history"]
            axis.plot(history[xkey], np.array(history["test_accuracy"]) * 100,
                      color=color, linewidth=1.2, label=str(batch))
    axes[0].set(title="Equal epochs: 1,500", xlabel="Epoch", ylabel="Test accuracy (%)", ylim=(45, 102))
    axes[1].set(title="Equal updates: 1,500", xlabel="Optimizer update", ylim=(45, 102))
    axes[1].legend(title="Batch size", ncol=2, frameon=False, loc="lower right")
    fig.suptitle("Learning trajectories depend on the comparison budget (seed 42)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "mlp_batch_trajectories")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.95))
    for budget, color, marker, label in (("equal_epoch", BLUE, "o", "1,500 epochs"),
                                        ("equal_update", ORANGE, "s", "1,500 updates")):
        rows = [next(r for r in payload["summary"] if r["budget"] == budget and r["batch_size"] == b)
                for b in BATCHES]
        for axis, metric, scale in ((axes[0], "test_accuracy", 100), (axes[1], "test_loss", 1)):
            axis.errorbar(BATCHES, [r[metric + "_mean"] * scale for r in rows],
                          yerr=[r[metric + "_sd"] * scale for r in rows],
                          color=color, marker=marker, markersize=4, capsize=3, label=label)
            axis.set_xscale("log")
            axis.set_xticks(BATCHES, [str(x) for x in BATCHES])
            axis.set_xlabel("Batch size (log scale)")
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[1].set_ylabel("Final test cross entropy")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Five independent pipeline seeds: mean ± one sample SD", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "mlp_budget_comparison")

    x_train, x_test, y_train, y_test = dataset(42)
    all_x = np.concatenate([x_train, x_test])
    gx = np.linspace(all_x[:, 0].min() - .2, all_x[:, 0].max() + .2, 220)
    gy = np.linspace(all_x[:, 1].min() - .2, all_x[:, 1].max() + .2, 180)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.9), sharex=True, sharey=True)
    for axis, batch in zip(axes, (800, 1, 8)):
        run = get("equal_epoch", batch)
        model = restore(run["parameters"])
        probability = model.forward(grid)[:, 1].reshape(xx.shape)
        image = axis.contourf(xx, yy, probability, levels=np.linspace(0, 1, 15), cmap="RdBu_r", alpha=.65)
        axis.contour(xx, yy, probability, levels=[.5], colors="black", linewidths=.9)
        for category, color in ((0, BLUE), (1, ORANGE)):
            train_mask = y_train[:, category] == 1
            test_mask = y_test[:, category] == 1
            axis.scatter(x_train[train_mask, 0], x_train[train_mask, 1], s=4,
                         color=color, alpha=.2, linewidths=0)
            axis.scatter(x_test[test_mask, 0], x_test[test_mask, 1], s=12,
                         color=color, edgecolors="white", linewidths=.35)
        wrong = model.forward(x_test).argmax(axis=1) != y_test.argmax(axis=1)
        axis.scatter(x_test[wrong, 0], x_test[wrong, 1], marker="x", s=25,
                     color="black", linewidths=.8)
        name = "Full batch" if batch == 800 else ("SGD" if batch == 1 else "Mini-batch")
        axis.set_title(f"{name}, B={batch}\nTest {run['final']['test_accuracy']:.1%}")
        axis.set_xlabel("Feature 1")
        axis.grid(False)
    axes[0].set_ylabel("Feature 2")
    fig.suptitle("Learned probability surfaces after 1,500 epochs (seed 42)", fontsize=11)
    fig.text(.5, .01, "Faint dots: train • white-edged dots: held-out test • black ×: test error • black line: p(class 1)=0.5",
             ha="center", fontsize=7)
    fig.tight_layout(rect=[0, .05, .95, 1])
    cax = fig.add_axes([.945, .25, .012, .47])
    fig.colorbar(image, cax=cax, label="p(class 1)", ticks=[0, .5, 1])
    save_figure(fig, "mlp_decision_surfaces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Replot from saved JSON without training")
    args = parser.parse_args()
    if args.plot_only:
        plots(json.loads((HERE / "results" / "mlp.json").read_text()))
        return
    started = time.perf_counter()
    payload = dict(schema_version=1,
                   config=dict(seeds=SEEDS, batch_sizes=BATCHES, hidden_units=[20],
                               learning_rate=LR, n_samples=1000, train_size=800,
                               test_size=200, noise=.2, equal_epoch_budget=EPOCHS,
                               equal_update_budget=UPDATES, eval_every=10,
                               weight_initialization="0.01 * np.random.randn; zero bias",
                               split="random stratified 80/20; original trainer",
                               seed_interpretation="Each seed jointly changes generated data, split, weight initialization and shuffle order.",
                               endpoint_policy="Final budget endpoint only. No checkpoint or hyperparameter selection on test set.",
                               uncertainty="Sample SD across five pipeline seeds, ddof=1; not a confidence interval.",
                               budget_caveat="Equal updates does not equalize processed observations, floating-point work or wall time."),
                   environment=dict(python=platform.python_version(),
                                    numpy=np.__version__,
                                    sklearn=importlib.metadata.version("scikit-learn"),
                                    matplotlib=matplotlib.__version__,
                                    threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}),
                   verification=gradient_evidence(), runs=[], summary=[])
    originals = {}
    for seed in SEEDS:
        for batch in BATCHES:
            run_started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                model, history = train("20", LR, EPOCHS, 10,
                                       batch_size=None if batch == 800 else batch,
                                       random_seed=seed)
            history["epochs"] = [s + 1 for s in history["steps"]]
            history["updates"] = [e * math.ceil(800 / batch) for e in history["epochs"]]
            history["samples_seen"] = [e * 800 for e in history["epochs"]]
            run = dict(budget="equal_epoch", seed=seed, batch_size=batch,
                       epochs=EPOCHS, optimizer_updates=EPOCHS * math.ceil(800 / batch),
                       samples_seen=EPOCHS * 800, epoch_equivalent=float(EPOCHS),
                       elapsed_seconds=time.perf_counter() - run_started,
                       history=history,
                       final={k: float(history[k][-1]) for k in
                              ("train_accuracy", "test_accuracy", "train_loss", "test_loss")})
            if seed == 42:
                run["parameters"] = parameters(model)
                run["diagnostics"] = endpoint_diagnostics(model, seed)
                originals[batch] = parameters(model)
            payload["runs"].append(run)
            save_json(payload)
            print(f"equal_epoch seed={seed} B={batch} test={run['final']['test_accuracy']:.3f} seconds={run['elapsed_seconds']:.2f}", flush=True)
    for seed in SEEDS:
        for batch in BATCHES:
            run_started = time.perf_counter()
            model, history, samples, completed = equal_updates(seed, batch)
            run = dict(budget="equal_update", seed=seed, batch_size=batch,
                       completed_epochs=completed, optimizer_updates=UPDATES,
                       samples_seen=samples, epoch_equivalent=samples / 800,
                       elapsed_seconds=time.perf_counter() - run_started,
                       history=history,
                       final={k: float(history[k][-1]) for k in
                              ("train_accuracy", "test_accuracy", "train_loss", "test_loss")})
            if seed == 42:
                run["parameters"] = parameters(model)
                run["diagnostics"] = endpoint_diagnostics(model, seed)
                if batch == 800:
                    difference = max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
                                     for a, b in zip(originals[800], run["parameters"])
                                     for k in a)
                    payload["verification"]["equal_update_vs_original_fullbatch_seed42_max_parameter_difference"] = difference
                    assert difference < 1e-12, difference
            payload["runs"].append(run)
            save_json(payload)
            print(f"equal_update seed={seed} B={batch} test={run['final']['test_accuracy']:.3f} exposure={samples}", flush=True)
    payload["compute_elapsed_seconds"] = time.perf_counter() - started
    save_json(payload)
    plots(payload)
    print(f"Completed 70 runs in {payload['compute_elapsed_seconds']:.1f} seconds; JSON and four figure pairs saved.", flush=True)


if __name__ == "__main__":
    main()
