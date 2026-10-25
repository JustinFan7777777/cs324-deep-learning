"""Verify saved assignment evidence without retraining the expensive study.

Run from any directory. By default this checks the saved 70-run experiment,
recomputes small finite-difference checks and saved model predictions, and
reproduces the three inexpensive original perceptron baselines. Pass
``--reproduce-defaults`` to additionally train the original seed-42 MLP for
1,500 epochs with full batch, SGD, and batch size 8 (about 25 seconds in the
recorded environment). No existing evidence or artifact is modified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "assignment1-mpl"))

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import mlp_experiments as experiment

SEEDS = [0, 1, 2, 3, 42]
BATCHES = [1, 8, 16, 32, 64, 128, 800]
METRICS = ("train_accuracy", "test_accuracy", "train_loss", "test_loss")
BUDGETS = ("equal_epoch", "equal_update")
ABS_GRAD_TOL = 1e-7
REL_GRAD_TOL = 1e-5
METRIC_TOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label, tolerance=1e-14):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
            f"{label}: nonfinite value")
    require(abs(float(actual) - float(expected)) <= tolerance,
            f"{label}: {actual!r} differs from {expected!r} (tolerance {tolerance})")


def numeric_tree_finite(value, label="evidence"):
    if isinstance(value, dict):
        for key, child in value.items():
            numeric_tree_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            numeric_tree_finite(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"{label} is nonfinite")


def sample_exposure(updates, batch):
    """Count actual examples, retaining each epoch's short final batch."""
    batches_per_epoch = math.ceil(800 / batch)
    completed, remaining = divmod(updates, batches_per_epoch)
    partial = sum(min(batch, 800 - start)
                  for start in list(range(0, 800, batch))[:remaining])
    return completed * 800 + partial


class Checks:
    def __init__(self):
        self.records = []

    def run(self, name, function):
        try:
            details = function()
        except Exception as error:
            self.records.append(dict(name=name, passed=False,
                                     error=f"{type(error).__name__}: {error}"))
            return None
        self.records.append(dict(name=name, passed=True, details=details))
        return details


def check_sources(perceptron):
    expected = perceptron["provenance"]["original_source_sha256"]
    paths = ["Part_1/perceptron.py", "Part_1/experiments.py"]
    require(set(expected) == set(paths), "Unexpected original Part I source list")
    for relative in paths:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        require(digest == expected[relative], f"Source hash mismatch: {relative}")
    return dict(source_files=len(paths), method="SHA-256 exact byte comparison")


def check_configuration(mlp):
    expected = dict(seeds=SEEDS, batch_sizes=BATCHES, hidden_units=[20],
                    learning_rate=0.01, n_samples=1000, train_size=800,
                    test_size=200, noise=0.2, equal_epoch_budget=1500,
                    equal_update_budget=1500, eval_every=10)
    for key, value in expected.items():
        require(mlp["config"][key] == value, f"Unexpected saved config {key}")
    require(experiment.SEEDS == SEEDS and experiment.BATCHES == BATCHES,
            "Experiment script seeds or batches differ")
    require(experiment.EPOCHS == 1500 and experiment.UPDATES == 1500
            and experiment.LR == 0.01, "Experiment script defaults differ")
    import train_mlp_numpy as original
    require(original.DNN_HIDDEN_UNITS_DEFAULT == "20"
            and original.LEARNING_RATE_DEFAULT == 0.01
            and original.MAX_EPOCHS_DEFAULT == 1500
            and original.EVAL_FREQ_DEFAULT == 10
            and original.BATCH_SIZE_DEFAULT == 0, "Original trainer defaults differ")
    require(mlp["schema_version"] == 1, "Unsupported MLP evidence schema")
    return dict(seeds=SEEDS, batch_sizes=BATCHES, default_epochs=1500,
                equal_update_budget=1500)


def check_mlp_runs(mlp):
    runs = mlp["runs"]
    expected = {(budget, seed, batch) for budget in BUDGETS
                for seed in SEEDS for batch in BATCHES}
    observed = [(run["budget"], run["seed"], run["batch_size"]) for run in runs]
    require(len(observed) == 70 and len(set(observed)) == 70
            and set(observed) == expected, "Missing, extra or duplicate MLP run")
    numeric_tree_finite(mlp, "mlp")
    for run in runs:
        tag = f"{run['budget']}/seed{run['seed']}/B{run['batch_size']}"
        batch = run["batch_size"]
        per_epoch = math.ceil(800 / batch)
        history = run["history"]
        if run["budget"] == "equal_epoch":
            steps = list(range(0, 1500, 10)) + [1499]
            epochs = [step + 1 for step in steps]
            expected_updates = [epoch * per_epoch for epoch in epochs]
            expected_samples = [epoch * 800 for epoch in epochs]
            require(history["steps"] == steps, f"{tag}: evaluation steps differ")
            require(history["epochs"] == epochs, f"{tag}: epoch labels differ")
            require(run["epochs"] == 1500, f"{tag}: endpoint epoch differs")
            total_updates = 1500 * per_epoch
            total_samples = 1200000
        else:
            expected_updates = [1] + list(range(10, 1501, 10))
            expected_samples = [sample_exposure(update, batch) for update in expected_updates]
            require(history["epoch_equivalent"] == [s / 800 for s in expected_samples],
                    f"{tag}: history exposure epochs differ")
            require(run["completed_epochs"] == 1500 // per_epoch,
                    f"{tag}: complete epoch count differs")
            total_updates = 1500
            total_samples = sample_exposure(1500, batch)
        require(history["updates"] == expected_updates, f"{tag}: update history differs")
        require(history["samples_seen"] == expected_samples, f"{tag}: exposure history differs")
        require(run["optimizer_updates"] == total_updates, f"{tag}: endpoint updates differ")
        require(run["samples_seen"] == total_samples, f"{tag}: endpoint exposure differs")
        require(run["epoch_equivalent"] == total_samples / 800,
                f"{tag}: endpoint equivalent epochs differ")
        for key in METRICS:
            values = np.asarray(history[key], dtype=float)
            require(len(values) == len(expected_updates), f"{tag}: {key} history length differs")
            require(np.isfinite(values).all(), f"{tag}: nonfinite {key}")
            require((values >= 0).all(), f"{tag}: negative {key}")
            if key.endswith("accuracy"):
                require((values <= 1).all(), f"{tag}: accuracy exceeds one")
            require(float(values[-1]) == run["final"][key], f"{tag}: final {key} differs")
        if run["seed"] == 42:
            params = run["parameters"]
            require(len(params) == 2, f"{tag}: wrong number of affine layers")
            for layer, shapes in zip(params, [((2, 20), (20,)), ((20, 2), (2,))]):
                require(np.asarray(layer["weight"]).shape == shapes[0]
                        and np.asarray(layer["bias"]).shape == shapes[1],
                        f"{tag}: wrong parameter shapes")
    return dict(runs=70, histories_checked=70,
                partial_batch_exposures={str(b): sample_exposure(1500, b) for b in BATCHES})


def check_mlp_summaries(mlp):
    rows = mlp["summary"]
    identities = [(row["budget"], row["batch_size"]) for row in rows]
    require(len(identities) == 14 and len(set(identities)) == 14
            and set(identities) == {(budget, batch) for budget in BUDGETS for batch in BATCHES},
            "Missing, extra or duplicate summary row")
    for row in rows:
        selected = [run for run in mlp["runs"] if run["budget"] == row["budget"]
                    and run["batch_size"] == row["batch_size"]]
        require(row["n_seeds"] == 5, "Summary seed count differs")
        for key in METRICS:
            values = np.array([run["final"][key] for run in selected])
            near(row[key + "_mean"], values.mean(), key + " mean")
            near(row[key + "_sd"], values.std(ddof=1), key + " sample SD")
            require(row[key + "_by_seed"] == {str(run["seed"]): run["final"][key]
                                              for run in selected}, "Per-seed summary differs")
    for seed in SEEDS:
        paired = [run for run in mlp["runs"] if run["seed"] == seed and run["batch_size"] == 800]
        for key in METRICS:
            near(paired[0]["final"][key], paired[1]["final"][key], "Full-batch control " + key,
                 tolerance=METRIC_TOL)
    near(mlp["verification"]["equal_update_vs_original_fullbatch_seed42_max_parameter_difference"],
         0, "Stored full-batch weight equality", tolerance=1e-12)
    return dict(summary_rows=14, metrics_per_row=4, sd_denominator="n-1",
                full_batch_control_seeds=5)


def check_gradients(mlp):
    current = experiment.gradient_evidence()
    saved = mlp["verification"]
    for record, label in ((saved, "saved"), (current, "recomputed")):
        rows = record["tensors"]
        require(len(rows) == 20, f"{label}: wrong gradient tensor count")
        require(sum(row["parameter_count"] for row in rows) == 114,
                f"{label}: wrong scalar gradient count")
        for row in rows:
            require(0 <= row["max_absolute_error"] < ABS_GRAD_TOL,
                    f"{label}: absolute gradient error exceeds tolerance")
            require(0 <= row["relative_l2_error"] < REL_GRAD_TOL,
                    f"{label}: relative gradient error exceeds tolerance")
        near(record["max_absolute_error"], max(row["max_absolute_error"] for row in rows),
             f"{label}: maximum absolute error")
        near(record["max_relative_l2_error"], max(row["relative_l2_error"] for row in rows),
             f"{label}: maximum relative error")
    identity = lambda row: (tuple(row["hidden"]), row["batch_size"], row["layer_index"], row["parameter"])
    require([identity(row) for row in saved["tensors"]]
            == [identity(row) for row in current["tensors"]], "Gradient cases differ")
    caveat = current["clipped_loss_caveat"]
    near(caveat["stable_logits_loss"], 200, "Stable logit loss")
    near(caveat["reported_clipped_loss"], -math.log(1e-12), "Clipped loss")
    return dict(scalar_parameters_checked=114, tensors=20,
                max_absolute_error=current["max_absolute_error"],
                max_relative_l2_error=current["max_relative_l2_error"],
                absolute_tolerance=ABS_GRAD_TOL, relative_tolerance=REL_GRAD_TOL)


def check_saved_models(mlp):
    rows = [run for run in mlp["runs"] if run["seed"] == 42]
    largest = 0.0
    for row in rows:
        model = experiment.restore(row["parameters"])
        require(all(np.isfinite(value).all() for layer in model.layers
                    if hasattr(layer, "params") for value in layer.params.values()),
                "Nonfinite saved model parameters")
        actual = experiment.metrics(model, experiment.dataset(42))
        for key in METRICS:
            largest = max(largest, abs(actual[key] - row["final"][key]))
            near(actual[key], row["final"][key], "Saved model " + key, METRIC_TOL)
        diagnostic = experiment.endpoint_diagnostics(model, 42)
        for split in ("train", "test"):
            saved = row["diagnostics"][split]
            actual_diagnostic = diagnostic[split]
            require(actual_diagnostic["confusion_matrix_true_rows_predicted_columns"]
                    == saved["confusion_matrix_true_rows_predicted_columns"],
                    "Saved confusion matrix differs")
            require(actual_diagnostic["clipped_true_class_count"]
                    == saved["clipped_true_class_count"], "Saved clipping count differs")
            near(actual_diagnostic["minimum_true_class_probability"],
                 saved["minimum_true_class_probability"],
                 "Minimum target probability", METRIC_TOL)
            require(diagnostic[split]["clipped_true_class_count"] == 0,
                    "A saved seed-42 endpoint clips a target probability")
    return dict(seed42_endpoints_recomputed=14, largest_metric_difference=largest,
                clipping_claim_scope="Seed-42 train/test final endpoints only; all seven batches and both budgets.")


def check_perceptron(perceptron):
    numeric_tree_finite(perceptron, "perceptron")
    cfg = perceptron["config"]
    require(cfg["max_epochs"] == 1000 and cfg["learning_rate"] == 0.01
            and cfg["train_per_class"] == 80 and cfg["test_per_class"] == 20,
            "Perceptron defaults differ")
    sys.path.insert(0, str(ROOT / "Part_1"))
    from experiments import create_dataset
    from perceptron import Perceptron
    require([row["key"] for row in perceptron["baselines"]]
            == ["separated", "close_means", "high_variance"], "Wrong Part I cases")
    for row in perceptron["baselines"]:
        require(row["random_seed"] == 42, "Wrong Part I baseline seed")
        x_train, y_train, x_test, y_test = create_dataset(
            row["mean_negative"], row["mean_positive"], row["variance"], random_seed=42)
        require(len(x_train) == 160 and len(x_test) == 40, "Part I split differs")
        model = Perceptron(2, max_epochs=1000, learning_rate=0.01)
        model.train(x_train, y_train)
        history = row["history"]
        require([item["epoch"] for item in history] == list(range(len(history))),
                "Part I history epochs are not consecutive")
        require(np.array_equal(model.weights, np.asarray(history[-1]["weights"])),
                "Part I terminal parameters differ from original trainer")
        for x, y, key in ((x_train, y_train, "train_accuracy"), (x_test, y_test, "test_accuracy")):
            near(np.mean(model.forward(x) == y), history[-1][key], "Part I " + key)
        require(row["trajectory_matches_original_weights_exactly"] is True,
                "Missing Part I trajectory equivalence")
    sweep = perceptron["sweep"]
    require(sweep["seeds"] == [0, 1, 2, 3, 4] and len(sweep["cells"]) == 36,
            "Wrong Part I sweep size or seeds")
    expected = {(distance, variance) for distance in sweep["distances"] for variance in sweep["variances"]}
    observed = [(cell["distance"], cell["variance"]) for cell in sweep["cells"]]
    require(len(set(observed)) == 36 and set(observed) == expected, "Part I sweep grid differs")
    for cell in sweep["cells"]:
        rows = cell["runs"]
        require([row["seed"] for row in rows] == sweep["seeds"], "Part I sweep seeds differ")
        scores = np.asarray([row["test_accuracy"] for row in rows])
        gaps = np.asarray([row["oracle_test_accuracy"] - row["test_accuracy"] for row in rows])
        near(cell["mean_test_accuracy"], scores.mean(), "Part I sweep mean")
        near(cell["sd_test_accuracy"], scores.std(ddof=1), "Part I sweep SD")
        near(cell["mean_paired_oracle_minus_perceptron"], gaps.mean(), "Part I oracle gap")
    return dict(original_baselines_reproduced=3, terminal_weights="bit-identical",
                sweep_cells=36, sweep_saved_runs=180)


def reproduce_defaults(mlp):
    details = []
    for batch in (800, 1, 8):
        saved = next(run for run in mlp["runs"] if run["budget"] == "equal_epoch"
                     and run["seed"] == 42 and run["batch_size"] == batch)
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            model, history = experiment.train("20", 0.01, 1500, 10,
                                               batch_size=None if batch == 800 else batch,
                                               random_seed=42)
        final = {key: float(history[key][-1]) for key in METRICS}
        for key in METRICS:
            near(final[key], saved["final"][key], f"Reproduced B{batch} {key}", METRIC_TOL)
        require(all(np.isfinite(value).all() for layer in model.layers
                    if hasattr(layer, "params") for value in layer.params.values()),
                f"Reproduced B{batch}: nonfinite model parameters")
        difference = max(float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
                         for actual, expected in zip(experiment.parameters(model), saved["parameters"])
                         for key in ("weight", "bias"))
        require(difference <= METRIC_TOL, f"Reproduced B{batch}: parameter difference {difference}")
        details.append(dict(batch_size=batch, final=final, maximum_parameter_difference=difference,
                            elapsed_seconds=round(time.perf_counter() - started, 3)))
    return dict(original_trainer_runs=details, metric_tolerance=METRIC_TOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce-defaults", action="store_true",
                        help="Additionally rerun original full batch, SGD and batch-8 defaults (~25 s)")
    parser.add_argument("--output", type=Path,
                        help="Write this actual verification result as JSON to the requested path")
    args = parser.parse_args()
    started = time.perf_counter()
    checks = Checks()
    evidence = {}
    def load_evidence():
        for name in ("mlp", "perceptron"):
            evidence[name] = json.loads((HERE / "results" / (name + ".json")).read_text())
        return dict(loaded=["results/mlp.json", "results/perceptron.json"])
    loaded = checks.run("load_saved_evidence", load_evidence)
    if loaded is not None:
        mlp, perceptron = evidence["mlp"], evidence["perceptron"]
        checks.run("original_part1_source_hashes", lambda: check_sources(perceptron))
        checks.run("required_mlp_defaults", lambda: check_configuration(mlp))
        checks.run("all_mlp_histories_endpoints_and_counts", lambda: check_mlp_runs(mlp))
        checks.run("five_seed_summaries_and_fullbatch_control", lambda: check_mlp_summaries(mlp))
        checks.run("fresh_finite_difference_checks", lambda: check_gradients(mlp))
        checks.run("recompute_saved_seed42_models", lambda: check_saved_models(mlp))
        checks.run("part1_baselines_and_saved_sweep_summaries", lambda: check_perceptron(perceptron))
        if args.reproduce_defaults:
            checks.run("fresh_original_mlp_default_training", lambda: reproduce_defaults(mlp))
    report = dict(schema_version=1, passed=all(item["passed"] for item in checks.records),
                  mode="reproduce-defaults" if args.reproduce_defaults else "saved-evidence-and-small-checks",
                  elapsed_seconds=round(time.perf_counter() - started, 3), checks=checks.records)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
