"""Run a controlled palindrome-length study and preserve each completed run."""

import argparse
from pathlib import Path
import tempfile

from train import atomic_save, parse_args, train


def run_study(lengths, seeds, train_steps=10000, eval_freq=500,
              eval_samples=1000, device="cpu", output_dir="results/rnn_length_study"):
    """Compare independently trained models at a fixed optimizer-update budget."""
    if not lengths or min(lengths) < 2 or len(set(lengths)) != len(lengths):
        raise ValueError("Supply distinct palindrome lengths of at least two.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least one distinct seed.")
    if min(train_steps, eval_freq, eval_samples) <= 0:
        raise ValueError("Training and evaluation counts must be positive.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    study_dir = Path(tempfile.mkdtemp(prefix="study-", dir=root))
    summary = {
        "status": "running",
        "planned_lengths": list(lengths), "seeds": list(seeds),
        "train_steps": train_steps, "eval_samples": eval_samples,
        "comparison_budget": "equal optimizer updates and batch size",
        "records": [],
    }
    atomic_save(study_dir / "summary.json", summary)

    for length in lengths:
        for seed in seeds:
            config = parse_args([])
            config.input_length = length - 1
            config.seed = seed
            config.train_steps = train_steps
            config.eval_freq = eval_freq
            config.eval_samples = eval_samples
            config.device = device
            # Each pair owns a separate parent folder; train adds its unique run folder.
            run_root = study_dir / "runs" / f"T{length}-seed{seed}"
            config.output_dir = str(run_root)
            model, history = train(config)
            history_path = next(run_root.glob("*/history.json"))
            summary["records"].append({
                "palindrome_length": length,
                "seed": seed,
                "train_steps": train_steps,
                "eval_samples": eval_samples,
                "final_eval_loss": history["eval_loss"][-1],
                "final_eval_accuracy": history["eval_accuracy"][-1],
                "history_path": str(history_path.relative_to(study_dir)),
            })
            atomic_save(study_dir / "summary.json", summary)
            del model

    summary["status"] = "complete"
    atomic_save(study_dir / "summary.json", summary)
    print(f"Study results: {(study_dir / 'summary.json').resolve()}")
    return study_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 9, 13, 17, 21])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output_dir", default=str(
        Path(__file__).resolve().parent / "results" / "rnn_length_study"
    ))
    args = vars(parser.parse_args())
    run_study(**args)


if __name__ == "__main__":
    main()
