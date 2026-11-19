"""Train a hand-written RNN to predict the final digit of a palindrome."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN


def atomic_save(path, value, *, binary=False):
    """Publish a complete JSON/checkpoint file with an atomic replacement."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".saving-")
    try:
        with os.fdopen(descriptor, "wb" if binary else "w") as stream:
            if binary:
                torch.save(value, stream)
            else:
                json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_eval_loader(
    input_length,
    sample_count,
    batch_size,
    seed,
):
    """
    Build a finite, fixed evaluation set.

    Evaluation samples use a separate NumPy seed. Restore the previous
    RNG state afterward so training samples are unaffected.
    """
    if min(input_length, sample_count, batch_size) <= 0:
        raise ValueError("Evaluation lengths and sizes must be positive.")
    previous_state = np.random.get_state()

    try:
        np.random.seed(seed)

        source = PalindromeDataset(input_length + 1)

        # Materialize the evaluation examples once.
        # Repeated evaluations will use these same examples.
        examples = [
            source[index]
            for index in range(sample_count)
        ]

    finally:
        np.random.set_state(previous_state)

    # Each input has shape (input_length,).
    # Stacking produces (sample_count, input_length).
    inputs = torch.from_numpy(
        np.stack([item[0] for item in examples])
    )

    # CrossEntropyLoss expects integer class indices.
    targets = torch.tensor(
        [item[1] for item in examples],
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,

        # Keep evaluation-loader randomness separate from the
        # global PyTorch generator.
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(model, loader):
    """Compute sample-weighted loss and accuracy on fixed examples."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    count = 0

    try:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            # Sum per-example losses so a smaller final batch
            # receives its correct weight in the dataset mean.
            loss_sum += nn.functional.cross_entropy(
                logits,
                targets,
                reduction="sum",
            )

            correct += (
                logits.argmax(dim=1)
                .eq(targets)
                .sum()
            )

            count += targets.numel()

    finally:
        model.train(was_training)

    if count == 0:
        raise ValueError(
            "Evaluation loader must contain samples."
        )

    return {
        "loss": loss_sum.item() / count,
        "accuracy": correct.item() / count,
    }


def train(config):
    """
    Train for exactly config.train_steps parameter updates.

    Returns:
        model: The final trained model.
        history: Recorded training-batch and evaluation metrics.
    """
    if min(
        config.input_length,
        config.num_hidden,
        config.batch_size,
        config.train_steps,
        config.eval_freq,
        config.eval_samples,
    ) <= 0:
        raise ValueError(
            "Lengths, sizes, and step counts must be positive."
        )

    # The supplied dataset uses scalar digits and ten digit classes.
    if config.input_dim != 1 or config.num_classes != 10:
        raise ValueError(
            "The supplied scalar-digit dataset requires "
            "input_dim=1 and num_classes=10."
        )

    if (
        not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be finite and positive."
        )

    if (
        not math.isfinite(config.max_norm)
        or config.max_norm <= 0
    ):
        raise ValueError(
            "max_norm must be finite and positive."
        )

    device = torch.device(config.device)

    if (
        device.type == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise ValueError(
            "MPS is unavailable. Use --device cpu."
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CUDA is unavailable. Select an available device."
        )

    num_threads = getattr(config, "num_threads", 1)
    if num_threads <= 0:
        raise ValueError("num_threads must be positive.")
    torch.set_num_threads(num_threads)

    # A checkpoint is read on CPU; optimizer.load_state_dict moves state
    # to the corresponding parameter device when needed.
    resume = getattr(config, "resume", None)
    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported RNN checkpoint schema.")
        for key in (
            "input_length", "input_dim", "num_hidden", "num_classes",
            "batch_size", "learning_rate", "max_norm", "seed",
            "eval_samples", "eval_freq",
        ):
            if getattr(config, key) != checkpoint["config"][key]:
                raise ValueError(f"Resume configuration differs for {key}.")
        if config.train_steps <= checkpoint["completed_steps"]:
            raise ValueError("train_steps must exceed the checkpoint's completed steps.")

    # NumPy controls palindrome generation.
    # PyTorch controls model initialization.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = VanillaRNN(
        seq_length=config.input_length,
        input_dim=config.input_dim,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
        batch_size=config.batch_size,
    ).to(device)

    # Keep the +1: the dataset length includes the target digit.
    dataset = PalindromeDataset(
        config.input_length + 1
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,

        # Each access already generates a new random palindrome.
        # Randomly permuting sys.maxsize indices would be impractical.
        shuffle=False,

        # Generate samples in this process using the NumPy seed above.
        num_workers=0,

        generator=torch.Generator().manual_seed(config.seed),
    )

    eval_loader = make_eval_loader(
        input_length=config.input_length,
        sample_count=config.eval_samples,
        batch_size=config.batch_size,
        seed=config.seed + 10000,
    )

    # Inputs to the loss:
    #   logits:  (B, 10)
    #   targets: (B,)
    #
    # Each example contributes one final-digit prediction.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = {
        key: []
        for key in (
            "steps",
            "batch_loss",
            "batch_accuracy",
            "gradient_norm_before_clip",
            "eval_loss",
            "eval_accuracy",
        )
    }

    iterator = iter(train_loader)
    start_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        start_step = checkpoint["completed_steps"]
        state = checkpoint["numpy_rng_state"]
        np.random.set_state((
            state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
            state["position"], state["has_gauss"], state["cached_gaussian"],
        ))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        # With num_workers=0, __getitem__ ignores its index and draws
        # from NumPy. Restoring that state restores the next sample.

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(
        prefix=f"T{config.input_length + 1}-seed{config.seed}-",
        dir=output_root,
    ))
    settings = dict(vars(config))
    settings["output_dir"] = str(config.output_dir)
    settings["resume"] = str(resume) if resume is not None else None
    settings["num_threads"] = num_threads
    atomic_save(run_dir / "config.json", settings)
    print(f"Run directory: {run_dir.resolve()}", flush=True)

    model.train()

    # Starting at 1 and including train_steps performs exactly
    # train_steps updates, avoiding an off-by-one stopping rule.
    for step in range(start_step + 1, config.train_steps + 1):
        inputs, targets = next(iterator)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Model internally expands scalar inputs:
        #   (B, input_length) -> (B, input_length, 1)
        #
        # It then returns final-step logits:
        #   (B, 10)
        logits = model(inputs)

        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite loss at update {step}."
            )

        # Build gradients through every recurrent time step.
        loss.backward()

        # Clip the global norm of the parameter-gradient vector.
        #
        # This must happen after backward() creates the gradients,
        # and before optimizer.step() uses them.
        #
        # The returned value is the norm BEFORE clipping.
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.max_norm,
            error_if_nonfinite=True,
        )

        optimizer.step()

        if (
            step % config.eval_freq == 0
            or step == config.train_steps
        ):
            # Fixed-set evaluation uses the updated model.
            metrics = evaluate(model, eval_loader)

            # These batch metrics describe the forward pass that
            # produced this update, before the parameter change.
            batch_accuracy = (
                logits.detach()
                .argmax(dim=1)
                .eq(targets)
                .float()
                .mean()
                .item()
            )

            history["steps"].append(step)
            history["batch_loss"].append(loss.item())
            history["batch_accuracy"].append(batch_accuracy)
            history["gradient_norm_before_clip"].append(
                gradient_norm.item()
            )
            history["eval_loss"].append(metrics["loss"])
            history["eval_accuracy"].append(
                metrics["accuracy"]
            )

            print(
                f"T={config.input_length + 1} | "
                f"Step {step}/{config.train_steps} | "
                f"Batch loss {loss.item():.4f} | "
                f"Eval loss {metrics['loss']:.4f} | "
                f"Eval accuracy {metrics['accuracy']:.2%}"
            )

            # Save at every evaluation so interruption preserves progress.
            payload = {
                "schema_version": 1,
                "config": settings,
                "palindrome_length": config.input_length + 1,
                "completed_steps": step,
                "samples_seen": step * config.batch_size,
                "status": "complete" if step == config.train_steps else "running",
                "initialization": "weights uniform +/- 1/sqrt(hidden_dim); zero biases",
                "evaluation_protocol": (
                    "fixed independent random draws; sequence overlap with training is possible"
                ),
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                "history": history,
            }
            state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
            rng = np.random.get_state()
            saved_checkpoint = {
                **payload,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                # Plain values keep the checkpoint compatible with weights_only=True.
                "numpy_rng_state": {
                    "algorithm": rng[0], "keys": rng[1].tolist(),
                    "position": rng[2], "has_gauss": rng[3],
                    "cached_gaussian": rng[4],
                },
                "torch_rng_state": torch.get_rng_state(),
            }
            atomic_save(run_dir / "checkpoint.pt", saved_checkpoint, binary=True)
            atomic_save(run_dir / "history.json", payload)
            if step == config.train_steps:
                atomic_save(run_dir / "model.pt", state, binary=True)

    print(f"Saved results to {run_dir.resolve()}")

    return model, history


def parse_args(argv=None):
    """
    Parse configuration.

    Passing [] returns the defaults without reading notebook arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a manual palindrome RNN."
    )

    # Preserve the supplied starter defaults.
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--max_norm", type=float, default=10.0)

    # Additional controls for reproducible evaluation and saved results.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument(
        "--resume", default=None,
        help="Checkpoint to continue; train_steps is the total target update count.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--output_dir",
        default="results/rnn",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
