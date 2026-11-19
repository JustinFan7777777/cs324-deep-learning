"""CIFAR10 training and evaluation for the assignment CNN."""

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import CNN


# Adam's default learning rate is 0.001.
# The assignment PDF takes precedence over the starter's 0.0001.
LEARNING_RATE_DEFAULT = 1e-3

BATCH_SIZE_DEFAULT = 32

# This is a count of optimizer updates, not complete epochs.
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500

# Store CIFAR10 under assignment2/data regardless of the working directory.
DATA_DIR_DEFAULT = str(
    Path(__file__).resolve().parent.parent / "data"
)


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute the fraction of correct predictions.

    Args:
        predictions: Logits or probabilities with shape (B, C).
        targets: Class indices (B,) or one-hot labels (B, C).

    Returns:
        Accuracy in [0, 1].
    """
    if targets.ndim == 2:
        targets = targets.argmax(dim=1)

    return (
        predictions.argmax(dim=1)
        .eq(targets)
        .float()
        .mean()
        .item()
    )


def make_loaders(
    data_dir,
    batch_size=32,
    seed=42,
    download=False,
):
    """
    Create training and evaluation data loaders.

    Returns:
        train_loader: Shuffled batches used for parameter updates.
        train_eval_loader: Ordered batches used for training-set metrics.
        test_loader: Ordered batches used for test-set metrics.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    transform = transforms.Compose([
        # PIL image (H, W, C), uint8 [0, 255]
        # -> tensor (C, H, W), float32 [0, 1].
        transforms.ToTensor(),

        # Apply (x - mean) / std independently to each channel.
        # With mean=std=0.5, [0, 1] maps to [-1, 1].
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    train_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )

    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    # Keep the training-order generator separate from evaluation.
    # Evaluation should not change the ordering of future training batches.
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        drop_last=False,
    )

    # Use a separate loader for evaluation on the training set.
    # It visits all examples without shuffling them.
    train_eval_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed + 2),
    )

    # num_workers=0 keeps data loading in the main process.
    # This is a straightforward starting point for macOS and notebooks.
    return train_loader, train_eval_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    Compute metrics over every example yielded by a loader.

    criterion must return the mean loss of each batch.
    """
    if len(loader.dataset) == 0:
        raise ValueError("Evaluation dataset must be non-empty.")

    device = next(model.parameters()).device
    was_training = model.training

    # BatchNorm uses running statistics in evaluation mode.
    # Its running statistics must not be updated using test examples.
    model.eval()

    # Accumulate on the device and synchronize only when returning metrics.
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0

    try:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_count = targets.size(0)

            # Convert the batch-mean loss into a batch-summed loss.
            # This handles a smaller final batch correctly.
            loss_sum += loss.detach() * batch_count

            # Count individual correct predictions.
            correct += (
                logits.argmax(dim=1)
                .eq(targets)
                .sum()
            )
            count += batch_count

    finally:
        # Restore the original mode even if evaluation raises an error.
        model.train(was_training)

    if count == 0:
        raise ValueError("Evaluation loader yielded no samples.")

    return {
        "loss": loss_sum.item() / count,
        "accuracy": correct.item() / count,
    }


def atomic_save(path, value, *, binary=False):
    """Replace one artifact only after its temporary file is fully written."""
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


def fit(
    model,
    train_loader,
    train_eval_loader,
    test_loader,
    max_steps=5000,
    eval_freq=500,
    learning_rate=1e-3,
    run_dir=None,
    config=None,
):
    """
    Run exactly max_steps optimizer updates.

    When the current training iterator is exhausted, start a new
    shuffled epoch and continue until the update budget is reached.
    """
    if max_steps <= 0 or eval_freq <= 0:
        raise ValueError("max_steps and eval_freq must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")

    if len(train_loader) == 0:
        raise ValueError("Training loader must be non-empty.")

    device = next(model.parameters()).device

    # CrossEntropyLoss receives logits (B, 10) and class indices (B,).
    # Its default reduction is the mean over the batch.
    criterion = nn.CrossEntropyLoss()

    # Leave betas, eps, weight_decay, and other Adam options at defaults.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        key: []
        for key in (
            "steps",
            "samples_seen",
            "epoch_equivalent",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
        )
    }

    iterator = iter(train_loader)
    samples_seen = 0
    model.train()

    for step in range(1, max_steps + 1):
        try:
            images, targets = next(iterator)

        except StopIteration:
            # All batches in this epoch have been consumed.
            # A new iterator creates a fresh shuffled pass.
            iterator = iter(train_loader)
            images, targets = next(iterator)

        images = images.to(device)
        targets = targets.to(device)

        # Clear gradients accumulated by the previous update.
        optimizer.zero_grad(set_to_none=True)

        # Forward:
        #   images (B, 3, 32, 32) -> logits (B, 10).
        logits = model(images)

        # logits (B, 10), targets (B,) -> scalar mean loss.
        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite loss at update {step}."
            )

        # Backward computes parameter gradients.
        loss.backward()

        # Adam updates parameters using its moment estimates.
        optimizer.step()

        # Count actual examples, including a potentially shorter batch.
        samples_seen += targets.size(0)

        should_evaluate = (
            step % eval_freq == 0
            or step == max_steps
        )

        if should_evaluate:
            # Both evaluations use the same model weights and
            # the same BatchNorm running statistics.
            train_metrics = evaluate(
                model,
                train_eval_loader,
                criterion,
            )
            test_metrics = evaluate(
                model,
                test_loader,
                criterion,
            )

            history["steps"].append(step)
            history["samples_seen"].append(samples_seen)

            # This records example exposure in units of dataset size.
            # A value of 1.5 means 1.5 training-set equivalents were seen.
            history["epoch_equivalent"].append(
                samples_seen / len(train_loader.dataset)
            )

            for split, metrics in (
                ("train", train_metrics),
                ("test", test_metrics),
            ):
                history[f"{split}_loss"].append(metrics["loss"])
                history[f"{split}_accuracy"].append(
                    metrics["accuracy"]
                )

            print(
                f"Update {step}/{max_steps} | "
                f"Train loss {train_metrics['loss']:.4f} | "
                f"Train accuracy {train_metrics['accuracy']:.2%} | "
                f"Test loss {test_metrics['loss']:.4f} | "
                f"Test accuracy {test_metrics['accuracy']:.2%}"
            )

            if run_dir is not None:
                # Save at each evaluation, including the final update.
                # The checkpoint holds a consistent model/optimizer/history
                # snapshot; history.json is a lightweight plotting export.
                payload = {
                    "config": config or {},
                    "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
                    "torch_version": str(torch.__version__),
                    "torchvision_version": str(torchvision.__version__),
                    "status": "complete" if step == max_steps else "running",
                    "completed_steps": step,
                    "train_examples": len(train_loader.dataset),
                    "train_eval_examples": len(train_eval_loader.dataset),
                    "test_examples": len(test_loader.dataset),
                    "history": history,
                }
                state = {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
                checkpoint = {
                    **payload,
                    "model_state_dict": state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "samples_seen": samples_seen,
                }
                atomic_save(Path(run_dir) / "checkpoint.pt", checkpoint, binary=True)
                atomic_save(Path(run_dir) / "history.json", payload)
                if step == max_steps:
                    atomic_save(Path(run_dir) / "model.pt", state, binary=True)

    return history


def train(
    data_dir=DATA_DIR_DEFAULT,
    batch_size=BATCH_SIZE_DEFAULT,
    max_steps=MAX_STEPS_DEFAULT,
    eval_freq=EVAL_FREQ_DEFAULT,
    learning_rate=LEARNING_RATE_DEFAULT,
    seed=42,
    device="cpu",
    download=False,
    output_dir=None,
):
    """
    Construct the experiment and return its final model and history.

    device can be "cpu", "mps", or "cuda" when supported locally.
    If output_dir is supplied, create a unique run subdirectory there.
    """
    if max_steps <= 0 or eval_freq <= 0:
        raise ValueError("max_steps and eval_freq must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    requested_device = torch.device(device)
    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Select device='cpu'.")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select an available device.")

    config = {
        "data_dir": str(data_dir), "batch_size": batch_size,
        "max_steps": max_steps, "eval_freq": eval_freq,
        "learning_rate": learning_rate, "seed": seed,
        "device": str(device), "download": download,
    }
    run_dir = None
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        # mkdtemp reserves a unique directory even for concurrent launches.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=output_root))
        atomic_save(run_dir / "config.json", config)
        print(f"Run directory: {run_dir.resolve()}", flush=True)

    # Controls model initialization.
    # Training-data ordering has its own seeded generator.
    torch.manual_seed(seed)

    loaders = make_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        download=download,
    )

    model = CNN(
        n_channels=3,
        n_classes=10,
    ).to(device)

    # *loaders passes the three loaders as positional arguments.
    history = fit(
        model,
        *loaders,
        max_steps=max_steps,
        eval_freq=eval_freq,
        learning_rate=learning_rate,
        run_dir=run_dir,
        config=config,
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train the assignment CNN."
    )

    parser.add_argument(
        "--data_dir",
        default=DATA_DIR_DEFAULT,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR10 if it is not already available.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/cnn",
        help="Parent directory for unique run folders and periodic checkpoints.",
    )

    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
