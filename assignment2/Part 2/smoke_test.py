"""Validate the CNN pipeline on small, explicitly labelled CIFAR10 subsets."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, Subset

from cnn_model import CNN
from cnn_train import DATA_DIR_DEFAULT, atomic_save, fit, make_loaders


def main():
    # One CPU thread and six updates keep this check inexpensive.
    torch.set_num_threads(1)
    torch.manual_seed(42)
    full_train, _, full_test = make_loaders(DATA_DIR_DEFAULT, download=False)
    train_data = Subset(full_train.dataset, range(16))
    test_data = Subset(full_test.dataset, range(8))
    train_loader = DataLoader(
        train_data, batch_size=4, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    train_eval = DataLoader(train_data, batch_size=4)
    test_loader = DataLoader(test_data, batch_size=4)

    output_root = Path(__file__).resolve().parent / "results" / "smoke" / "cpu"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=output_root))
    config = {
        "scope": "smoke", "data_dir": DATA_DIR_DEFAULT, "device": "cpu",
        "seed": 42, "batch_size": 4, "max_steps": 6, "eval_freq": 2,
        "learning_rate": 0.001, "train_indices": list(range(16)),
        "test_indices": list(range(8)), "cpu_threads": 1,
    }
    atomic_save(run_dir / "config.json", config)
    model = CNN()
    history = fit(
        model, train_loader, train_eval, test_loader,
        max_steps=6, eval_freq=2, run_dir=run_dir, config=config,
    )
    assert history["steps"] == [2, 4, 6]
    assert history["samples_seen"] == [8, 16, 24]
    assert all(torch.isfinite(torch.tensor(values)).all() for values in history.values())

    # Check the saved final model against the current model on real images.
    restored = CNN()
    restored.load_state_dict(torch.load(
        run_dir / "model.pt", map_location="cpu", weights_only=True,
    ))
    model.eval()
    restored.eval()
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    print(f"PASS: real CIFAR10 subset smoke test and checkpoint reload. Results: {run_dir}")


if __name__ == "__main__":
    main()
