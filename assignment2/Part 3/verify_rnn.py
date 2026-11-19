"""Lightweight recurrence, gradient, evaluation, and recovery regression checks."""

import contextlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.func import functional_call

from dataset import PalindromeDataset
from train import atomic_save, evaluate, make_eval_loader, parse_args, train
from vanilla_rnn import VanillaRNN


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    for length in (2, 3, 4, 5, 10, 11):
        dataset = PalindromeDataset(length)
        for _ in range(5):
            sequence = dataset.generate_palindrome()
            assert len(sequence) == length
            np.testing.assert_array_equal(sequence, sequence[::-1])
            inputs, target = dataset[0]
            assert inputs.shape == (length - 1,) and inputs[0] == target
    print("PASS palindrome symmetry and input/target alignment")

    model = VanillaRNN(4, 1, 128, 10, 128)
    assert sum(p.numel() for p in model.parameters()) == 17930
    for batch in (1, 7):
        inputs = torch.randn(batch, 4, 1)
        hidden = torch.zeros(batch, 128)
        for index in range(4):
            hidden = torch.tanh(
                inputs[:, index] @ model.input_to_hidden.weight.T
                + hidden @ model.hidden_to_hidden.weight.T
                + model.input_to_hidden.bias
            )
        expected = hidden @ model.hidden_to_output.weight.T + model.hidden_to_output.bias
        torch.testing.assert_close(model(inputs), expected)
        torch.testing.assert_close(model(inputs.squeeze(-1)), expected)
        torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)
    print("PASS recurrence equations, zero hidden state, dynamic batch shapes")

    # Central finite differences cover every parameter and the input tensor.
    small = VanillaRNN(3, 1, 2, 2, 2).double()
    names, parameters = zip(*small.named_parameters())
    inputs = torch.randn(2, 3, 1, dtype=torch.double, requires_grad=True)
    targets = torch.tensor([0, 1])

    def objective(x, *values):
        logits = functional_call(small, dict(zip(names, values)), (x,))
        return torch.nn.functional.cross_entropy(logits, targets)

    assert torch.autograd.gradcheck(objective, (inputs, *parameters))
    print("PASS finite-difference input and parameter gradients through time")

    state = np.random.get_state()
    loader = make_eval_loader(4, 17, 8, 10042)
    observed = np.random.randint(0, 10, 20)
    np.random.set_state(state)
    np.testing.assert_array_equal(observed, np.random.randint(0, 10, 20))
    for mode in (True, False):
        model.train(mode)
        metrics = evaluate(model, loader)
        assert model.training == mode
    model.eval()
    with torch.no_grad():
        x, y = loader.dataset.tensors
        expected_loss = torch.nn.functional.cross_entropy(model(x), y).item()
    assert abs(metrics["loss"] - expected_loss) < 1e-6
    print("PASS independent evaluation RNG, partial-batch weighting, mode restoration")

    with tempfile.TemporaryDirectory(prefix="rnn-regression-") as directory:
        config = parse_args([])
        config.input_length = 4
        config.num_hidden = 8
        config.batch_size = 4
        config.eval_samples = 17
        config.eval_freq = 2
        config.train_steps = 6
        config.output_dir = str(Path(directory) / "continuous")
        with contextlib.redirect_stdout(io.StringIO()):
            continuous, full_history = train(config)
            config.train_steps = 4
            config.output_dir = str(Path(directory) / "first")
            train(config)
            checkpoint_path = next(Path(config.output_dir).glob("*/checkpoint.pt"))
            config.resume = str(checkpoint_path)
            config.train_steps = 6
            config.output_dir = str(Path(directory) / "resumed")
            resumed, resumed_history = train(config)
        assert full_history == resumed_history
        for key, value in continuous.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
        final_checkpoint = torch.load(
            next(Path(config.output_dir).glob("*/checkpoint.pt")), weights_only=True,
        )
        assert final_checkpoint["completed_steps"] == 6
        assert final_checkpoint["samples_seen"] == 24
        assert final_checkpoint["optimizer_state_dict"]["state"]

        # A failed save must preserve the previous complete artifact.
        record = Path(directory) / "record.json"
        atomic_save(record, {"valid": True})
        try:
            atomic_save(record, {"invalid": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite JSON values were accepted.")
        assert '"valid": true' in record.read_text()
        assert not list(Path(directory).glob(".saving-*"))

        # Simulate a failure after one successful periodic checkpoint.
        config.resume = None
        config.train_steps = 6
        config.output_dir = str(Path(directory) / "interrupted")
        calls = 0

        def interrupted_evaluate(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Simulated evaluation interruption")
            return evaluate(*args)

        with patch("train.evaluate", side_effect=interrupted_evaluate):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    train(config)
            except RuntimeError as error:
                assert str(error) == "Simulated evaluation interruption"
            else:
                raise AssertionError("The interruption was not triggered.")
        checkpoint = torch.load(
            next(Path(config.output_dir).glob("*/checkpoint.pt")), weights_only=True,
        )
        assert checkpoint["completed_steps"] == 2 and checkpoint["status"] == "running"

    print("PASS exact CPU continuation, RMSProp state, atomic save, interruption recovery")


if __name__ == "__main__":
    main()
