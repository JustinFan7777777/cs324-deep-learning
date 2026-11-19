"""Controlled Assignment 1 NumPy versus Assignment 2 PyTorch experiments.

Both implementations use float64, the same initial weights, SGD, batch order,
and evaluation epochs. No tuning is performed against the displayed test set.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

from numpy_reference.mlp_numpy import MLP as NumpyMLP
from numpy_reference.modules import Linear, CrossEntropy
from pytorch_mlp import MLP
from pytorch_train_mlp import train_model


def paired_models(seed):
    """Map NumPy's (in,out) matrices to PyTorch's (out,in) storage."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    reference = NumpyMLP(2, [20], 2)
    model = MLP(2, [20], 2).double()
    pairs = list(zip((x for x in reference.layers if isinstance(x, Linear)),
                     (x for x in model.modules() if isinstance(x, nn.Linear))))
    with torch.no_grad():
        for numpy_layer, torch_layer in pairs:
            torch_layer.weight.copy_(torch.from_numpy(numpy_layer.params['weight'].T))
            torch_layer.bias.copy_(torch.from_numpy(numpy_layer.params['bias']))
    return reference, model, pairs


def verify_parity(seed=42):
    """Check probabilities, mean CE, every parameter gradient and SGD update."""
    reference, model, pairs = paired_models(seed)
    rng = np.random.default_rng(seed)
    # A non-square batch detects transposition and reduction mistakes.
    x, y = rng.normal(size=(7, 2)), rng.integers(0, 2, size=7)
    one_hot = np.eye(2)[y]
    criterion = CrossEntropy()
    probabilities = reference.forward(x)
    logits = model(torch.from_numpy(x))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y))
    reference.backward(criterion.backward(probabilities, one_hot))
    loss.backward()
    errors = {
        'probability_max_abs': float(np.max(np.abs(probabilities - logits.softmax(1).detach().numpy()))),
        'loss_abs': abs(float(loss.detach()) - float(criterion.forward(probabilities, one_hot))),
    }
    gradient_errors, update_errors = [], []
    with torch.no_grad():
        for numpy_layer, torch_layer in pairs:
            for name, parameter in [('weight', torch_layer.weight), ('bias', torch_layer.bias)]:
                grad = parameter.grad.numpy().T if name == 'weight' else parameter.grad.numpy()
                gradient_errors.append(float(np.max(np.abs(numpy_layer.grads[name] - grad))))
                numpy_layer.params[name] -= 0.01 * numpy_layer.grads[name]
                parameter.add_(parameter.grad, alpha=-0.01)
                actual = parameter.numpy().T if name == 'weight' else parameter.numpy()
                update_errors.append(float(np.max(np.abs(numpy_layer.params[name] - actual))))
    errors['gradient_max_abs'] = max(gradient_errors)
    errors['update_max_abs'] = max(update_errors)
    if max(errors.values()) > 1e-12:
        raise AssertionError(f'Numerical parity failed: {errors}')
    return errors


def run_pair(dataset, seed, batch_size, epochs, directory):
    """Train independently, with an identical deterministic permutation stream."""
    if dataset == 'moons':
        x, y = make_moons(n_samples=1000, noise=0.2, random_state=seed)
    else:
        x, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed)
    directory.mkdir()
    np.savez_compressed(directory / 'data.npz', x_train=x_train, x_test=x_test,
                        y_train=y_train, y_test=y_test)
    reference, model, pairs = paired_models(seed)
    initial = {f'{i}_{name}': value.copy() for i, (layer, _) in enumerate(pairs)
               for name, value in layer.params.items()}
    np.savez(directory / 'initial_weights.npz', **initial)
    criterion, rng = CrossEntropy(), np.random.default_rng(seed)
    effective_batch_size = batch_size or len(x_train)
    history = {key: [] for key in ['epochs', 'updates', 'train_loss', 'train_accuracy',
                                   'test_loss', 'test_accuracy']}
    minimum_probability, updates = 1.0, 0
    start_time = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), effective_batch_size):
            indices = order[start:start + effective_batch_size]
            probabilities = reference.forward(x_train[indices])
            minimum_probability = min(minimum_probability,
                                      float(probabilities[np.arange(len(indices)), y_train[indices]].min()))
            reference.backward(criterion.backward(probabilities, np.eye(2)[y_train[indices]]))
            for layer, _ in pairs:
                for name in layer.params:
                    layer.params[name] -= 0.01 * layer.grads[name]
            updates += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            history['epochs'].append(epoch + 1)
            history['updates'].append(updates)
            for split, features, labels in [('train', x_train, y_train), ('test', x_test, y_test)]:
                p = reference.forward(features)
                minimum_probability = min(minimum_probability, float(p[np.arange(len(labels)), labels].min()))
                history[f'{split}_loss'].append(float(criterion.forward(p, np.eye(2)[labels])))
                history[f'{split}_accuracy'].append(float(np.mean(p.argmax(1) == labels)))
    numpy_seconds = time.perf_counter() - start_time
    start_time = time.perf_counter()
    # Keep the reusable trainer's verbose logging out of notebook outputs.
    with contextlib.redirect_stdout(io.StringIO()):
        torch_history = train_model(model, x_train, y_train, x_test, y_test,
                                    epochs=epochs, batch_size=batch_size, seed=seed)
    torch_seconds = time.perf_counter() - start_time
    parameter_error = max(float(np.max(np.abs(layer.params[name] - (
        parameter.detach().numpy().T if name == 'weight' else parameter.detach().numpy()))))
        for layer, torch_layer in pairs
        for name, parameter in [('weight', torch_layer.weight), ('bias', torch_layer.bias)])
    loss_error = max(float(np.max(np.abs(np.array(history[key]) - torch_history[key])))
                     for key in ['train_loss', 'test_loss'])
    if not np.isfinite(loss_error) or parameter_error > 1e-8 or loss_error > 1e-8:
        raise AssertionError(f'Training trajectories diverged: {parameter_error=}, {loss_error=}')
    torch.save(model.state_dict(), directory / 'model.pt')
    np.savez(directory / 'numpy_weights.npz', **{
        f'{i}_{name}': value for i, (layer, _) in enumerate(pairs) for name, value in layer.params.items()})
    result = dict(dataset=dataset, seed=seed, batch_size=batch_size, epochs=epochs,
                  learning_rate=0.01, dtype='float64', hidden_units=[20],
                  numpy=history, pytorch=torch_history,
                  numpy_seconds=numpy_seconds, pytorch_seconds=torch_seconds,
                  parameter_max_abs=parameter_error, loss_max_abs=loss_error,
                  minimum_true_class_probability=minimum_probability,
                  clipping_active=minimum_probability < 1e-12)
    (directory / 'history.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_study(epochs=1500, seeds=(42, 43, 44), batches=(0, 32), output_dir=None):
    if epochs <= 0 or not seeds or not batches or any(b < 0 for b in batches):
        raise ValueError('Require positive epochs, nonempty seeds/batches and nonnegative batch sizes.')
    if len(set(seeds)) != len(seeds) or len(set(batches)) != len(batches):
        raise ValueError('Seeds and batch sizes must be distinct.')
    torch.set_num_threads(1)
    root = Path(output_dir or Path(__file__).parent / 'results' / 'comparison')
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='study-', dir=root))
    summary = dict(status='running', epochs=epochs, seeds=list(seeds), batch_sizes=list(batches),
                   parity=verify_parity(), torch_version=str(torch.__version__),
                   numpy_version=np.__version__, records=[])
    for dataset in ['moons', 'circles']:
        for batch in batches:
            for seed in seeds:
                name = f'{dataset}-batch{batch}-seed{seed}'
                result = run_pair(dataset, seed, batch, epochs, directory / name)
                summary['records'].append(dict(run=name, dataset=dataset, seed=seed, batch_size=batch,
                    accuracy=result['pytorch']['test_accuracy'][-1],
                    parameter_max_abs=result['parameter_max_abs']))
                (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
                print(f'{name}: test={summary["records"][-1]["accuracy"]:.3f}, '
                      f'parameter difference={result["parameter_max_abs"]:.2e}', flush=True)
    summary['status'] = 'complete'
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Saved {directory}', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--batches', type=int, nargs='+', default=[0, 32])
    parser.add_argument('--output_dir', type=Path)
    run_study(**vars(parser.parse_args()))
