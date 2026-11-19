"""Small deterministic checks for the shared CIFAR10 experiment protocol."""

import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from cifar_experiments import (VARIANTS, build_model, evaluate, fit, loader,
                               make_splits)


def main():
    torch.set_num_threads(1)
    data, train_ids, val_ids = make_splits()
    labels = np.array(data.targets)
    assert np.array_equal(np.bincount(labels[train_ids]), [4500] * 10)
    assert np.array_equal(np.bincount(labels[val_ids]), [500] * 10)
    assert len(np.unique(np.concatenate([train_ids, val_ids]))) == 50000

    # Dropout consumes no randomness during initialization. Every ablation
    # must start with identical Linear parameters at a given seed.
    initial = build_model('mlp', 'baseline', 42, 'cpu').state_dict()
    for variant in VARIANTS:
        model = build_model('mlp', variant, 42, 'cpu')
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
        assert sum(p.numel() for p in model.parameters()) == 1707274

    generator = torch.Generator().manual_seed(91)
    x = torch.randn(7, 3, 32, 32, generator=generator)
    y = torch.arange(7) % 10
    dataset = TensorDataset(x, y)
    model = build_model('mlp', 'combined', 42, 'cpu')
    model.train()
    rng_state = torch.get_rng_state().clone()
    measured = evaluate(model, loader(dataset, 3, 9), nn.CrossEntropyLoss())
    assert model.training and torch.equal(rng_state, torch.get_rng_state())
    model.eval()
    with torch.no_grad():
        expected = nn.functional.cross_entropy(model(x), y).item()
        assert abs(measured['loss'] - expected) < 1e-6
        assert torch.equal(model(x), model(x))

    with tempfile.TemporaryDirectory(prefix='cifar-protocol-') as temporary:
        directory = Path(temporary)
        config = dict(architecture='mlp', variant='combined', steps=4, eval_freq=2,
                      learning_rate=0.001, weight_decay=1e-4)
        history = fit(model, loader(dataset, 3, 42, True), loader(dataset, 3, 43),
                      loader(dataset, 3, 44), config, directory)
        assert [row['samples_seen'] for row in history] == [6, 10]
        assert [row['step'] for row in history] == [2, 4]
        payload = json.loads((directory / 'history.json').read_text())
        assert payload['status'] == 'complete'
        best = max(history, key=lambda r: (r['val_accuracy'], -r['val_loss']))
        assert payload['best_step'] == best['step']
        saved = torch.load(directory / 'best.pt', weights_only=True)
        reloaded = build_model('mlp', 'combined', 42, 'cpu')
        reloaded.load_state_dict(saved['model_state_dict'])
        actual = evaluate(reloaded, loader(dataset, 3, 44), nn.CrossEntropyLoss())
        assert abs(actual['accuracy'] - best['val_accuracy']) < 1e-12
        assert abs(actual['loss'] - best['val_loss']) < 1e-6
        assert not (directory / 'test_metrics.json').exists()
    # Exercise the same runner with the required CNN, including BatchNorm.
    cnn = build_model('cnn', 'baseline', 42, 'cpu')
    tiny_data = TensorDataset(x[:5], y[:5])
    with tempfile.TemporaryDirectory(prefix='cnn-protocol-') as temporary:
        config = dict(architecture='cnn', variant='baseline', steps=2, eval_freq=2,
                      learning_rate=0.001, weight_decay=0.0)
        fit(cnn, loader(tiny_data, 3, 42, True), loader(tiny_data, 3, 43),
            loader(tiny_data, 3, 44), config, Path(temporary))
        buffers = {name: value.clone() for name, value in cnn.named_buffers()}
        evaluate(cnn, loader(tiny_data, 3, 44), nn.CrossEntropyLoss())
        assert all(torch.equal(buffers[name], value) for name, value in cnn.named_buffers())
        assert cnn.training
    print('PASS: stratified/disjoint splits, matched initialization, evaluation RNG/mode,')
    print('sample-weighted metrics, partial batches, validation selection, checkpoint reload, CNN/BN.')


if __name__ == '__main__':
    main()
