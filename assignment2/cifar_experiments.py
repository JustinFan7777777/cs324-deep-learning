"""Shared validation-only CIFAR10 training protocol for Parts I and II.

Train/validation indices, preprocessing, optimizer updates and initialization
seeds are controlled. Official test images are loaded only by the separate
test command, after the user has selected a run using validation results.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
for part in ['Part 1', 'Part 2']:
    sys.path.insert(0, str(ROOT / part))
from cifar_mlp import CIFARMLP
from cnn_model import CNN
from cnn_train import atomic_save, evaluate

# These are prespecified hypotheses, not results of searching the test set.
VARIANTS = {'baseline': (0.0, 0.0), 'dropout': (0.3, 0.0),
            'weight_decay': (0.0, 1e-4), 'combined': (0.3, 1e-4)}


def image_transform():
    """Use the same fixed normalization for both architectures; no augmentation."""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


def make_splits(data_dir=ROOT / 'data'):
    """Stratify the official training set once, independently of model seeds."""
    data = datasets.CIFAR10(data_dir, train=True, download=False, transform=image_transform())
    train_ids, val_ids = train_test_split(np.arange(len(data)), test_size=5000,
                                        stratify=data.targets, random_state=42)
    assert len(train_ids) == 45000 and len(val_ids) == 5000
    assert not np.intersect1d(train_ids, val_ids).size
    return data, train_ids, val_ids


def loader(data, batch_size, seed, shuffle=False):
    # A private generator prevents evaluation from consuming the dropout RNG.
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def build_model(architecture, variant, seed, device):
    if architecture == 'cnn' and variant != 'baseline':
        raise ValueError('The required CNN uses its unchanged lecture architecture.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN() if architecture == 'cnn' else CIFARMLP(VARIANTS[variant][0])
    return model.to(device)


def synchronize(device):
    """Include asynchronous accelerator work in wall-clock measurements."""
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def check_device(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this execution environment.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this execution environment.')


def benchmark(architecture, batch_size=32, device='cpu', measured_steps=20):
    """Measure real batches; extrapolations are estimates, not completed runs."""
    check_device(device)
    if batch_size <= 0 or measured_steps <= 0:
        raise ValueError('Batch size and measured steps must be positive.')
    torch.set_num_threads(1)
    data, train_ids, val_ids = make_splits()
    train_loader = loader(Subset(data, train_ids), batch_size, 42, True)
    model = build_model(architecture, 'baseline', 42, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = iter(train_loader)
    times = []
    for step in range(measured_steps + 3):
        synchronize(device)
        start = time.perf_counter()
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        synchronize(device)
        if step >= 3:  # exclude lazy optimizer allocation and initial warmup
            times.append(time.perf_counter() - start)
    eval_count = min(512, len(val_ids))
    eval_loader = loader(Subset(data, val_ids[:eval_count]), batch_size, 43)
    synchronize(device)
    start = time.perf_counter()
    evaluate(model, eval_loader, nn.CrossEntropyLoss())
    synchronize(device)
    eval_seconds_per_sample = (time.perf_counter() - start) / eval_count
    result = dict(architecture=architecture, device=device, batch_size=batch_size,
                  measured_steps=measured_steps, warmup_steps=3,
                  mean_step_seconds=float(np.mean(times)), median_step_seconds=float(np.median(times)),
                  eval_samples=eval_count, eval_seconds_per_sample=eval_seconds_per_sample,
                  estimated_5000_update_seconds=float(np.mean(times) * 5000),
                  estimated_full_train_val_eval_seconds=eval_seconds_per_sample * 50000,
                  params=sum(p.numel() for p in model.parameters()),
                  torch_version=str(torch.__version__),
                  note='Short CPU/accelerator timing; excludes checkpoint I/O and sustained thermal slowdown.')
    root = ROOT / 'results' / 'benchmarks'
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'{architecture}-', dir=root))
    atomic_save(directory / 'benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def fit(model, train_loader, train_eval_loader, val_loader, config, directory):
    """Optimize on train, select on validation, and keep test entirely unseen.

    All history metrics are post-update, whole-split, sample-weighted means.
    Select maximum validation accuracy, breaking ties by lower validation loss.
    Checkpoints preserve optimizer state but do not implement exact resumption.
    """
    directory = Path(directory)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    history, best_key, best_step, samples_seen = [], (-1.0, -float('inf')), None, 0
    iterator = iter(train_loader)
    synchronize(str(device))
    started = time.perf_counter()
    for step in range(1, config['steps'] + 1):
        model.train()
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at step {step}.')
        loss.backward()
        optimizer.step()
        samples_seen += len(labels)
        if step % config['eval_freq'] == 0 or step == config['steps']:
            train_metrics = evaluate(model, train_eval_loader, criterion)
            val_metrics = evaluate(model, val_loader, criterion)
            train_loss, train_accuracy = train_metrics['loss'], train_metrics['accuracy']
            val_loss, val_accuracy = val_metrics['loss'], val_metrics['accuracy']
            synchronize(str(device))
            record = dict(step=step, samples_seen=samples_seen,
                          epoch_equivalent=samples_seen / len(train_loader.dataset),
                          train_loss=train_loss, train_accuracy=train_accuracy,
                          val_loss=val_loss, val_accuracy=val_accuracy,
                          elapsed_seconds=time.perf_counter() - started)
            history.append(record)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            key = (val_accuracy, -val_loss)
            if key > best_key:
                best_key, best_step = key, step
                atomic_save(directory / 'best.pt', dict(config=config, step=step,
                            model_state_dict=state, val_accuracy=val_accuracy, val_loss=val_loss), binary=True)
            atomic_save(directory / 'checkpoint.pt', dict(config=config, step=step,
                        model_state_dict=state, optimizer_state_dict=optimizer.state_dict(),
                        history=history, best_step=best_step), binary=True)
            atomic_save(directory / 'history.json', dict(config=config, history=history,
                        best_step=best_step, status='complete' if step == config['steps'] else 'running'))
            print(f'{config["architecture"]}/{config["variant"]} step={step}: '
                  f'train={train_accuracy:.3f}, val={val_accuracy:.3f}', flush=True)
    return history


def train(architecture='mlp', variant='baseline', seed=42, steps=5000,
          eval_freq=500, batch_size=32, device='cpu'):
    check_device(device)
    if min(steps, eval_freq, batch_size) <= 0:
        raise ValueError('Steps, evaluation interval and batch size must be positive.')
    torch.set_num_threads(1)
    data, train_ids, val_ids = make_splits()
    train_data, val_data = Subset(data, train_ids), Subset(data, val_ids)
    model = build_model(architecture, variant, seed, device)
    root = ROOT / ('Part 1' if architecture == 'mlp' else 'Part 2') / 'results' / 'cifar_validation'
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'{variant}-seed{seed}-', dir=root))
    np.savez(directory / 'split_indices.npz', train=train_ids, validation=val_ids)
    split_hash = hashlib.sha256((directory / 'split_indices.npz').read_bytes()).hexdigest()
    config = dict(architecture=architecture, variant=variant, seed=seed, split_seed=42,
                  steps=steps, eval_freq=eval_freq, batch_size=batch_size, device=device,
                  learning_rate=0.001, weight_decay=VARIANTS[variant][1],
                  dropout=VARIANTS[variant][0], train_samples=len(train_ids), val_samples=len(val_ids),
                  params=sum(p.numel() for p in model.parameters()),
                  normalization=dict(mean=[0.5]*3, std=[0.5]*3), augmentation=False,
                  selection='highest validation accuracy, then lowest validation loss',
                  split_file_sha256=split_hash, torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__),
                  numpy_version=np.__version__, num_threads=1,
                  exact_resume_supported=False)
    atomic_save(directory / 'config.json', config)
    fit(model, loader(train_data, batch_size, seed, True),
        loader(train_data, batch_size, seed + 1), loader(val_data, batch_size, seed + 2),
        config, directory)
    print(f'Saved {directory}', flush=True)
    return directory


@torch.no_grad()
def test_selected(directory):
    """Evaluate a frozen validation-selected model; export per-example evidence."""
    directory = Path(directory)
    report = json.loads((directory / 'history.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Finish the prespecified budget before final test evaluation.')
    torch.set_num_threads(1)
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = build_model(config['architecture'], config['variant'], config['seed'], 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_data = datasets.CIFAR10(ROOT / 'data', train=False, download=False, transform=image_transform())
    probabilities, targets = [], []
    for images, labels in loader(test_data, config['batch_size'], 0):
        probabilities.append(model(images).softmax(1))
        targets.append(labels)
    probabilities, targets = torch.cat(probabilities), torch.cat(targets)
    predictions = probabilities.argmax(1)
    confusion = torch.bincount(targets * 10 + predictions, minlength=100).reshape(10, 10)
    # Save probabilities for reproducible confidence/error analysis; loss is
    # evaluated from logits separately to avoid log(softmax) underflow.
    metrics = evaluate(model, loader(test_data, config['batch_size'], 0), nn.CrossEntropyLoss())
    np.savez_compressed(directory / 'test_predictions.npz', probabilities=probabilities.numpy(),
                        targets=targets.numpy(), predictions=predictions.numpy())
    result = dict(checkpoint_step=checkpoint['step'], test_samples=len(targets),
                  test_loss=metrics['loss'], test_accuracy=metrics['accuracy'],
                  confusion_matrix=confusion.tolist(), classes=test_data.classes,
                  checkpoint_sha256=hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest())
    atomic_save(directory / 'test_metrics.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['train', 'benchmark']:
        p = sub.add_parser(command)
        p.add_argument('--architecture', choices=['mlp', 'cnn'], default='mlp')
        p.add_argument('--batch_size', type=int, default=32)
        p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
        if command == 'train':
            p.add_argument('--variant', choices=list(VARIANTS), default='baseline')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--steps', type=int, default=5000)
            p.add_argument('--eval_freq', type=int, default=500)
        else:
            p.add_argument('--measured_steps', type=int, default=20)
    p = sub.add_parser('test')
    p.add_argument('directory', type=Path)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'train': train, 'benchmark': benchmark, 'test': test_selected}[command](**args)
