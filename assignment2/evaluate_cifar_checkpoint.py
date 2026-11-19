"""Evaluate a frozen CIFAR checkpoint on CPU or CUDA in a single forward pass."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from cifar_experiments import ROOT, atomic_save, build_model, check_device, image_transform, loader


@torch.inference_mode()
def evaluate_checkpoint(directory, device='cpu'):
    directory = Path(directory)
    check_device(device)
    torch.set_num_threads(1)
    history = json.loads((directory / 'history.json').read_text())
    if history['status'] != 'complete':
        raise ValueError('Complete the prespecified training budget before testing.')
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = build_model(config['architecture'], config['variant'], config['seed'], device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = CIFAR10(ROOT / 'data', train=False, download=False, transform=image_transform())
    all_probabilities, all_targets = [], []
    loss_sum = torch.zeros((), device=device)
    for images, targets in loader(dataset, config['batch_size'], 0):
        logits = model(images.to(device))
        loss_sum += torch.nn.functional.cross_entropy(logits, targets.to(device), reduction='sum')
        all_probabilities.append(logits.softmax(1).cpu())
        all_targets.append(targets)
    probabilities, targets = torch.cat(all_probabilities), torch.cat(all_targets)
    predictions = probabilities.argmax(1)
    confusion = torch.bincount(10 * targets + predictions, minlength=100).reshape(10, 10)
    np.savez_compressed(directory / 'test_predictions.npz', probabilities=probabilities.numpy(),
                        targets=targets.numpy(), predictions=predictions.numpy())
    result = dict(checkpoint_step=checkpoint['step'], test_samples=len(targets),
                  test_loss=loss_sum.item()/len(targets),
                  test_accuracy=(predictions == targets).sum().item()/len(targets),
                  confusion_matrix=confusion.tolist(), classes=dataset.classes, device=device,
                  checkpoint_sha256=hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest())
    atomic_save(directory / 'test_metrics.json', result)
    print(f'{config["architecture"]}: test accuracy={result["test_accuracy"]:.4f}, '
          f'loss={result["test_loss"]:.4f}, selected update={checkpoint["step"]}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    evaluate_checkpoint(**vars(parser.parse_args()))
