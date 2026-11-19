"""Estimate the default length-study cost without running the full experiment."""

import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from train import atomic_save, evaluate, make_eval_loader
from vanilla_rnn import VanillaRNN


def benchmark():
    torch.set_num_threads(1)
    records = []
    for total_length in [5, 9, 13, 17, 21]:
        np.random.seed(42)
        torch.manual_seed(42)
        model = VanillaRNN(total_length - 1, 1, 128, 10, 128)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        batches = iter(DataLoader(PalindromeDataset(total_length), batch_size=128,
                                  num_workers=0, generator=torch.Generator().manual_seed(42)))
        durations = []
        for step in range(110):
            start = time.perf_counter()
            x, y = next(batches)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            if step >= 10:
                durations.append(time.perf_counter() - start)
        eval_loader = make_eval_loader(total_length - 1, 1000, 128, 10042)
        start = time.perf_counter()
        evaluate(model, eval_loader)
        eval_seconds = time.perf_counter() - start
        records.append(dict(total_length=total_length, mean_step_seconds=float(np.mean(durations)),
                            eval_1000_seconds=eval_seconds,
                            estimated_10000_steps_with_20_evals_seconds=float(np.mean(durations)*10000+20*eval_seconds)))
    result = dict(device='cpu', num_threads=1, warmup_steps=10, measured_steps_per_length=100,
                  records=records, estimated_three_seed_seconds=3*sum(
                      row['estimated_10000_steps_with_20_evals_seconds'] for row in records),
                  note='Timing only; excludes checkpoint I/O and sustained thermal slowdown.')
    root = Path(__file__).resolve().parent.parent / 'results' / 'benchmarks'
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='rnn-', dir=root))
    atomic_save(directory / 'benchmark.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    benchmark()
