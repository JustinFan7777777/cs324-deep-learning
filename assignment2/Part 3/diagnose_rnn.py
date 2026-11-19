"""Inspect time-specific hidden-state gradients of saved RNN checkpoints.

This is a fixed-checkpoint diagnostic, not a proof that vanishing gradients
caused a training failure. No parameters are updated and no checkpoint is
selected using these diagnostic samples.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from train import atomic_save, make_eval_loader
from vanilla_rnn import VanillaRNN


def diagnose(study_dir):
    torch.set_num_threads(1)
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / 'summary.json').read_text())
    records = []
    for entry in summary['records']:
        run_dir = (study_dir / entry['history_path']).parent
        checkpoint_path = run_dir / 'checkpoint.pt'
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config = saved['config']
        length = config['input_length']
        model = VanillaRNN(length, config['input_dim'], config['num_hidden'],
                           config['num_classes'], config['batch_size'])
        model.load_state_dict(saved['model_state_dict'])
        model.eval()
        # A fresh, fixed diagnostic batch uses a distinct seed. Its purpose is
        # to compare gradients across time, not to replace the study metric.
        inputs, targets = next(iter(make_eval_loader(length, 256, 256, 20261119)))
        oracle_accuracy = float((inputs[:, 0].long() == targets).float().mean())
        assert oracle_accuracy == 1.0  # A palindrome's final digit equals its first.
        hidden = inputs.new_zeros(len(inputs), config['num_hidden'])
        states = []
        for t in range(length):
            hidden = torch.tanh(model.input_to_hidden(inputs[:, t:t+1]) +
                                model.hidden_to_hidden(hidden))
            hidden.retain_grad()
            states.append(hidden)
        logits = model.hidden_to_output(hidden)
        with torch.no_grad():
            torch.testing.assert_close(logits, model(inputs))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        # Mean reduction introduces 1/B. Multiply by B to recover each
        # example's loss gradient, then average its L2 norm across examples.
        norms = [float((h.grad * len(inputs)).norm(dim=1).mean()) for h in states]
        records.append(dict(total_length=length+1, seed=config['seed'],
                            checkpoint_step=saved['completed_steps'],
                            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                            oracle_accuracy=oracle_accuracy,
                            diagnostic_accuracy=float((logits.argmax(1)==targets).float().mean()),
                            diagnostic_loss=float(loss.detach()),
                            mean_per_example_hidden_gradient_norm=norms,
                            hidden_saturation_fraction=[float((h.detach().abs()>0.99).float().mean()) for h in states]))
    result = dict(diagnostic_seed=20261119, diagnostic_samples=256, records=records,
                  gradient_definition='Mean over examples of ||d individual CE / d h_t||_2',
                  limitation='Checkpoint-local sensitivity; no causal attribution of training failures.')
    atomic_save(study_dir / 'diagnostics.json', result)
    seeds = sorted({row['seed'] for row in records})
    lengths = sorted({row['total_length'] for row in records})
    colors = {length: plt.get_cmap('tab10')(i) for i, length in enumerate(lengths)}
    fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 3.3*len(seeds)),
                             squeeze=False, constrained_layout=True)
    for row in records:
        t = list(range(1, row['total_length']))
        label = f'T={row["total_length"]}'
        pair = axes[seeds.index(row['seed'])]
        color = colors[row['total_length']]
        pair[0].semilogy(t, row['mean_per_example_hidden_gradient_norm'], marker='.', label=label, color=color)
        pair[1].plot(t, row['hidden_saturation_fraction'], marker='.', label=label, color=color)
    for seed, pair in zip(seeds, axes):
        pair[0].set(xlabel='Observed time step t', ylabel='Mean per-example gradient norm',
                    title=f'Final-checkpoint sensitivity: seed {seed}')
        pair[1].set(xlabel='Observed time step t', ylabel='Fraction |h| > 0.99',
                    title=f'Tanh saturation: seed {seed}')
        pair[0].legend(fontsize=7)
        for ax in pair:
            ax.grid(alpha=0.2)
    fig.savefig(study_dir / 'temporal_diagnostics.png', dpi=160)
    plt.close(fig)
    print(f'Saved diagnostics for {len(records)} checkpoints in {study_dir}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_dir', type=Path)
    diagnose(parser.parse_args().study_dir)
