"""Cross-check saved numerical evidence before report writing or packaging.

This audit does not run training, install dependencies, or claim clean-machine
portability. It checks internal consistency of the current local artifacts.
"""

import ast
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def main():
    counts = dict(python_files=0, notebooks=0, comparison_pairs=0, cifar_runs=0,
                  completed_rnn_runs=0, incomplete_studies=0)
    for path in ROOT.rglob('*.py'):
        if 'data' not in path.relative_to(ROOT).parts:
            ast.parse(path.read_text(), filename=str(path))
            counts['python_files'] += 1
    for path in ROOT.rglob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and cell.source.strip():
                assert cell.execution_count is not None, f'Unexecuted cell: {path}'
            assert not any(o.get('output_type') == 'error' for o in cell.get('outputs', [])), path
        counts['notebooks'] += 1
    for path in ROOT.glob('Part 1/results/comparison/*/summary.json'):
        study = json.loads(path.read_text())
        assert study['status'] == 'complete'
        assert len(study['records']) == 2 * len(study['seeds']) * len(study['batch_sizes'])
        for row in study['records']:
            run = json.loads((path.parent / row['run'] / 'history.json').read_text())
            assert run['numpy']['epochs'] == run['pytorch']['epochs']
            assert run['numpy']['updates'] == run['pytorch']['updates']
            assert run['numpy']['epochs'][-1] == study['epochs']
            assert run['parameter_max_abs'] < 1e-8 and not run['clipping_active']
            for split in ['train', 'test']:
                np.testing.assert_allclose(run['numpy'][f'{split}_loss'], run['pytorch'][f'{split}_loss'], atol=1e-8, rtol=0)
                # The original PyTorch logger averages float32 correctness,
                # whereas NumPy records float64. Compare discrete correct
                # counts, not the binary representation of rounded fractions.
                count = 800 if split == 'train' else 200
                numpy_counts = np.rint(count*np.array(run['numpy'][f'{split}_accuracy'])).astype(int)
                torch_counts = np.rint(count*np.array(run['pytorch'][f'{split}_accuracy'])).astype(int)
                np.testing.assert_array_equal(numpy_counts, torch_counts)
            counts['comparison_pairs'] += 1
    selection = json.loads((ROOT / 'results/cifar_mlp_selection.json').read_text())
    for relative, expected in selection['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    split_reference = None
    cifar_entries = list(selection['candidates'])
    extra_directories = [p.parent for p in ROOT.glob('Part 2/results/cifar_validation/*/history.json')]
    extended_path = ROOT / 'results/cifar_extended_study.json'
    if extended_path.exists():
        extended = json.loads(extended_path.read_text())
        assert extended['status'] == 'complete'
        extra_directories.extend(ROOT / row['run'] for row in extended['records'])
    seen_directories = {row['run'] for row in cifar_entries}
    for directory in extra_directories:
        relative = str(directory.relative_to(ROOT))
        if relative in seen_directories:
            continue
        seen_directories.add(relative)
        run = json.loads((directory / 'history.json').read_text())
        cifar_entries.append(dict(run=relative, selected_step=run['best_step']))
    for row in cifar_entries:
        directory = ROOT / row['run']
        run = json.loads((directory / 'history.json').read_text())
        config = run['config']
        assert run['status'] == 'complete' and run['history'][-1]['step'] == config['steps']
        assert config['split_file_sha256'] == hashlib.sha256((directory / 'split_indices.npz').read_bytes()).hexdigest()
        with np.load(directory / 'split_indices.npz') as data:
            split = (data['train'], data['validation'])
            assert len(split[0]) == 45000 and len(split[1]) == 5000
            assert len(np.intersect1d(*split)) == 0
            if split_reference is None:
                split_reference = tuple(x.copy() for x in split)
            else:
                for actual, expected in zip(split, split_reference):
                    np.testing.assert_array_equal(actual, expected)
        best = max(run['history'], key=lambda x: (x['val_accuracy'], -x['val_loss']))
        assert best['step'] == run['best_step'] == row['selected_step']
        saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == best['step'] and saved['val_accuracy'] == best['val_accuracy']
        if (directory / 'test_metrics.json').exists():
            metric = json.loads((directory / 'test_metrics.json').read_text())
            assert metric['checkpoint_step'] == best['step']
            assert metric['checkpoint_sha256'] == hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest()
            with np.load(directory / 'test_predictions.npz') as predictions:
                assert float(np.mean(predictions['targets'] == predictions['predictions'])) == metric['test_accuracy']
                np.testing.assert_array_equal(predictions['predictions'], predictions['probabilities'].argmax(1))
                confusion = np.bincount(10*predictions['targets']+predictions['predictions'], minlength=100).reshape(10,10)
                np.testing.assert_array_equal(confusion, metric['confusion_matrix'])
        counts['cifar_runs'] += 1
    selected = ROOT / selection['selected_run']
    metrics = json.loads((selected / 'test_metrics.json').read_text())
    assert metrics['checkpoint_sha256'] == hashlib.sha256((selected / 'best.pt').read_bytes()).hexdigest()
    with np.load(selected / 'test_predictions.npz') as data:
        assert len(data['targets']) == metrics['test_samples'] == 10000
        np.testing.assert_array_equal(data['predictions'], data['probabilities'].argmax(1))
        accuracy = float(np.mean(data['targets'] == data['predictions']))
        assert accuracy == metrics['test_accuracy']
        confusion = np.bincount(10 * data['targets'] + data['predictions'], minlength=100).reshape(10, 10)
        np.testing.assert_array_equal(confusion, metrics['confusion_matrix'])
    rnn_summaries = [*ROOT.glob('Part 3/results/rnn_length_study/*/summary.json'),
                     *ROOT.glob('Part 3/results/rnn_even_lengths/*/summary.json')]
    for path in rnn_summaries:
        study = json.loads(path.read_text())
        counts['incomplete_studies'] += study['status'] != 'complete'
        if study['status'] == 'complete':
            assert len(study['records']) == len(study['planned_lengths']) * len(study['seeds'])
        for row in study['records']:
            saved = json.loads((path.parent / row['history_path']).read_text())
            history = saved['history']
            assert saved['status'] == 'complete'
            assert history['steps'][-1] == study['train_steps'] == saved['completed_steps']
            assert history['eval_accuracy'][-1] == row['final_eval_accuracy']
            assert history['eval_loss'][-1] == row['final_eval_loss']
            checkpoint = torch.load((path.parent / row['history_path']).parent / 'checkpoint.pt',
                                    map_location='cpu', weights_only=True)
            assert checkpoint['completed_steps'] == saved['completed_steps']
            counts['completed_rnn_runs'] += 1
    print('PASS: source syntax, executed notebooks, paired trajectories, split isolation,')
    print('selection/checkpoint identity, test predictions and RNN summary consistency.')
    print(json.dumps(counts, indent=2))
    print('Scope: local evidence consistency; full CNN/report/archive readiness is a separate check.')
    return counts


if __name__ == '__main__':
    main()
