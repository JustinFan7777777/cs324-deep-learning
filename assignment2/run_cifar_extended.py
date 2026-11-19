"""Prespecified matched-budget follow-up after the 5,000-update baseline.

Run on the authorized idle GPU. These are fresh 20,000-update runs, not a
claim of exact continuation from earlier checkpoints. Hyperparameters and
test protocol are frozen before either extended run starts.
"""

import json
from pathlib import Path

from cifar_experiments import ROOT, atomic_save, train
from evaluate_cifar_checkpoint import evaluate_checkpoint


def main():
    path = ROOT / 'results' / 'cifar_extended_study.json'
    if path.exists():
        raise FileExistsError('Preserve the previous study; do not overwrite its frozen protocol.')
    path.parent.mkdir(parents=True, exist_ok=True)
    study = dict(status='running', architectures=['mlp', 'cnn'], variant='baseline',
                 seed=42, steps=20000, batch_size=32, eval_freq=2000, device='cuda',
                 selection='Within each run: highest validation accuracy, then lowest validation loss.',
                 rationale='Matched longer budget to examine incomplete fitting at 5000 updates.',
                 records=[])
    atomic_save(path, study)
    for architecture in study['architectures']:
        directory = train(architecture=architecture, variant='baseline', seed=42,
                          steps=20000, batch_size=32, eval_freq=2000, device='cuda')
        # This record freezes the run identity before consulting test labels.
        study['records'].append(dict(architecture=architecture, run=str(directory.relative_to(ROOT))))
        atomic_save(path, study)
    study['status'] = 'training_complete'
    atomic_save(path, study)
    for record in study['records']:
        metrics = evaluate_checkpoint(ROOT / record['run'], device='cuda')
        record['test_accuracy'] = metrics['test_accuracy']
        record['test_loss'] = metrics['test_loss']
        atomic_save(path, study)
    study['status'] = 'complete'
    atomic_save(path, study)
    print(json.dumps(study, indent=2), flush=True)


if __name__ == '__main__':
    main()
