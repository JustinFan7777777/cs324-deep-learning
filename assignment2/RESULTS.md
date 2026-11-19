# Recorded experiment results

These results come from the completed coursework study. Saved notebook outputs
show the corresponding tables and curves. This public code release excludes
raw result folders and checkpoints; the full evidence remains in the local
submission archive. Values below are observed results, not expected outcomes.

## Matched NumPy/PyTorch MLP

Both implementations use a 2–20–2 MLP, float64, SGD with learning rate 0.01,
matched weights and batch order, and 1,500 epochs. Each dataset has a stratified
800/200 train/test split. Seeds 42, 43 and 44 vary data and initialization.

| Data | Batch size | Mean test accuracy | Sample SD |
|---|---:|---:|---:|
| Moons | Full | 84.50% | 3.12 percentage points |
| Moons | 32 | 89.17% | 5.51 percentage points |
| Circles | Full | 70.67% | 6.29 percentage points |
| Circles | 32 | 99.33% | 0.29 percentage points |

All 12 paired settings agree in final correct counts. The largest final
parameter difference is 2.28e-15; the largest loss-trajectory difference is
3.33e-16. Full batches make 1,500 updates; batch size 32 makes 37,500 updates.
The NumPy loss's extreme-probability clipping limitation was not activated.

## CIFAR10

A fixed stratified split uses 45,000 training and 5,000 validation images.
Selection maximizes validation accuracy, with validation loss as a tie-breaker.
All listed scores are for selected checkpoints, not necessarily final updates.

| Model | Training budget | Selected update | Validation accuracy | Test accuracy | Test CE |
|---|---:|---:|---:|---:|---:|
| MLP | 5,000 | 4,500 | 50.56% | 50.17% | 1.4630 |
| CNN | 5,000 | 5,000 | 79.64% | 79.24% | 0.6289 |
| MLP | 20,000 | 20,000 | 52.62% | 52.30% | 1.9808 |
| CNN | 20,000 | 16,000 | 83.68% | 82.83% | 0.6970 |

The 5,000-update MLP study compared baseline, dropout (p=0.3), weight decay
(1e-4), and their combination. Their best validation accuracies were 50.56%,
47.02%, 49.38%, and 45.72%. Only the validation-selected baseline was tested.
This does not establish that regularization is generally harmful.

The extended pair shares seed 42, preprocessing, batch size 32, Adam lr=0.001,
and the training budget. The CNN uses the specified reduced VGG with 9,231,114
parameters; the MLP has 1,707,274. Their 30.53-point test accuracy difference
is an architecture comparison, not an isolated causal effect of convolution.
Higher accuracy coexists with higher test cross-entropy across stages, consistent
with stronger penalties from confident errors. Each configuration has one seed.

## Explicit vanilla RNN

Each formal run uses hidden width 128, scalar digit inputs, RMSProp lr=0.001,
batch size 128, clipping norm 10, 10,000 updates, and 1,000 fixed evaluation
sequences. T is total palindrome length; the model observes T−1 digits.

| T | Seed 42 | Seed 43 | Seed 44 | Mean |
|---:|---:|---:|---:|---:|
| 5 | 100.0% | 100.0% | 100.0% | 100.00% |
| 9 | 100.0% | 100.0% | 100.0% | 100.00% |
| 13 | 100.0% | 39.7% | 88.4% | 76.03% |
| 17 | 20.8% | 52.7% | 20.7% | 31.40% |
| 21 | 18.7% | 18.6% | 100.0% | 45.77% |

Final-budget endpoints are retained, including failures. The nonmonotonic
results support declining reliability at greater lengths under this protocol,
not a universal memory cutoff. Three seeds have limited precision and vary
initialization, training samples and evaluation samples. Supplementary even
lengths T=6 and T=10 reached 100% and 90.9% with seed 42.

Checkpoint-local gradient diagnostics show early-state attenuation at some
failed formal checkpoints, but a failed pilot provides a counterexample.
They do not prove the cause of a training collapse. Short palindromes have
small finite support; perfect evaluation does not imply unseen-length transfer.

## Recorded environments and verification

| Stage | Python | PyTorch | torchvision | NumPy |
|---|---|---|---|---|
| Local CPU | 3.11.15 | 2.5.1 | 0.20.1 | 1.26.4 |
| Initial CNN, A800 | 3.12.16 | 2.10.0+cu128 | 0.25.0+cu128 | 2.2.6 |
| Extended pair, A800 | 3.12.3 | 2.8.0+cu128 | 0.23.0+cu128 | 2.3.2 |

Before submission, five notebooks reran from an independent archive extraction.
The evidence audit covered 12 toy pairs, seven CIFAR runs and 22 RNN runs
(15 formal, two even-length supplements, five pilots). Split isolation,
checkpoint selection, prediction consistency, finite-difference RNN gradients,
atomic recovery and exact same-environment CPU continuation passed.
This reused the tested environment and documented CIFAR10 data; it was not a
fresh dependency installation or a repeat of full training.
