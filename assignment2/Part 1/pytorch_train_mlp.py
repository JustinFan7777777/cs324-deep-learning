from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn

from pytorch_mlp import MLP

# Default constants (match in assignemnt 1)
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

BATCH_SIZE_DEFAULT = 0 # Batch size for training. 0 means full batch
SEED_DEFAULT = 42 # random seed for reproducibility
# seed is used for reproducibility,
# ensuring that the random processes,
# like weight initialization and data shuffling,
# produce the same results across different runs.
# without setting a seed, the results may vary each time the code is run.

def make_moons_data(seed: int = SEED_DEFAULT):
    """
    Generates a synthetic dataset using the make_moons function from sklearn.
    Args:
        seed: random seed for reproducibility
    Returns:
        x_train: shape (800, 2) -> 800 samples, each with 2 features
        x_test: shape (200, 2) -> 200 samples, each with 2 features
        y_train: shape (800,) -> 800 samples, each with a binary label (0 or 1)
        y_test: shape (200,) -> 200 samples, each with a binary label (0 or 1)
    """

    # make_moons generates a 2D binary classification dataset,
    # with two interleaving half circles (moons).
    # why the name "moons"? Because the shape of the dateset resembles two crescent moons.
    features, labels = make_moons(
        n_samples=1000, # total number of samples to generate
        noise=0.2, # standard deviation of Gaussian noise added to the data
        random_state=seed # ensures reproducibility of the dataset
    )

    # split the dataset into training and testing sets,
    # with 80% of the data used for training and 20% for testing.
    return train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

def accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.
    Args:
        predictions: Logits or probabilities with shape (B, C).
        targets: Class indices with shape (B,) or one-hot labels
            with shape (B, C).
    Returns:
        Fraction of correct predictions in [0, 1].
    """

    predicted_classes = torch.argmax(predictions, dim=1)
    # torch.argmax returns the indices of the maximum values along a specified dimension,
    # in this case, dim=1, which corresponds to the class dimension.

    if targets.ndim == 2:
        targets = targets.argmax(dim=1)

    correct = predicted_classes.eq(targets)
    # correct is a boolean tensor indicating which predictions match the true labels,
    # for example, if predicted_classes = [0, 1, 1] and targets = [0, 0, 1],
    # then we say that -> correct = [True, False, True],
    # which means that the first and last predictions are correct, while the second prediction is incorrect.

    return correct.float().mean().item()
    # correct.float() converts the boolean tensor to a float tensor, where True becomes 1.0 and False becomes 0.0.
    # .mean() computes the average of the float tensor, which gives the proportion of correct predictions.
    # .item() extracts the scalar value from the tensor, returning it as a standard Python float.

@torch.no_grad()
def evaluate(
    model: MLP,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, float]:

    was_training = model.training
    # model.training is a boolean attribute of the model,
    # that indicates whether the model is in training mode or evaluation mode.
    
    model.eval()
    # model.eval() sets the model to evaluation mode,
    # which affects certain layers like dropout and batch normalization,
    # ensuring that they behave appropriately during evaluation.

    logits = model(features)
    # logits are the raw, unnormalized scores output by the model,
    # which are then passed to the loss function and used for computing accuracy.

    metrics = {
        "loss": criterion(logits, targets).item(),
        "accuracy": accuracy(logits, targets),
    }

    model.train(was_training)
    # model.train(was_training) restores the model to its original training state,
    # ensuring that if it was in training mode before evaluation, it returns to that state.

    return metrics
    # The function returns a dictionary containing the computed loss and accuracy metrics,
    # for the given features and targets, using the provided model and loss criterion.

def train_model(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    epochs: int = MAX_EPOCHS_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    eval_freq: int = EVAL_FREQ_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict[str, list]:
    """
    Trains the MLP model on the provided training data and evaluates it on the test data.
    Args:
        model: an instance of the MLP class, representing the neural network to be trained.
        x_train: 2D numpy array of shape (num_train_samples, num_features), containing the training features.
        y_train: 1D numpy array of shape (num_train_samples,), containing the training labels.
        x_test: 2D numpy array of shape (num_test_samples, num_features), containing the test features.
        y_test: 1D numpy array of shape (num_test_samples,), containing the test labels.
        learning_rate: float, the learning rate for the optimizer.
        epochs: int, the number of complete passes through the training dataset.
        batch_size: int, the number of samples per batch for training. If 0, full batch is used.
        eval_freq: int, frequency (in epochs) at which to evaluate the model on the test set.
        seed: int, random seed for reproducibility.
    Returns:
        metrics: a dictionary containing lists of training and test metrics (loss and accuracy) recorded at each evaluation step.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive.")

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Training and test sets must be non-empty.")

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Each example must have one target label.")

    parameter = next(model.parameters())
    # next retrieves the first parameter of the model, which is used to determine the device (CPU or GPU) on which the model is located.
    # this is important for ensuring that the input data is moved to the same device as the model before training.

    device = parameter.device
    # device is a torch.device object that indicates whether the model is on the CPU or GPU

    dtype = parameter.dtype
    # dtype is a torch.dtype object that indicates the data type of the model's parameters (e.g., torch.float32, torch.float64)

    # Reuse existing storage when dtype and device permit it.
    train_features = torch.as_tensor(
        x_train,
        dtype=dtype,
        device=device,
    )

    test_features = torch.as_tensor(
        x_test,
        dtype=dtype,
        device=device,
    )

    train_targets = torch.as_tensor(
        y_train,
        dtype=torch.long,
        device=device,
    )

    test_targets = torch.as_tensor(
        y_test,
        dtype=torch.long,
        device=device,
    )

    effective_batch_size = (
        len(train_features) if batch_size == 0 else batch_size
    )

    criterion = nn.CrossEntropyLoss(reduction="mean")
    # mean reduction averages the loss over the current batch.

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.0,
        weight_decay=0.0,
    )
    # with zero momentum and weight decay,
    # the optimizer performs standard stochastic gradient descent (SGD),
    # without any additional regularization or acceleration.

    rng = np.random.default_rng(seed)
    # np.random.default_rng(seed) creates a new random number generator instance with the specified seed,
    # ensuring that the random processes, like shuffling the training data, are reproducible.

    history = {
        "epochs": [],
        "updates": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    # history is a dictionary that will store the training and evaluation metrics at each evaluation step,
    # including the number of epochs completed, the number of updates performed, and the corresponding loss and accuracy values for both training and test sets.

    updates = 0
    # updates is a counter that keeps track of the total number of parameter updates performed during training.

    for epoch_index in range(epochs):
        model.train()
        # model.train() sets the model to training mode.

        order = rng.permutation(len(train_features))
        # rng.permutation generates a random permutation of indices for the training data,
        # which is used to shuffle the training data at the beginning of each epoch.

        order = torch.as_tensor(
            order,
            dtype=torch.long,
            device=device,
        )

        for start in range(0, len(train_features), effective_batch_size):
            batch_indices = order[start:start + effective_batch_size]
            # batch_indices selects a subset of indices from the shuffled order to create a mini-batch of training data.

            batch_features = train_features[batch_indices]
            # batch_features selects the corresponding features for the current mini-batch using the batch_indices.

            batch_targets = train_targets[batch_indices]
            # batch_targets selects the corresponding target labels for the current mini-btach using the batch_indices.

            optimizer.zero_grad(set_to_none=True)
            # optimizer.zero_grad(set_to_none=True) clears the gradients of all optimized parameters,
            # setting them to None instead of zeroing them out, which can improve memory efficiency.

            logits = model(batch_features)
            # forward computation:
            # (B, D) -> (B, C)

            loss = criterion(logits, batch_targets)
            # loss computation:
            # logits: (B, C), targets: (B,) -> loss: scalar
            # (B,) means that the targets are a 1D tensor of length B,
            # where each element is an integer representing the class label for the corresponding sample in the batch.
            # for example, if B=4 and there are 3 classes (C=3), the targets tensor might look like [0, 2, 1, 0],
            # indicating that the first sample belongs to class 0, the second sample belongs to class 2, the third sample belongs to class 1, and the fourth sample belongs to class 0.

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch_index + 1}."
                )

            loss.backward()
            # backpropagation computes the gradients for every trainable parameter involved in the forward computation.

            # This is where the actual parameter updates happen.

            optimizer.step()
            # optimizer.step() updates the model's parameters based on the computed gradients and the learning rate.

            updates += 1
            # updates counter is incremented to keep track of the total number of parameter updates performed during training.

        # Match the evaluation timing of the Assignment 1 trainer:
        # epoch indices 0, 10, 20, ..., plus the final epoch.
        #
        # Store completed epoch counts as 1, 11, 21, ...
        # so plot labels accurately describe the training performed.
        should_evaluate = (
            epoch_index % eval_freq == 0
            or epoch_index == epochs - 1
        )

        if should_evaluate:
            train_metrics = evaluate(
                model,
                train_features,
                train_targets,
                criterion,
            )
            test_metrics = evaluate(
                model,
                test_features,
                test_targets,
                criterion,
            )

            completed_epochs = epoch_index + 1

            history["epochs"].append(completed_epochs)
            history["updates"].append(updates)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["test_loss"].append(test_metrics["loss"])
            history["test_accuracy"].append(test_metrics["accuracy"])

            print(
                f"Epoch {completed_epochs:4d}/{epochs} | "
                f"Updates {updates:6d} | "
                f"Train loss {train_metrics['loss']:.4f} | "
                f"Train accuracy {train_metrics['accuracy']:.2%} | "
                f"Test loss {test_metrics['loss']:.4f} | "
                f"Test accuracy {test_metrics['accuracy']:.2%}"
            )

    return history

def train(
    dnn_hidden_units: str = DNN_HIDDEN_UNITS_DEFAULT,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    max_steps: int = MAX_EPOCHS_DEFAULT,
    eval_freq: int = EVAL_FREQ_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    seed: int = SEED_DEFAULT,
):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """

    hidden_units = [
        int(value.strip())
        for value in dnn_hidden_units.split(",")
        if value.strip()
    ]

    if not hidden_units:
        raise ValueError("Specify at least one hidden-layer width.")

    torch.manual_seed(seed)
    # Seed model initialization. Data generation and shuffling have
    # their own explicitly seeded random generators.

    x_train, x_test, y_train, y_test = make_moons_data(seed)

    model = MLP(
        n_inputs=x_train.shape[1],
        n_hidden=hidden_units,
        n_classes=2,
    )
    # this small network runs efficiently on the CPU.

    history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_rate=learning_rate,
        epochs=max_steps,
        batch_size=batch_size,
        eval_freq=eval_freq,
        seed=seed,
    )

    return model, history

def main():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP on the two-moons dataset."
    )

    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma-separated hidden-layer widths.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_EPOCHS_DEFAULT,
        help="Number of complete training epochs.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="0 for full batch, 1 for SGD, or a mini-batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )

    args = parser.parse_args()

    model, history = train(
        dnn_hidden_units=args.dnn_hidden_units,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_freq=args.eval_freq,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
