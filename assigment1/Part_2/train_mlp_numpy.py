import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

BATCH_SIZE_DEFAULT = 0

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets

    # Get the predicted classes by taking the argmax along axis 1 (columns)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get the true classes from the one-hot encoded targets
    target_classes = np.argmax(targets, axis=1)

    # Calculate the accuracy as the mean of correct predictions
    # Note: np.mean returns a float, which is the fraction of correct predictions,
    # but not as a percentage, by multiplying by 100.
    return float(np.mean(predicted_classes == target_classes))

def train(
    dnn_hidden_units,
    learning_rate,
    max_steps,
    eval_freq,
    batch_size=None,
    random_seed=42,
):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # Validate training configuration before creating the dataset.
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None")

    # generate 1000 samples of a 2D dataset with two interleaving half circles (moons)
    x, labels = make_moons(
        n_samples=1000,
        noise=0.2,
        random_state=random_seed,
    )

    # convert labels 0 & 1 to one-hot encoding
    targets = np.zeros((len(labels), 2))
    targets[np.arange(len(labels)), labels] = 1

    # randomly split the dataset into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        targets,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels,  # ensures that both classes are represented proportionally in train and test sets
    )
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    hidden_units = [
        int(unit.strip())
        for unit in dnn_hidden_units.split(',')
        if unit.strip()
    ]
    if not hidden_units:
        raise ValueError("dnn_hidden_units must contain at least one layer size")

    # Seed NumPy so the model initialization is reproducible.
    np.random.seed(random_seed)
    model = MLP(
        n_inputs=x_train.shape[1],
        n_hidden=hidden_units,
        n_classes=y_train.shape[1],
    )

    loss_function = CrossEntropy()

    # Store metrics at each evaluation point so the notebook can plot curves.
    history = {
        "steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    # Use the whole training set for batch gradient descent by default.
    effective_batch_size = (
        len(x_train) if batch_size is None else batch_size
    )

    # Use a local generator so shuffling is reproducible without changing
    # NumPy's global random state after model initialization.
    rng = np.random.default_rng(random_seed)
    
    for step in range(max_steps):
        # TODO: Implement the training loop
        # Shuffle the examples once per epoch for SGD and mini-batch training.
        permutation = rng.permutation(len(x_train))
        shuffled_x = x_train[permutation]
        shuffled_y = y_train[permutation]

        for start in range(0, len(shuffled_x), effective_batch_size):
            end = start + effective_batch_size
            x_batch = shuffled_x[start:end]
            y_batch = shuffled_y[start:end]

            # 1. Forward pass
            predictions = model.forward(x_batch)

            # 2. Compute loss
            loss_gradient = loss_function.backward(
                predictions,
                y_batch,
            )

            # 3. Backward pass (compute gradients)
            model.backward(loss_gradient)

            # 4. Update weights
            for layer in model.layers:
                if hasattr(layer, "params"):
                    layer.params["weight"] -= (
                        learning_rate
                        * layer.grads["weight"]
                    )

                    layer.params["bias"] -= (
                        learning_rate
                        * layer.grads["bias"]
                    )
        
        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            test_predictions = model.forward(x_test)

            # 2. Compute loss and accuracy
            test_loss = loss_function.forward(
                test_predictions,
                y_test,
            )

            test_accuracy = accuracy(
                test_predictions,
                y_test,
            )

            # Evaluate the training set with the same model state.
            train_predictions = model.forward(x_train)
            train_loss = loss_function.forward(
                train_predictions,
                y_train,
            )
            train_accuracy = accuracy(
                train_predictions,
                y_train,
            )

            # Save both training and test metrics for later visualization.
            history["steps"].append(step)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            print(
                f"Step: {step}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2%}, "
                f"Loss: {test_loss:.4f}, "
                f"Accuracy: {test_accuracy:.2%}"
            )
    
    print("Training complete!")
    return model, history

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    # This argument is reserved for the Part III extension.
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help='Batch size; 0 uses the full training set',
    )
    FLAGS = parser.parse_known_args()[0]
    
    # Preserve the original full-batch behavior when batch_size is zero.
    train(
        FLAGS.dnn_hidden_units,
        FLAGS.learning_rate,
        FLAGS.max_steps,
        FLAGS.eval_freq,
        None if FLAGS.batch_size == 0 else FLAGS.batch_size,
    )

if __name__ == '__main__':
    main()
