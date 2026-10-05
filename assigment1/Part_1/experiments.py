import numpy as np

from perceptron import Perceptron


def create_dataset(
    mean_negative,
    mean_positive,
    variance,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)

    covariance = variance * np.eye(2)

    # negative_points: 100 samples from a 2D Gaussian distribution,
    # with mean = mean_negative and covariance = variance * I
    negative_points = rng.multivariate_normal(
        mean_negative,
        covariance,
        size=100,
    )

    # positive_points: 100 samples from a 2D Gaussian distribution,
    # with mean = mean_positive and covariance = variance * I
    positive_points = rng.multivariate_normal(
        mean_positive,
        covariance,
        size=100,
    )

    # first 80 samples of each class are used for training
    x_train = np.concatenate(
        [
            negative_points[:80],
            positive_points[:80],
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            -np.ones(80, dtype=int),
            np.ones(80, dtype=int),
        ]
    )

    # remaining 20 samples of each class are used for testing
    x_test = np.concatenate(
        [
            negative_points[80:],
            positive_points[80:],
        ],
        axis=0,
    )

    # test labels: first 20 samples are -1, last 20 samples are +1
    y_test = np.concatenate(
        [
            -np.ones(20, dtype=int),
            np.ones(20, dtype=int),
        ]
    )

    # shuffle the training and testing datasets,
    # to ensure that the order of samples does not affect the training process
    train_order = rng.permutation(len(x_train))
    test_order = rng.permutation(len(x_test))

    return (
        x_train[train_order],
        y_train[train_order],
        x_test[test_order],
        y_test[test_order],
    )


def accuracy(predictions, labels):
    """Return the fraction of correctly classified samples."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")

    return float(np.mean(predictions == labels))


def run_experiment(
    mean_negative,
    mean_positive,
    variance,
    title,
    random_seed=42,
):
    """Generate data, train a perceptron, and report both accuracies."""
    x_train, y_train, x_test, y_test = create_dataset(
        mean_negative=mean_negative,
        mean_positive=mean_positive,
        variance=variance,
        random_seed=random_seed,
    )

    model = Perceptron(
        n_inputs=2,
        max_epochs=1000,
        learning_rate=0.01,
    )
    model.train(x_train, y_train)

    train_accuracy = accuracy(model.forward(x_train), y_train)
    test_accuracy = accuracy(model.forward(x_test), y_test)

    print(title)
    print(f"  training samples: {len(x_train)}")
    print(f"  test samples: {len(x_test)}")
    print(f"  training accuracy: {train_accuracy:.2%}")
    print(f"  test accuracy: {test_accuracy:.2%}")
    print(f"  final weights: {model.weights}")
    print()

    return model, train_accuracy, test_accuracy


def main():
    """Run the separated, close-mean, and high-variance experiments."""
    run_experiment(
        mean_negative=[-2.0, -2.0],
        mean_positive=[2.0, 2.0],
        variance=0.25,
        title="Well-separated Gaussian distributions",
    )

    run_experiment(
        mean_negative=[-0.5, -0.5],
        mean_positive=[0.5, 0.5],
        variance=0.25,
        title="Gaussian means are close",
    )

    run_experiment(
        mean_negative=[-2.0, -2.0],
        mean_positive=[2.0, 2.0],
        variance=2.0,
        title="Gaussian variance is high",
    )


if __name__ == "__main__":
    main()