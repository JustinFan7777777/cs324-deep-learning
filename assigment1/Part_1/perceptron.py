import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1000, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """

        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")

        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        # Fill in: Initialize number of inputs
        self.n_inputs = n_inputs

        # Fill in: Initialize maximum number of epochs
        self.max_epochs = max_epochs

        # Fill in: Initialize learning rate
        self.learning_rate = learning_rate

        # Fill in: Initialize weights with zeros
        # the extra 1 stands for 'bias' term, so in total: n_inputs + 1 weights
        self.weights = np.zeros(n_inputs + 1)
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
        input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """

        # asarray converts the input to a numpy array if it is not already one
        input_vec = np.asarray(input_vec, dtype=float)

        # add a bias term to the input vector
        # e.g. [x1, x2] -> [x1, x2, 1]
        # ndim is the number of dimensions of the array,
        # if it is 1D, we add a bias term
        if input_vec.ndim == 1:
            input_with_bias = np.append(input_vec, 1)
        elif input_vec.ndim == 2:
            # if it is 2D, we add a bias term to each sample
            input_with_bias = np.concatenate(
                 [
                      input_vec,
                      # shape[0] = number of samples, we add a bias term to each sample
                      # this creates a column of ones with the same number of rows as input_vec
                      np.ones((input_vec.shape[0], 1)),
                 ],

                 # axis=1 means we concatenate along the columns (horizontally)
                 axis=1,
            )
        else:
            raise ValueError("input_vec must be 1D or 2D array")

        # matrix multiplication of input_with_bias and weights to get the scores
        # note that the score is a scalar for each sample, and we will use it to determine the predicted label
        scores = input_with_bias @ self.weights

        # step function: score -> predicted label
        # 1. score >= 0 -> predicted label = 1
        # 2. score < 0 -> predicted label = -1
        predictions = np.where(scores >= 0, 1, -1)

        # input = vector -> score = scalar -> predicted label = scalar (1 or -1)
        if input_vec.ndim == 1:
            return int(predictions)

        # input = matrix -> score = vector -> predicted label = vector (with each element being 1 or -1)
        return predictions
        
    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        # we need max_epochs to train our model

        training_inputs = np.asarray(training_inputs, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if training_inputs.ndim != 2:
            raise ValueError("training_inputs must be a 2D array")

        if training_inputs.shape[0] != labels.shape[0]:
            raise ValueError("Inputs and labels must have the same length")

        if not np.all(np.isin(labels, [-1, 1])):
            raise ValueError("Labels must be either -1 or +1")

        inputs_with_bias = np.concatenate(
            [
                training_inputs,
                np.ones((training_inputs.shape[0], 1)),
            ],
            axis=1,
        )

        for _ in range(self.max_epochs): 
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """

            scores = inputs_with_bias @ self.weights

            # misclassified samples:
            # 1. score >= 0 but was labeled -1 -> product <= 0
            # 2. score < 0 but was labeled 1 -> product <= 0
            misclassified = labels * scores <= 0

            # all samples are correctly classified -> stop training
            if not np.any(misclassified):
                break

            # loss function: mean of misclassified samples' negative product of label and score
            # gradient of the loss function with respect to weights = -mean(y_i * x_i),
            # where y_i is the label and x_i is the input vector (with bias) of the misclassified samples
            gradient = -np.mean(
                labels[misclassified, None]
                * inputs_with_bias[misclassified],
                axis=0,
            )

            # gradient descent update rule: w = w - learning_rate * gradient
            # if gradient > 0 -> decrease weights to reduce loss
            # if gradient < 0 -> increase weights to reduce loss
            # so there is a negative sign in front of the gradient to ensure we move in the right direction
            self.weights -= self.learning_rate * gradient
