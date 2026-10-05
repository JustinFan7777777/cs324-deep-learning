import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """

        # Initialize weights and biases with the correct shapes.
        self.params = {

            # weights: shape (in_features, out_features)
            # 0.01 * np.random.randn(...) generates small random values from a normal distribution
            'weight': 0.01 * np.random.randn(
                in_features,
                out_features,
            ),

            # biases: shape (out_features,)
            'bias': np.zeros(out_features),
        }

        self.grads = {
            # weights and biases gradients initialized to zeros with the same shape as weights and biases
            'weight': np.zeros_like(self.params['weight']),
            'bias': np.zeros_like(self.params['bias']),
        }

        # x is stored for use in the backward pass
        self.x = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """

        self.x = x
        # forward pass: linear transformation of input x
        # formula: output = x @ weight + bias
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """

        if self.x is None:
            raise RuntimeError("forward must be called before backward")

        # gradient w.r.t. weights: dL/dW = x^T @ dout
        self.grads['weight'][...] = self.x.T @ dout

        # gradient w.r.t. bias: dL/db = sum(dout) over all samples
        # dout has shape (batch_size, out_features), so we sum over axis=0 to get shape (out_features,)
        self.grads['bias'][...] = np.sum(dout, axis=0)

        # gradient w.r.t. input: dL/dx = dout @ W^T
        # dout has shape (batch_size, out_features), weight has shape (in_features, out_features)
        # so the result will have shape (batch_size, in_features)
        dx = dout @ self.params['weight'].T
        return dx


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """

        self.x = x
        # element-wise ReLU activation: output = max(0, x)
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """

        if self.x is None:
            raise RuntimeError("forward must be called before backward")

        # dout is the upstream gradient, we multiply it by the gradient of ReLU
        # gradient of ReLU:
        # 1 for x > 0, 0 for x <= 0
        # (self.x > 0) creates a boolean array where True (1) for x > 0 and False (0) for x <= 0
        return dout * (self.x > 0)

# softmax: scores -> probabilities
class SoftMax(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        self.x = x

        # axis=1 -> eliminate the second dimension (columns) for each row
        # To prevent overflow, we subtract the max value in each row from all elements in that row before exponentiating.
        shifted_x = x - np.max(x, axis=1, keepdims=True)

        # compute the exponentials of the shifted inputs
        exp_x = np.exp(shifted_x)

        # exp_x has shape (batch_size, num_classes)
        # sum has shape (batch_size, 1) because we sum over axis=1 (columns)
        # keepdims=True ensures that the result has the same number of dimensions as exp_x,
        # allowing for proper broadcasting in the division.
        # so out will have shape (batch_size, num_classes) and each row will sum to 1
        self.out = exp_x / np.sum(
            exp_x,
            axis=1,
            keepdims=True,
        )

        return self.out

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        # CrossEntropy.backward already returns the gradient with respect
        # to the logits, so no additional softmax Jacobian is needed.
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """

        # clip the input to avoid log(0) which is undefined
        # clip means we limit the values of x to be within a certain range, in this case between 1e-12 and 1.0
        probabilities = np.clip(x, 1e-12, 1.0)

        losses = -np.sum(
            y * np.log(probabilities),
            axis=1,
            keepdims=True,
        )

        return np.mean(losses)


    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """

        # batch size is the first dimension of x, which is the number of rows
        batch_size = x.shape[0]

        # gradient of the loss w.r.t. input x
        # for softmax followed by cross-entropy, the gradient simplifies to: p - y
        return (x - y) / batch_size
