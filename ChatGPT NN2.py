import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

### Objects
# Initialization and forward object
class LayerDense: 
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs  # Store inputs for the backward pass
    def backward(self, dvalues):
        # Compute gradients w.r.t. weights, biases, and input
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    def update(self, learning_rate):
        # Update weights and biases
        self.weights += -learning_rate * self.dweights
        self.biases += -learning_rate * self.dbiases

# ReLU object
class ReLU: 
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        # Gradient with respect to the inputs
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0

# Softmax object
class Softmax: 
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normal_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = normal_values
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Loss object
class Loss:
    def calculate(self, output, y):
        return self.forward(output, y)
    def backward(self, output, y):
        self.dinputs = self.dloss(output, y)
    def get_loss(self):
        return self.mean_loss

# Categorical Cross-Entropy object
class LossCCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        negative_log_likelihoods = -np.log(y_pred_clipped[range(samples), y_true])
        self.losses = negative_log_likelihoods
        self.mean_loss = np.mean(negative_log_likelihoods)
        return negative_log_likelihoods
    
    def dloss(self, output, y):
        samples = len(output)
        self.dinputs = output.copy()
        self.dinputs[range(samples), y] -= 1
        self.dinputs /= samples
        return self.dinputs


### Neural Network
np.random.seed(0)

# Data 
X, y = spiral_data(samples=100, classes=3)

# Define layer 1
layer1 = LayerDense(2, 64)
# Define ReLU activation for layer 1
activation1 = ReLU()

# Define layer 2
layer2 = LayerDense(64, 3)
# Define softmax activation for layer 2 (output layer)
activation2 = Softmax()

# Training parameters
epochs = 2000
learning_rate = 0.1
loss_function = LossCCE()

# Training loop
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)
    loss_history.append(loss)

    # Backpropagation
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # Update weights and biases
    layer2.update(learning_rate)
    layer1.update(learning_rate)

    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    accuracy_history.append(accuracy)

    # Print the updated loss and accuracy for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss_function.get_loss(), "Accuracy:", accuracy)

# Final loss after training
print("Final Loss:", loss_function.get_loss())

# Plotting loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
