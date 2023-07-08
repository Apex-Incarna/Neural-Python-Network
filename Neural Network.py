# Neural Network
# Created by Apex Incarna 5/2/23
# Inititalizes a neural network, runs a sample data set through it, and uses activation functions, as well as calculates loss


### Import libraries
import numpy as np

# Here is a library created by the video creators to make sure everyone gets the same answers
# when they write the same code
import nnfs

# One thing we can use nnfs for is to create data sets on which we can train our models; this can
# be useful because we don't have to type it all out by hand and we can get some really interesting
# sample data sets really quickly:
from nnfs.datasets import spiral_data

# nnfs also gets rid of the need for a randomizer seed:
nnfs.init()

### Objects
# Initialization and forward object
class LayerDense: 
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU object
class relu: 
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

# Softmax object
class softmax: 
    def forward(self,inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # The reason we subtract off the maximum value of a batch before exponentiating is so that
        # we don't get massive values approaching infinity (because the most they could be is 0)
        normalValues = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = normalValues

# Loss object: 
class Loss:    
    def calculate(self, output, y): 
        # output is the softmax output from the model, y is the target values
        sampleLosses = self.forward(output, y)
        # The reason we use the forward method instead of just using catagorical cross-entropy
        # here is that it gives us the flexibility to do multiple different loss functions
        dataLoss = np.mean(sampleLosses)
        return dataLoss

# Categorical Cross-Entropy object
class LossCCE(Loss): 
    # The (Loss) means this class will "inherit" from the base Loss class
    def forward(self, yPred, yTrue): 
        # yPred are the values predicted by the model, yTrue are the target values
        samples = len(yPred)
        yPredClipped = np.clip(yPred, 1e-7, 1-1e-7)
        
        # Sometimes people pass the target values using one-hot encoding: [[0,1],[1,0],etc.]
        # or by just listing as a scalar the indices where the hot neuron should be: [1,0]
        # These methods are used pretty interchangeably, so a polite thing for us to do is to
        # make sure this model can handle both

        # The following code is from ChatGPT; all it does is tests if the given targets are 
        # one-hot encoded--if they are, it will convert them to the scalar format using 
        # np.argmax; it then returns the proper calculation for the loss:
        ### Begin ChatGPT Code ###
        if len(yTrue.shape) == 2:
            # Convert one-hot encoded targets to categorical indices
            yTrue = np.argmax(yTrue, axis=1)

        negativeLogLikelihoods = -np.log(yPredClipped[range(samples), yTrue])
        return negativeLogLikelihoods
        ### End ChatGPT Code ###

### Neural Networking

# Data 
# Again, X denotes our training data; y denotes how many classes that training data can be 
# subdivided into; it's like how many categories the data fits in, and we might train our model
# to be able to sort those categories
X, y = spiral_data(samples=100, classes=3)

# Defining layer 1
layer1 = LayerDense(2,3)
# Defining our activation function for layer 1
active1 = relu()

# Defining layer 2
layer2 = LayerDense(3,3)
# Defining our activation function for layer 2 (output layer)
active2 = softmax()

layer1.forward(X)
# We run our activation function after we pass the data forward through our layer
active1.forward(layer1.output)

layer2.forward(active1.output)
active2.forward(layer2.output)

print(active2.output[:5])


### Calculating Accuracy

# Accuracy is a % reflection of how accurate the model is at correctly classifying the given data

outputs = np.array(active2.output)
predictions = np.argmax(outputs, axis=1)
accuracy = np.mean(predictions == y)

print("Accuracy:", accuracy)


### Calculating Loss

# Here we choose which loss function we want to use: 
lossFunction = LossCCE()

# Here we actually calculate the loss by putting in the predicted values (in this case activation
# 2.output) and the target values (y)
loss = lossFunction.calculate(active2.output, y)

print("Loss:", loss)


