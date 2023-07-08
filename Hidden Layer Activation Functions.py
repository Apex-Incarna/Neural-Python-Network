# Hidden Layer Activation Functions
# Created by Apex Incarna 4/26/2023
# Implementing activation functions to determine the activation of a neuron
# Created based on this video:  
# https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6


# We will be using the ReLU activation function; while there are many options for activation 
# functions, RELU is both one of the most popular, one of the simplest and fastest, and one
# that does not have "gradient loss" that other popular functions like the sigmoid have

import numpy as np

# Here is a library created by the video creators to make sure everyone gets the same answers
# when they write the same code
import nnfs

# One thing we can use nnfs for is to create data sets on which we can train our models; this can
# be useful because we don't have to type it all out by hand and we can get some really interesting
# sample data sets really quickly:
from nnfs.datasets import spiral_data

nnfs.init()

# nnfs also gets rid of the need for a randomizer seed: 
# np.random.seed(0)

'''
# Simple ReLU case

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

# Raw Python
for i in inputs: 
    if i>0:
        output.append(i)
    elif i<=0:
        output.append(0)

# With numpy
for i in inputs: 
    output.append(max(0,i))
print(output)
'''

# Again, X denotes our training data; y denotes how many classes that training data can be 
# subdivided into; it's like how many categories the data fits in, and we might train our model
# to be able to sort those categories
X, y = spiral_data(100, 3)


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

# Instead of 4 inputs to our neural network, we just have 2: one for y and one for x, because our
# spiral data set creates a bunch of points on the (x,y) plain 
# And again, 5 is the number of neurons in layer1, which will be our only layer for this example
layer1 = LayerDense(2,5)
# Defining our activation function
active1 = relu()

layer1.forward(X)
# We run our activation function after we pass the data forward through our layer
active1.forward(layer1.output)

print(active1.output)

