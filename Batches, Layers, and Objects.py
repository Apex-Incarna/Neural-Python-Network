# Batches, Layers, and Objects
# Created by Apex Incarna 4/25/2023
# Progresses to modeling multiple layers of neurons instead of just one, leading towards creating objects
# Created based on this video:  
# https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4


import numpy as np

'''
# Single Layer

# This is a list:
aList = [1,
         2,
         3,
         2.5]


# This is a list of list, AKA a batch: 
inputs = [[1,2,3,2.5],
        [2,5,-1,2],
        [-1.5,2.7,3.3,-0.8]]

# We don't need to change our weights when chainging our batch size, 
# because the weights correspond to the number of outputs, not the number of inputs

# This layer contains 3 output neurons because there are three weight sets
weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# Since inputs is now a different shape of array (now it's a 3 x 4 instead of a 4 by 1),
# and you can't do a dot product of a 3 x 4 with a 3 x 4 because 3 != 4, 
# we must change how we're calculating the dot product: we must do a transpose

# A transpose is when you switch rows and columns of an array; that way, our 3 x 4 weights
# array becomes a 4 x 3 array, and suddenly we can do the dot product
output = np.dot(inputs, np.array(weights).T) + biases
print(output)
'''
'''
# Multiple layers

# The weights and biases for layer 1
inputs = [[1,2,3,2.5],
        [2,5,-1,2],
        [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# The weights and biases for layer 2
# We need weights2 to be a 3 x n matrix because we are getting a 3 x 3 matrix of inputs
# from the 3 x 4 dot 4 x 3 in layer 1
weights2 = [[0.1,-0.14,0.5],
           [-0.5,0.12,-0.33],
           [-0.44,0.73,-0.13]]

biases2 = [-1,2,-0.5]

layerOneOutput = np.dot(inputs, np.array(weights).T) + biases
# The output from layer one becomes the input to layer two
layerTwoOutput = np.dot(layerOneOutput, np.array(weights2).T) + biases2

print(layerTwoOutput)
'''

# Objects

# At this point, it is very cumersome to add more layers or to change
# the number of neurons in each layer; with objects, we can change that

np.random.seed(0)

# Raw input data (standard to denote with 'X')
# Here we have a "batch" of data, or several different input data sets that we want the model to
# interpret all at once; here we use only three, but for training a typical batch size might be 32
# This means that our output will be the network's interpretation of the first data set, followed 
# by the second, followed by the third
X = [[1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8]]

# Making the Object: 
# Don't be bothered by all of the fancy python; only focus on what this is doing
# This code creates a "LayerDense" object
# It will "initialize" ("__init__" method, runs on activation) a neural network by 
# coming up with random weights 
# It then forwards ("forward" method) the data from each successive layer to the next
class LayerDense: 
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
# Some key elements of the above code: 
# - randn is trying to give values between -1 and 1, which is good for a neural network
#   because you wnat small numbers
# - The multiplication by 0.10 is just trying to really make sure these numbers are small
# - Making the self.weights matrix nInputs x nNeurons instead of the other way around like
#   we did originally just means we don't have to do the transpose every single time we want 
#   to "forward pass," or pass our output data forward into the next layer of neurons
# - The double () for the self.biases line is necessary; it's just necessary to tell python
#   that we want the matrix shape of biases to be 1 x nNeurons

# Now we are creating the layers; the first component will be the number of inputs, and the 
# second number will be the number of neurons in the layer, which is however many we want! 
layer1 = LayerDense(4,5)
# The only requirements for layer 2 is that it has the same number of inputs as layer 1
# had outputs, and then we get to choose again however many neurons (and thus outputs) this
# layer will have
layer2 = LayerDense(5,2)

# Now we can start actually passing data to our layers with our forward method;
# remember that our initialization of the model, where we come up with our 
# random weights, has already occured
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)