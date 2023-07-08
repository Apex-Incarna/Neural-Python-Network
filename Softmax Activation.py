# Softmax Activation
# Created by Apex Incarna 4/29/2023
# Implementing the softmax activation function for use in the output layer of a neural network
# Created based on this video:  
# https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7


# A lot of times we are determining how correct a neural network's prediction was based on 
# a neuron's output relative to other neurons; but if we use ReLU, there is no connection to 
# other neurons; a 50 on one neuron means nothing relative to a -39 on another
# An additional problem is that ReLU is unbounded; 50 might seem like a large number, but we
# really can't say, because the possible range is actually infinite; 1,374,293,438 is just as
# valid an output for our neuron as 50
# Using a different activation function than ReLU on our output neurons fixes all of these 
# problems

# The activation function we will be using is the softmax activation function; the reasoning 
# behind this is better explained by the video than by this blurb, but in essence softmax is 
# an exponential function that allows us to cram our values within a bound that we can then
# use to compare them to each other (it also deals with some weird problems with negative 
# numbers)

import numpy as np
import nnfs

nnfs.init()


'''
# Simple example in raw python
# If we're using the e for our exponential, we can import it using math
import math
E = math.e

layerOutputs = [4.8,1.21,2.385]

expValues = []

# Exponentiation (gets rid of the negatives but keeps the meaning)
for output in layerOutputs: 
    expValues.append(E**output)

print(expValues)

# Normalization (makes all of the numbers mean something relative to each other)
normalBase = sum(expValues)
normalValues = []

for value in expValues: 
    normalValues.append(value/normalBase)

print(normalValues)
# The sum of normalValues should be equal to (approximately) 1
print(sum(normalValues))
'''

'''
# numpy example
layerOutputs = [4.8,1.21,2.385]

expValues = []

# Exponentiation (gets rid of the negatives but keeps the meaning)
expValues = np.exp(layerOutputs)

# Normalization (makes all of the numbers mean something relative to each other)
normalValues = expValues / np.sum(expValues)

print(normalValues)
# The sum of normalValues should be equal to (approximately) 1
print(sum(normalValues))

# This combination of exponentiation and normalization is what makes up the softmax
# activation function
'''

# Upgrading to using batches
layerOutputs = [[4.8,1.21,2.385],
                [8.9,-1.81,0.2],
                [1.41,1.051,0.026]]

expValues = []


# Exponentiation (gets rid of the negatives but keeps the meaning)
expValues = np.exp(layerOutputs)

# Normalization (makes all of the numbers mean something relative to each other)
# To normalize something, we divide each value by the sum of all of the values, so our
# result will be something that is between 0 and 1 and can be treated almost like a %
# Since we're adding up batches, we need to pass the axis parameter so numpy knows we
# want to add up each individual batch, instead of just adding up the entire matrix, 
# and the keepdims command to keep the dimensions of the original matrix so that when 
# we go to divide each value in a batch is divided by the correct sum of that batch instead
# of being divided by some other batch's sum: 
normalValues = expValues / np.sum(expValues, axis=1, keepdims=True)


print(normalValues)
# The sum of normalValues should be equal to (approximately) 1
print(sum(normalValues))

# This combination of exponentiation and normalization is what makes up the softmax
# activation function
