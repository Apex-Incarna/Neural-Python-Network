# The Dot Product
# Created by Apex Incarna 4/18/2023
# Uses vectors and matrices to clean up the neural network code
# Created based on this video:  
# https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4


# Raw Python without dot product: 
'''
inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# This code does all of the calculations for all of the neurons. 
layerOutputs = [] # Output of current layer
for neuronWeights, neuronBias in zip(weights, biases):
    neuronOutput = 0 # Output of a given neuron
    for nInput, weight in zip (inputs,neuronWeights):
        neuronOutput += nInput*weight
    neuronOutput += neuronBias
    layerOutputs.append(neuronOutput)

print(layerOutputs)
'''

# Python with numpy and with dot product: 
import numpy as np

inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

output = np.dot(weights, inputs) + biases
print(output)