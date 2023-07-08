# Individual Neuron
# Created by Apex Incarna 4/18/2023
# Represents an individual neuron in a neural network
# Created based on this video:  
# https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3 


inputs = [1.2,7.1,2.1]
weights = [3.1,0.1,8.7]
bias = -30

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)