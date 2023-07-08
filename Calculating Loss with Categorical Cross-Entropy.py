# Calculating Loss with Categorical Cross-Entropy
# Created by Apex Incarna 5/6/2023
# Calculates the loss or cost of a neural network (finally!)
# Created based on this video:  
# https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8

# Categorical Cross-Entropy is a popular loss function that is used to 
# determine how wrong or incorrect our neural network is, because once we
# know how wrong it is, we can use that information to tweak our weights and
# biases in order to make the model more right. 

# CCE takes the negative sum of the target values multiplied by the ln of
# the predicted values for each value in the distribution
# Using something called "one-hot encoding" (using a vector filled with zeros
# for every value exept the one the model should predict) CCE simplifies to 
# taking the negative ln of the predicted value that was supposed to be 1
# The 'why' behind using CCE instead of other loss functions is that, for a 
# neural network who's primary function is to classify given data into one 
# category or another, it works (a big plus, I know), but it also makes 
# backpropagation and optimization easier

# An example using the math library
import math

# An example of data from the output neurons of a neural network
softmaxOuput = [0.7,0.1,0.2]

# This is where we start to use the one-hot encoding; the target class, or the
# nueron (class) that was supposed to be equal to 1 (hot) is at the 0th index
targetOutput = [1,0,0]

loss = -(math.log(softmaxOuput[0])*targetOutput[0] + 
         math.log(softmaxOuput[1])*targetOutput[1] +
         math.log(softmaxOuput[2])*targetOutput[2])
# Looking at this calculation, we see that targetOuput[1] and [2] are both 
# equal to zero, so those terms will drop out of the sum; that means that 
# our entire loss calculation only cares about how wrong the 'hot' neuron was
# not about how wrong any of the others were

print(loss)
