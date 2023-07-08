# Implementing Loss
# Created by Apex Incarna 5/24/2023
# Takes what we learned from the loss calculation to improve the model
# Created based on this video:  
# https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=12

import numpy as np

# Calculating Loss with a batch
# We managed to calculate loss with one sample, but here's how to do it from a batch: 

# Sample outputs from an array of output neurons; remember, this is a batch, so each inner
# bracket pair indicates one output from the model based on one sample of data, while all of them
# together represent a batch of samples given to the model, which of course returned a batch of 
# outputs
softmaxOutputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
# The targets for our output data; the first sample in the batch should have given a 1 for the 
# zeroeth index, the second for the first index, and the third for the first: 
targets = [0, 1, 1]

# What this code does is it takes the numpy array that we created and only gives us back the 
# specific values we ask for; our array has two dimensions, so our argument has two parameters:
# the first just simply asks the array to look at the first (0) sample, then the next, then 
# the next. The second parameter asks numpy to only look at the numbers in the index that the
# target is looking for; for instance, if the model was supposed to return a 1 at the 0th index,
# then when we run this code it will present us with whatever the model returned at the 0th index;
# from there, we can determine how far off from 1 that result was based on our loss calculation
hotOutputs = softmaxOutputs[[0, 1, 2], targets]

# We can simplify this even further by just having python do the first parameter (which would be 
# tedious to do manually for a larger batch): 
hotOutputs = softmaxOutputs[range(len(softmaxOutputs)),targets]
print(hotOutputs)

# Now to get the loss we just do categorical cross-entropy, which just means taking the negative
# log of the whole thing: 
loss = -np.log(softmaxOutputs[range(len(softmaxOutputs)),targets])

# This will give us the loss for each sample in our batch: 
print(loss)

# We can calculate the average loss of the entire batch by taking the mean of the loss array: 
averageLoss = np.mean(loss)
print(averageLoss)

# One problem we might run into now is if the confidence of the hot nueron (the value of the 
# neuron that's supposed to be equal to 1) is zero, we get an error, because you can't take 
# the log of zero; the model would have to be very, very wrong to get the completely opposite
# answer from the one we were looking for, but because it's possible, we have to deal with it.
# A simple fix is to clip all values so that they can't be equal to zero, but are still close: 
clippedData = np.clip(softmaxOutputs,1e-7,1 - 1e-7)