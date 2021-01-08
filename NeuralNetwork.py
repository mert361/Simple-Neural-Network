# -*- coding: utf-8 -*-
"""
@author: Mert
"""
#importing libraries
import numpy as np
#Using Sigmoid Normalization function to normalize data as a function.
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)
#Training input values and datas.
training_inputs=np.array([[0,0,1],
                          [1,1,1],
                          [1,0,1],
                          [0,1,1]])
#Training output datas as answers.
training_outputs=np.array([[0,1,1,0]]).T

np.random.seed(1)
#Adjusting synaptic weights
synaptic_weights=2*np.random.random((3,1))-1
print('begining synaptic weights')
print(synaptic_weights)
#Iteration process and Error handling
for iteration in range(100000):
    input_layer=training_inputs
    outputs=sigmoid(np.dot(input_layer,synaptic_weights))
    error=training_outputs-outputs
    adjustments=error-sigmoid_derivative(outputs)
    synaptic_weights=np.dot(input_layer.T,adjustments)
    
print('after training')
print(outputs)                    
