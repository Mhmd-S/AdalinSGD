import numpy as np
import os
import pandas as pd

rng = np.random.default_rng()

# Adaline - Adaptive Linear Neuron Classifier
# Net input function -> Activation function -> Update Weights
# loss function = 1/nΣ(y-y^)^2 | MSE in this case
# ∂l/∂w = -2/nΣ(y-y^)*xi
# ∂l/∂b = -2/nΣ(y-y^)
# δw = -η * ∂L/∂w
# δb = -η * ∂L/∂b

# Add a partial fil method which does not reinitialize the weights
# In order to check whether our algorithm converged after training, we will calculate the loss as the average loss of the training examples in each epoch.
# Add the option to reshuffle the data b4 each epoch 
 

class Adaline(object):
    
    def __init__(self, x, y):
        #  Having it on random does change the learning process
        self.eta = 0.01 
        #self.weights = np.zeros(units) Boundary wont change bcz the weights are all 0
        self.bias = np.float64(0.)
        self.X = x
        self.y = y
        self.losses = []
        self.weights = rng.normal(loc=0.0, scale=0.01, size=x.shape[1])
        
    def fit(self, epochs):

        for _ in range(epochs):
            # Got the outputs for all xi
            output = self.net_input(self.X)
            errors = self.y - output
            
            change_w = self.eta * 2 * (self.X.T.dot(errors) / len(self.X))
            change_b = self.eta * 2 * errors.mean()
            
            self.weights += change_w
            self.bias += change_b
            self.losses.append(self.calculate_loss(self.X, self.y)) 

        return self
    
    def net_input(self, X):
        outputs = []
        for xi in X:
            outputs.append(np.dot(xi, self.weights) + self.bias)
        return outputs
    
    def calculate_loss(self, x, y):
        output = self.net_input(x)
        loss = ((y-output)**2).mean()
        return loss
    
    def predict(self, x):
        return np.where(np.dot(x, self.weights)+ self.bias >= 0.1, 1, 0)
                          
    def test(self, x, y):
        correct_predictions = 0
        total_predictions = len(x)
        
        for xi, target in zip(x, y):
            prediction = self.predict(xi)
            
            if prediction == target:
                correct_predictions += 1
            else:
                print(f'Incorrect Prediction: {prediction}, Target: {target}, Input: {xi}')
        
        accuracy = correct_predictions / total_predictions
        return accuracy