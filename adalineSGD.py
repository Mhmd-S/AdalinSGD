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

# Add a partial fit method which does not reinitialize the weights
# calculate the loss as the average loss of the training examples in each epoch.
# Add the option to reshuffle the data b4 each epoch 
# SGD - Sochastic Gradient Descent
# Difference between SGD and Perceptron is the loss function. We are using the loss function to compute the change for weights.
 

class Adaline(object):
    
    def __init__(self, x, y):
        #  Having it on random does change the learning process
        self.eta = 0.01 
        #self.weights = np.zeros(units) Boundary wont change bcz the weights are all 0
        self.bias = np.float64(0.)
        self.losses_avg = []
        self.X = x
        self.y = y
        self.weights = rng.normal(loc=0.0, scale=0.01, size=x.shape[1])
        
    def fit(self, epochs, reshuffle = False):
        for i in range(epochs):
            losses = []
            
            # Adaptive learning rate, using, new_eta = c1 / [# of iterations] + c2, c1 and c2 are constants.
            self.eta = 1 / (i+100)
            
            if reshuffle:
                concat_data = np.column_stack((self.X, self.y))
                np.random.shuffle(concat_data)
                self.X = concat_data[:, 0:2]
                self.y = concat_data[:, 2]
            
            for xi, target in zip(self.X, self.y):
                # Got the outputs for all xi
                output = self.net_input(xi)
                error = target - output

                # Change this, check the func again
                change_w = self.eta * error * xi
                change_b = self.eta * error

                self.weights += change_w
                self.bias += change_b
                losses.append(self.calculate_loss(self.X, self.y)) 
            self.losses_avg.append(sum(losses)/len(losses))

        return self
    
    def partial_fit(self, X, y, epochs, reshuffle):
        for i in range(epochs):
            losses = []
            
            eta = 0.01
            
            if reshuffle:
                concat_data = pd.concat(X, y)
                np.random.shuffle(concat_data)
                self.X = concat_data[0:2]
                self.y = concat_data[2]
            
            for xi, target in zip(X, y):
                # Got the outputs for all xi
                output = self.net_input(xi)
                error = target - output

                # Change this, check the func again
                change_w = self.eta * error * xi
                change_b = self.eta * error

                self.weights += change_w
                self.bias += change_b
                losses.append(self.calculate_loss(self.X, self.y)) 
            
            self.eta = 1 / (i+100)
            self.losses_avg.append(losses.mean())
    
    def net_input(self, xi):
        return np.dot(xi, self.weights) + self.bias
    
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