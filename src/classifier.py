# -*- coding: utf-8 -*-
"""
@author Nam Ly

Base classifier
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
Class to implement a binary classifier. Any classification system must inherit 
from it
"""
class Binary_Classifier_Base(metaclass=ABCMeta):
    ##########################################################################
    ## Initializer
    ##
    ## @param self 
    ## @param eta learning rate(between 0.0 and 1.0)
    ## @param num_iter number of iteration
    ##########################################################################
    def __init__(self, eta=0.1, num_iter=10):
        self.m_eta = eta
        self.m_num_iter = num_iter
        # [M x 1] weight vector
        self.m_weights = []
        self.m_error = []
    
    ##########################################################################
    ## Calculate perceptron weights for a given data train set
    ##
    ## @param self 
    ## @param data_train data train set in matrix format[number of samples,
    ##  number of features]
    ## @param data_labels labels of samples in data train set
    ##########################################################################
    @abstractmethod
    def fit(self, data_train, data_labels):
        return NotImplemented
    
    ##########################################################################
    ## Net input
    ##
    ## @param self 
    ## @param sample a [M x 1] data sample to calculate dot product
    ##
    ## @remarks calculate dot product between weights and sample
    ##########################################################################
    def net_input(self, sample):
        return (np.dot(sample, self.m_weights[1:]) + self.m_weights[0])

    ##########################################################################
    ## Predict label of a data sample
    ##
    ## @param self 
    ## @param sample a data sample to predict
    ##
    ##########################################################################
    @abstractmethod
    def predict(self, sample):
        return NotImplemented

    ##########################################################################
    ## Plot error rate of a classification system
    ##
    ## @param self 
    ##
    ##########################################################################
    def plot_error_rate(self):
        plt.plot(range(1, len(self.m_error) + 1), self.m_error, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.show()
    
    ##########################################################################
    ## Plot decision boundary of a classifiation system
    ##
    ## @param self 
    ##
    ##########################################################################
    def plot_decision_boundary(self, X, y, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red','blue','lightgreen','gray','cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() -1, X[:,0].max() + 1
        x2_min, x2_max = X[:, 1].min() -1, X[:,1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        # Plot decision surface
        plt.contourf(xx1, xx2, Z, alpha=0.4,cmap=cmap)
        
        # Plot class samples
        