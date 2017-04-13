# -*- coding: utf-8 -*-
"""
@author Nam Ly

Implement perceptron learning
"""
import numpy as np

"""
Class to implement perceptron learning
"""
class Perceptron(object):
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
        self.m_weights = []
        self.m_error = 0
    
    ##########################################################################
    ## Calculate perceptron weights for a given data train set
    ##
    ## @param self 
    ## @param data_train data train set in matrix format[number of samples,
    ##  number of features]
    ## @param data_labels labels of samples in data train set
    ##########################################################################
    def fit(self, data_train, data_labels):
        self.m_weights = np.zeros(1 + data_train.shape[1])
        for _ in range(self.m_num_iter):
            for sample, target in zip(data_train, data_labels):
                update = self.m_eta(target - self.predict(sample))
                #self.m_weights[0] += update
                #self.m_weights[1:] += update*sample
                self.m_weights += update*(np.array[1, sample])
                
    
    ##########################################################################
    ## Predict label of a data sample
    ##
    ## @param self 
    ## @param sample a data sample to predict
    ##
    ## @return label of a data sample, 1 or -1
    ##########################################################################
    def predict(self, sample):
        return np.where((np.dot(sample, self.m_weights[1:]) +
                         self.m_weights[0]) >= 0.0, 1, -1)