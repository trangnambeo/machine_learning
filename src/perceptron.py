# -*- coding: utf-8 -*-
"""
@author Nam Ly

Implement perceptron learning
"""
from classifier import Binary_Classifier_Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Class to implement perceptron learning
"""
class Perceptron(Binary_Classifier_Base):
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
            errors = 0
            for sample, target in zip(data_train, data_labels):
                update = self.m_eta * (target - self.predict(sample))
                self.m_weights[0] += update
                self.m_weights[1:] += update*sample
                errors += int(update != 0.0)
            self.m_error.append(errors)
    
    ##########################################################################
    ## Predict label of a data sample
    ##
    ## @param self 
    ## @param sample a data sample to predict
    ##
    ## @return label of a data sample, 1 or -1
    ##########################################################################
    def predict(self, sample):
        return np.where(self.net_input(sample) >= 0.0, 1, -1)

## Main
# Getting data
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x',
            label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

# Apply Perceptron on data
ppn = Perceptron(eta=0.1, num_iter=10)
ppn.fit(X, y)

# Plot error rate
ppn.plot_error_rate()

