# original: https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
# my avg result for sgd: 66.016%
# my avg result for batch: 56.748%
# Logistic Regression on Diabetes Dataset
import random
import numpy as np
from csv import reader
from math import exp
from sklearn.preprocessing import MinMaxScaler

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    return sum(actual == predicted) / float(len(actual)) * 100.0

# Make a prediction with coefficients
def predict(w, x):
    return 1.0 / (1.0 + exp(-np.dot(w, x)))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train_x, train_y, l_rate, n_epoch):
    w = [0.0 for i in range(len(train_x[0]))]
    for epoch in range(n_epoch):
        for i in range(len(train_y)):
            yhat = predict(w, train_x[i]) # probability of it being +1
            error = train_y[i] - yhat # define loss function here
            w = w + l_rate * error * train_x[i]
    return w

def coefficients_batch(train_x, train_y, l_rate, n_epoch):
    w = [0.0 for i in range(len(train_x[0]))]
    for epoch in range(n_epoch):
        summation = 0
        for i in range(len(train_y)):
            yhat = predict(w, train_x[i])
            summation = summation + (train_y[i] - yhat)*train_x[i]
        w = w + l_rate * summation
    return w

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train_x, train_y, test_x, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_batch(train_x, train_y, l_rate, n_epoch)
    for x in test_x:
        yhat = predict(x, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(x, y, algorithm, n_folds, *args):
    fold_size = int(len(dataset) / n_folds)
    scores = list()
    for i in range(n_folds):
        idx = random.sample(range(len(y)), fold_size)
        train_x = x[idx]
        train_y = y[idx]
        test_x = np.delete(x, idx, axis=0)
        test_y = np.delete(y, idx, axis=0)

        predicted = algorithm(train_x, train_y, test_x, *args)
        accuracy = accuracy_metric(test_y, predicted)
        scores.append(accuracy)
    return scores

# Test the logistic regression algorithm on the diabetes dataset
random.seed(1)

# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = np.genfromtxt(filename, delimiter=',')
np.array(dataset).astype(float)
split = np.split(dataset, [-1], axis=1)
x = np.insert(split[0], 0, 1, axis=1)

y = np.ndarray.flatten(split[1])

# normalize
x = MinMaxScaler().fit_transform(x)

# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(x, y, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % np.mean(scores))