# original: https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
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
def coefficients_sgd(train_x, train_y, l_rate, n_epoch, n_cats):
    n_examples = len(train_y)
    x_dim = len(train_x[0])
    w = np.zeros((n_cats, x_dim))
    for epoch in range(n_epoch): # for the number of epochs
        for i in range(n_examples): # for each of the examples
            for cat in range(n_cats): # calculate the prediction for each of the cats
                yhat = predict(w[cat], train_x[i]) # probability of it being category cat
                error = (train_y[i] == cat) - yhat # define loss function here
                w[cat] = w[cat] + l_rate * error * train_x[i]
        print(epoch)
    return w

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train_x, train_y, test_x, l_rate, n_epoch):
    predictions = list()
    n_cats = len(np.unique(train_y))
    print(n_cats)
    coef = coefficients_sgd(train_x, train_y, l_rate, n_epoch, n_cats)
    for x in test_x:
        yhat = np.zeros(n_cats)
        for cat in range(n_cats):
            yhat[cat] = predict(x, coef[cat])
        prediction = np.argmax(yhat)
        predictions.append(prediction)
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
filename = 'letter-recognition.csv'
dataset = np.genfromtxt(filename, delimiter=',', dtype='<U3')

split = np.split(dataset, [1], axis=1)
x = np.insert(split[1], 0, 1, axis=1).astype('float')
convert_to_num = np.vectorize(ord)
y = convert_to_num(np.ndarray.flatten(split[0])) - ord('A')

# normalize
x = MinMaxScaler().fit_transform(x)

# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 150
scores = evaluate_algorithm(x, y, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % np.mean(scores))