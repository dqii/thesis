import random
import numpy as np

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    return sum(actual == predicted) / float(len(actual)) * 100.0

# Estimate logistic regression coefficients using stochastic gradient descent
def sgd(train_x, train_y, lr, n_epoch, n_classes, alg):
    
    n_examples = len(train_y)
    x_dim = len(train_x[0])
    w = np.zeros((n_classes, x_dim))
    loss = np.zeros(n_epoch)
    
    for epoch in range(n_epoch):
        
        shuffle = np.random.permutation(n_examples)
        train_x = train_x[shuffle]
        train_y = train_y[shuffle]
        
        for i in range(n_examples):
            x = train_x[i,:]
            y = np.zeros(n_classes)
            y[train_y[i]] = 1
            
            score = alg['score'](w, x)    
            loss[epoch] += alg['loss'](y, score)
            gradient = alg['gradient'](x, y, score)
            
            w -= lr * gradient
    return w, loss/n_examples

# Run SGD over the cross-validation split
def evaluate_split(train_x, train_y, test_x, lr, n_epoch, alg):    
    n_classes = len(np.unique(train_y))
    coef, loss = sgd(train_x, train_y, lr, n_epoch, n_classes, alg)
    
    train_predictions = list()
    test_predictions = list()
    for x in train_x:
        prediction = alg['prediction'](coef, x)
        train_predictions.append(prediction)
    for x in test_x:
        prediction = alg['prediction'](coef, x)
        test_predictions.append(prediction)
    return train_predictions, test_predictions, loss

# Evaluate an algorithm using a cross validation split
def evaluate(x, y, n_folds, lr, n_epoch, alg):
    fold_size = int(len(y) / n_folds)
    train_accuracy = list()
    test_accuracy = list()
    losses = list()
    for i in range(n_folds):
        idx = random.sample(range(len(y)), fold_size)
        train_x = x[idx]
        train_y = y[idx]
        test_x = np.delete(x, idx, axis=0)
        test_y = np.delete(y, idx, axis=0)

        train_predictions, test_predictions, loss = evaluate_split(train_x, train_y, test_x, lr, n_epoch, alg)
        train_accuracy.append(accuracy_metric(train_y, train_predictions))     
        test_accuracy.append(accuracy_metric(test_y, test_predictions))
        losses.append(loss)
    return train_accuracy, test_accuracy, losses