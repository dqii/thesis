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
    correct = np.zeros(n_examples*n_epoch)

    for epoch in range(n_epoch):

        shuffle = np.random.permutation(n_examples)
        train_x = train_x[shuffle]
        train_y = train_y[shuffle]

        for i in range(n_examples):
            x = train_x[i,:]
            y = np.zeros(n_classes)
            y[train_y[i]] = 1

            score = alg['score'](w, x)
            correct[epoch*n_examples + i] +=  (alg['predict'](w, x) == train_y[i])
            loss[epoch] += alg['loss'](y, score)
            gradient = alg['gradient'](x, y, score)

            w -= lr * gradient
    return w, loss/n_examples, np.cumsum(correct)/np.arange(1, n_examples*n_epoch + 1)

# Run SGD over the cross-validation split
def evaluate_split(train_x, train_y, test_x, lr, n_epoch, alg):
    n_classes = len(np.unique(train_y))
    coef, loss, accuracy_while_training = sgd(train_x, train_y, lr, n_epoch, n_classes, alg)

    predictions_trainset = list()
    predictions_testset = list()
    for x in train_x:
        prediction = alg['predict'](coef, x)
        predictions_trainset.append(prediction)
    for x in test_x:
        prediction = alg['predict'](coef, x)
        predictions_testset.append(prediction)
    return predictions_trainset, predictions_testset, loss, accuracy_while_training

# Evaluate an algorithm using a cross validation split
def evaluate(x, y, n_folds, lr, n_epoch, alg):
    fold_size = int(len(y) / n_folds)
    accuracy_trainset = list()
    accuracy_testset = list()
    losses = list()
    accuracy_while_training = list()
    for i in range(n_folds):
        idx = random.sample(range(len(y)), fold_size)
        test_x = x[idx]
        test_y = y[idx]
        train_x = np.delete(x, idx, axis=0)
        train_y = np.delete(y, idx, axis=0)

        predictions_trainset, predictions_testset, loss, awt = evaluate_split(train_x, train_y, test_x, lr, n_epoch, alg)
        accuracy_trainset.append(accuracy_metric(train_y, predictions_trainset))
        accuracy_testset.append(accuracy_metric(test_y, predictions_testset))
        losses.append(loss)
        accuracy_while_training.append(awt)
    return accuracy_trainset, accuracy_testset, losses, accuracy_while_training