import numpy as np
import matplotlib.pyplot as plt

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    return sum(actual == predicted) / float(len(actual)) * 100.0

# Plot results of training and print accuracy metrics
def disp_results(x, y, train_accuracy, test_accuracy, loss, accuracy_while_training, loss_string, baseline_f):
    plt.subplot(1, 2, 1)
    plt.plot(np.mean(loss, axis=0))
    plt.title(loss_string + " Loss Plot")
    plt.ylabel(loss_string + " Loss")
    plt.xlabel("Epoch Number")
    plt.xscale('log')
    
    plt.subplot(1, 2, 2)
    accuracy_while_training = np.mean(accuracy_while_training, axis=0)
    plt.plot(accuracy_while_training)
    plt.title(loss_string + " Loss Accuracy While Training Plot")
    plt.ylabel("Accuracy While Training")
    plt.xlabel("Epoch Number")
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    baseline = baseline_f()
    baseline.fit(x,y)
    y_pred = baseline.predict(x)
    
    print("Accuracy While Training for " + loss_string + " Loss: %.3f%%" % (accuracy_while_training[-1]*100))
    print("Mean Training Accuracy for " + loss_string + " Loss: %.3f%%" % np.mean(train_accuracy))
    print("Mean Testing Accuracy for " + loss_string + " Loss: %.3f%%" % np.mean(test_accuracy))
    print("Baseline accuracy for " + loss_string + " Loss: %.3f%%" % accuracy_metric(y, y_pred))