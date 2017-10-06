import numpy as np

# HINGE / MAX MARGIN
def get_hinge_prediction(w, x):
    score = get_hinge_score(w, x)
    return np.argmax(score)

def get_hinge_score(w, x):
    return np.dot(w, x)

def get_hinge_loss(y, p):
    y = y*2 - 1
    return np.sum(np.maximum(0, 1 - np.multiply(y,p)))

def get_hinge_gradient(x, y, p):
    y = y*2 - 1
    gradient = -np.outer(y, x)
    gradient[np.multiply(y,p) > 1] = 0
    return gradient

hinge = {'prediction': get_hinge_prediction, 'score': get_hinge_score,
         'loss': get_hinge_loss, 'gradient': get_hinge_gradient}

# LOGISTIC

def get_logistic_prediction(w, x):
    p = get_logistic_probs(w, x)
    return np.argmax(p)

def get_logistic_probs(w, x):
    exp_terms = np.exp(np.dot(w, x))
    return exp_terms / sum(exp_terms)

def get_log_loss(y, p):
    return -np.dot(y, np.log(p))

def get_log_loss_gradient(x, y, p):
    return -np.outer(y - p, x)

logistic = {'prediction': get_logistic_prediction, 'score': get_logistic_probs,
            'loss': get_log_loss, 'gradient': get_log_loss_gradient}

# SQUARE

# BINARY