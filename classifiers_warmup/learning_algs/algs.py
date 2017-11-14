import numpy as np

# HINGE / MAX MARGIN (treating margin as a confidence)
def get_hinge_prediction(w, x):
    score = get_hinge_score(w, x)
    return np.argmax(score)

def get_hinge_score(w, x):
    return np.dot(w, x)

def get_hinge_loss(y, score):
    y = y*2 - 1
    return np.sum(np.maximum(0, 1 - np.multiply(y,score)))

def get_hinge_gradient(x, y, score):
    y = y*2 - 1
    gradient = -np.outer(y, x)
    gradient[np.multiply(y,score) > 1] = 0
    return gradient

hinge = {'predict': get_hinge_prediction, 'score': get_hinge_score,
         'loss': get_hinge_loss, 'gradient': get_hinge_gradient}

# LOGISTIC

def get_logistic_prediction(w, x):
    score = get_logistic_score(w, x)
    return np.argmax(score)

def get_logistic_score(w, x):
    exp_terms = np.exp(np.dot(w, x))
    return exp_terms / sum(exp_terms)

def get_log_loss(y, score):
    return -np.dot(y, np.log(score))

def get_log_loss_gradient(x, y, score):
    return -np.outer(y - score, x)

logistic = {'predict': get_logistic_prediction, 'score': get_logistic_score,
            'loss': get_log_loss, 'gradient': get_log_loss_gradient}

# SQUARE
def get_square_prediction(w, x):
    return

def get_square_score(w, x):
    return

def get_square_loss(y, p):
    return

def get_square_gradient(x, y, p):
    return

# BINARY
