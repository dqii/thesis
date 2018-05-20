import numpy as np
import tensorflow as tf

# ====== SETTINGS CLASS ======
# Settings class represents settings used for training feature extractor. """
class Settings:
    def __init__(self, loss_fn, acc_fn):
        self.loss = loss_fn
        self.acc = acc_fn

# ====== BASELINE FUNCTIONS =======
# for verifying that the architecture can do basic classification

def baseline_loss(features, classes):
    return tf.losses.sparse_softmax_cross_entropy(classes, features)
    
def baseline_acc(features, classes):
    predictions = tf.argmax(features, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, classes), tf.float32))

Baseline_Settings = Settings(baseline_loss, baseline_acc)

# ====== PAIRWISE FUNCTIONS =======
# pairs of data points
    
def pair_split(features, classes):
    f1, f2 = tf.split(features, 2)
    c1, c2 = tf.split(classes, 2)
    return f1, f2, c1, c2

def pair_hinge_loss(features, classes):
    f1, f2, c1, c2 = pair_split(features, classes)
    inner_products = tf.reduce_sum(tf.multiply(f1, f2), axis=1)
    similarities = tf.sign(tf.cast(tf.equal(c1, c2), tf.float32) - 0.5)
    scores = tf.multiply(similarities, inner_products)
    return tf.reduce_mean(tf.maximum(1.0 - scores, 0))
    
def pair_log_loss(features, classes):
    f1, f2, c1, c2 = pair_split(features, classes)
    inner_products = tf.reduce_sum(tf.multiply(f1, f2), axis=1)
    similarities = tf.sign(tf.cast(tf.equal(c1, c2), tf.float32) - 0.5)
    scores = tf.multiply(similarities, inner_products)
    return tf.reduce_mean(tf.log1p(-scores))

def pair_acc(features, classes):
    f1, f2, c1, c2 = pair_split(features, classes)
    predictions = tf.sign(tf.reduce_sum(tf.multiply(f1, f2), axis=1))
    similarities = tf.sign(tf.cast(tf.equal(c1, c2), tf.float32) - 0.5)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, similarities), tf.float32))

Pair_Hinge_Settings = Settings(pair_hinge_loss, pair_acc)
Pair_Log_Settings = Settings(pair_log_loss, pair_acc)

# ====== TRIPLES FUNCTIONS =======
# triples of data points

def triplet_hinge_loss(features, classes):
    f1, f2, f3 = tf.split(features, 3)
    inner_products = tf.reduce_sum(tf.multiply(f1, f2 - f3), axis=1)
    return tf.reduce_mean(tf.maximum(1.0 - inner_products, 0))

def triplet_log_loss(features, classes):
    f1, f2, f3 = tf.split(features, 3)
    inner_products = tf.reduce_sum(tf.multiply(f1, f2 - f3), axis=1)
    return tf.reduce_mean(tf.log1p(-inner_products))

def triplet_acc(features, classes):
    f1, f2, f3 = tf.split(features, 3)
    inner_products = tf.reduce_sum(tf.multiply(f1, f2 - f3), axis=1)    
    return tf.reduce_mean(tf.cast(tf.greater(inner_products, 0), tf.float32))

Triplet_Hinge_Settings = Settings(triplet_hinge_loss, triplet_acc)
Triplet_Log_Settings = Settings(triplet_log_loss, triplet_acc)

# ===== TRIPLES DISTANCE FUNCTION =====

def triplet_hinge_loss_distance(features, classes):
    f1, f2, f3 = tf.split(features, 3)
    f1_to_f2 = tf.reduce_sum(tf.square(f1 - f2),axis=1)
    f1_to_f3 = tf.reduce_sum(tf.square(f1 - f3),axis=1)
    return tf.reduce_mean(tf.maximum(1.0 - f1_to_f3 + f1_to_f2, 0))

def triplet_acc_distance(features, classes):
    f1, f2, f3 = tf.split(features, 3)
    f1_to_f2 = tf.reduce_sum(tf.square(f1 - f2),axis=1)
    f1_to_f3 = tf.reduce_sum(tf.square(f1 - f3),axis=1)
    return tf.reduce_mean(tf.cast(tf.greater(f1_to_f3, f1_to_f2), tf.float32))

Triplet_Distance_Settings = Settings(triplet_hinge_loss_distance, triplet_acc_distance)