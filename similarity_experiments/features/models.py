from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from .spatial_transformer.spatial_transformer import transformer
import tensorflow as tf
import numpy as np

# --------------- variable helper functions

def get_parameters(initializer, var_shape_pairs):
    p = {}
    for v,s in var_shape_pairs:
        p[v] = tf.get_variable(v, shape=s, initializer=initializer)
    return p

# ---------------- convolution helper functions

def conv2d(x, w, b, s, training, dropout):
    conv = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME'),b)
    bn = tf.layers.batch_normalization(conv, training=training)
    relu = tf.nn.relu(bn)
    return tf.nn.dropout(relu, dropout)

def gcnn2d(input, w, b, s, training, dropout, gconv_indices, gconv_shape_info):
    conv = gconv2d(input=input, filter=w, strides=[1, s, s, 1], padding='SAME',
                   gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    conv_with_bias = tf.nn.bias_add(conv, b)
    bn = tf.layers.batch_normalization(conv, training=training)
    relu = tf.nn.relu(bn)
    return tf.nn.dropout(relu, dropout)
    
def max_pool(l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def fc_batch_relu(x, W, b, training, dropout):
    fc = tf.matmul(x, W) + b
    batch = tf.layers.batch_normalization(fc, training=training)
    relu = tf.nn.relu(batch)
    return tf.nn.dropout(relu, dropout)

def reg_loss_fn(W):
    return tf.nn.l2_loss(W['wd1'])

def resnet_conv2d(x, w1, w2, b1, b2, s1, s2, training, dropout):
    l1 = conv2d(x, w1, b1, s1, training, dropout)
    l2 = conv2d(l1, w2, b2, s2, training, dropout)
    return l2 + x

def resnet_gcnn2d(x, w1, w2, b1, b2, s1, s2, gi1, gi2, gs1, gs2, training, dropout):
    l1 = gcnn2d(x, w1, b1, s1, training, dropout, gi1, gs1)
    l2 = gcnn2d(l1, w2, b2, s2, training, dropout, gi2, gs2)
    return l2 + x 

# ---------------- spatial transformer network

def spatial_transformer_network(x, data_format, keep_prob):
    H, W, C = data_format
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)

    x_reshape = tf.reshape(x, [-1, H*W*C])

    W_fc_loc1 = tf.get_variable("W_fc_loc1", shape=[H*W*C, 20], initializer=initializer)
    b_fc_loc1 = tf.get_variable("b_fc_loc1", shape=[20], initializer=initializer)

    W_fc_loc2 = tf.get_variable("W_fc_loc2", shape=[20, 6], initializer=initializer)
    initial = np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()
    b_fc_loc2 = tf.get_variable("b_fc_loc2", initializer=tf.constant(initial))

    h_fc_loc1 = tf.nn.tanh(tf.matmul(x_reshape, W_fc_loc1) + b_fc_loc1)
    #h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)

    out_size = (H, W)
    h_trans = transformer(x, h_fc_loc2, out_size)
    return h_trans

# ---------------- conv architectures

class CNNSmall0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 16]),
                        ("wd1", [4*4*16, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [16]),
                        ("bd1", [64]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*16])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out

class CNNSmall1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 16]),
                        ("wc4", [3, 3, 16, 16]),
                        ("wd1", [4*4*16, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [16]),
                        ("bc4", [16]),
                        ("bd1", [64]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*16])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out    
    
class CNNSmall2(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wd2", [256, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [32]),
                        ("bd1", [256]),
                        ("bd2", [64]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out    
    
class CNNMedium0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 32]),
                        ("wc2", [3, 3, 32, 32]),
                        ("wc3", [3, 3, 32, 32]),
                        ("wc4", [3, 3, 32, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [32]),
                        ("bc2", [32]),
                        ("bc3", [32]),
                        ("bc4", [32]),
                        ("bd1", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out
    
class CNNMedium1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 32]),
                        ("wc2", [3, 3, 32, 32]),
                        ("wc3", [3, 3, 32, 32]),
                        ("wc4", [3, 3, 32, 32]),
                        ("wc5", [3, 3, 32, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [32]),
                        ("bc2", [32]),
                        ("bc3", [32]),
                        ("bc4", [32]),
                        ("bc5", [32]),
                        ("bd1", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)
        conv5 = conv2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out
    
class CNNMedium2(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 32]),
                        ("wc3", [3, 3, 32, 64]),
                        ("wc4", [3, 3, 64, 128]),
                        ("wd1", [4*4*128, 512]),
                        ("wd2", [512, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [32]),
                        ("bc3", [64]),
                        ("bc4", [128]),
                        ("bd1", [512]),
                        ("bd2", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*128])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out
        
class CNNLarge0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 32]),
                        ("wc3", [3, 3, 32, 64]),
                        ("wc4", [3, 3, 64, 128]),
                        ("wc5", [3, 3, 128, 128]),
                        ("wd1", [4*4*128, 1028]),
                        ("wd2", [1028, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [32]),
                        ("bc3", [64]),
                        ("bc4", [128]),
                        ("bc5", [128]),
                        ("bd1", [1028]),
                        ("bd2", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 1, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)
        conv5 = conv2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv5, [-1, 4*4*128])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out
    
class CNNLarge1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 32]),
                        ("wc2", [3, 3, 32, 128]),
                        ("wc3", [3, 3, 128, 128]),
                        ("wc4", [3, 3, 128, 128]),
                        ("wc5", [3, 3, 128, 128]),
                        ("wd1", [4*4*128, 1028]),
                        ("wd2", [1028, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [32]),
                        ("bc2", [128]),
                        ("bc3", [128]),
                        ("bc4", [128]),
                        ("bc5", [128]),
                        ("bd1", [1028]),
                        ("bd2", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 1, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)
        conv5 = conv2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv5, [-1, 4*4*128])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out
    
# ---------------- group conv architectures

class GCNNSmall0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)        
            self.gconv_indices1, self.gconv_shape_info1, w_shape1 = gconv2d_util(
                h_input='Z2', h_output='D4', in_channels=data_format[2], out_channels=16, ksize=3)
            self.gconv_indices2, self.gconv_shape_info2, w_shape2 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices3, self.gconv_shape_info3, w_shape3 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=32, ksize=3)

            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", w_shape1),
                        ("wc2", w_shape2),
                        ("wc3", w_shape3),
                        ("wd1", [4*4*256, 512]),
                        ("wd2", [512, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [128]),
                        ("bc2", [128]),
                        ("bc3", [256]),
                        ("bd1", [512]),
                        ("bd2", [64]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d(self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout,
                       self.gconv_indices1, self.gconv_shape_info1)
        conv2 = gcnn2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout,
                       self.gconv_indices2, self.gconv_shape_info2)
        conv3 = gcnn2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices3, self.gconv_shape_info3)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*256])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out    
    
class GCNNMedium0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)        
            self.gconv_indices1, self.gconv_shape_info1, w_shape1 = gconv2d_util(
                h_input='Z2', h_output='D4', in_channels=data_format[2], out_channels=16, ksize=3)
            self.gconv_indices2, self.gconv_shape_info2, w_shape2 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices3, self.gconv_shape_info3, w_shape3 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=32, ksize=3)
            self.gconv_indices4, self.gconv_shape_info4, w_shape4 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=32, out_channels=32, ksize=3)

            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", w_shape1),
                        ("wc2", w_shape2),
                        ("wc3", w_shape3),
                        ("wc4", w_shape4),
                        ("wd1", [4*4*256, 1024]),
                        ("wd2", [1024, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [128]),
                        ("bc2", [128]),
                        ("bc3", [256]),
                        ("bc4", [256]),
                        ("bd1", [1024]),
                        ("bd2", [256]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d(self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout,
                       self.gconv_indices1, self.gconv_shape_info1)
        conv2 = gcnn2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout,
                       self.gconv_indices2, self.gconv_shape_info2)
        conv3 = gcnn2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices3, self.gconv_shape_info3)
        conv4 = gcnn2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout,
                       self.gconv_indices4, self.gconv_shape_info4)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*256])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out    
    
    
class GCNNMedium1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)                
            self.gconv_indices = {}
            self.gconv_shape_info = {}

            self.gconv_indices['1'], self.gconv_shape_info['1'], w_shape1 = gconv2d_util(
                h_input='Z2', h_output='D4', in_channels=data_format[2], out_channels=16, ksize=3)
            self.gconv_indices['2'], self.gconv_shape_info['2'], w_shape2 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices['3'], self.gconv_shape_info['3'], w_shape3 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices['4'], self.gconv_shape_info['4'], w_shape4 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices['5'], self.gconv_shape_info['5'], w_shape5 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)

            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", w_shape1),
                        ("wc2", w_shape2),
                        ("wc3", w_shape3),
                        ("wc4", w_shape4),
                        ("wc5", w_shape5),
                        ("wd1", [4*4*128, 512]),
                        ("wd2", [512, 128]),
                        ("wout", [128, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [128]),
                        ("bc2", [128]),
                        ("bc3", [128]),
                        ("bc4", [128]),
                        ("bc5", [128]),
                        ("bd1", [512]),
                        ("bd2", [128]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout,
                       self.gconv_indices['1'], self.gconv_shape_info['1'])
        conv2 = gcnn2d(conv1, self.weights['wc2'], self.biases['bc2'], 1, self.training, self.dropout,
                       self.gconv_indices['2'], self.gconv_shape_info['2'])
        conv3 = gcnn2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices['3'], self.gconv_shape_info['3'])
        conv4 = gcnn2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout,
                       self.gconv_indices['4'], self.gconv_shape_info['4'])
        conv5 = gcnn2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout,
                       self.gconv_indices['5'], self.gconv_shape_info['5'])

        dense1_reshape = tf.reshape(conv5, [-1, 4*4*128])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out 
        
class GCNNLarge0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)                
            self.gconv_indices = {}
            self.gconv_shape_info = {}

            self.gconv_indices['1'], self.gconv_shape_info['1'], w_shape1 = gconv2d_util(
                h_input='Z2', h_output='D4', in_channels=data_format[2], out_channels=16, ksize=3)
            self.gconv_indices['2'], self.gconv_shape_info['2'], w_shape2 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
            self.gconv_indices['3'], self.gconv_shape_info['3'], w_shape3 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=16, out_channels=32, ksize=3)
            self.gconv_indices['4'], self.gconv_shape_info['4'], w_shape4 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=32, out_channels=32, ksize=3)
            self.gconv_indices['5'], self.gconv_shape_info['5'], w_shape5 = gconv2d_util(
                h_input='D4', h_output='D4', in_channels=32, out_channels=32, ksize=3)

            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", w_shape1),
                        ("wc2", w_shape2),
                        ("wc3", w_shape3),
                        ("wc4", w_shape4),
                        ("wc5", w_shape5),
                        ("wd1", [4*4*256, 512]),
                        ("wd2", [512, 128]),
                        ("wout", [128, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [128]),
                        ("bc2", [128]),
                        ("bc3", [256]),
                        ("bc4", [256]),
                        ("bc5", [256]),
                        ("bd1", [512]),
                        ("bd2", [128]),
                        ("bout", [num_features])])

            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d(self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout,
                       self.gconv_indices['1'], self.gconv_shape_info['1'])
        conv2 = gcnn2d(conv1, self.weights['wc2'], self.biases['bc2'], 1, self.training, self.dropout,
                       self.gconv_indices['2'], self.gconv_shape_info['2'])
        conv3 = gcnn2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices['3'], self.gconv_shape_info['3'])
        conv4 = gcnn2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout,
                       self.gconv_indices['4'], self.gconv_shape_info['4'])
        conv5 = gcnn2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout,
                       self.gconv_indices['5'], self.gconv_shape_info['5'])

        dense1_reshape = tf.reshape(conv5, [-1, 4*4*256])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out 
    
# ---------------- resnet architectures    

# ---------------- stn architectures

class STNCNNSmall0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 16]),
                        ("wd1", [4*4*16, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [16]),
                        ("bd1", [64]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*16])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out

class STNCNNSmall1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 16]),
                        ("wc4", [3, 3, 16, 16]),
                        ("wd1", [4*4*16, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [16]),
                        ("bc4", [16]),
                        ("bd1", [64]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*16])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out    
    
class STNCNNSmall2(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 16]),
                        ("wc3", [3, 3, 16, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wd2", [256, 64]),
                        ("wout", [64, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [16]),
                        ("bc3", [32]),
                        ("bd1", [256]),
                        ("bd2", [64]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv3, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out    
    
class STNCNNMedium0(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 32]),
                        ("wc2", [3, 3, 32, 32]),
                        ("wc3", [3, 3, 32, 32]),
                        ("wc4", [3, 3, 32, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [32]),
                        ("bc2", [32]),
                        ("bc3", [32]),
                        ("bc4", [32]),
                        ("bd1", [256]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out
    
class STNCNNMedium1(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 32]),
                        ("wc2", [3, 3, 32, 32]),
                        ("wc3", [3, 3, 32, 32]),
                        ("wc4", [3, 3, 32, 32]),
                        ("wc5", [3, 3, 32, 32]),
                        ("wd1", [4*4*32, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [32]),
                        ("bc2", [32]),
                        ("bc3", [32]),
                        ("bc4", [32]),
                        ("bc5", [32]),
                        ("bd1", [256]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)
        conv5 = conv2d(conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*32])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)

        out = tf.matmul(dense1, self.weights['wout']) + self.biases['bout']
        return out
    
class STNCNNMedium2(object):
    def __init__(self, x, y, settings, data_format, num_features, lr, reg, dropout, training, scope):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.weights = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("wc1", [3, 3, data_format[2], 16]),
                        ("wc2", [3, 3, 16, 32]),
                        ("wc3", [3, 3, 32, 64]),
                        ("wc4", [3, 3, 64, 128]),
                        ("wd1", [4*4*128, 512]),
                        ("wd2", [512, 256]),
                        ("wout", [256, num_features])])
            self.biases = get_parameters(
                    initializer=initializer, 
                    var_shape_pairs = [
                        ("bc1", [16]),
                        ("bc2", [32]),
                        ("bc3", [64]),
                        ("bc4", [128]),
                        ("bd1", [512]),
                        ("bd2", [256]),
                        ("bout", [num_features])])

            self.stn = spatial_transformer_network(self.x, data_format, dropout)
            self.features = self.feature_model()
            self.acc = settings.acc(self.features, self.y)
            self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d(self.stn, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, 4*4*128])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['wout']) + self.biases['bout']
        return out