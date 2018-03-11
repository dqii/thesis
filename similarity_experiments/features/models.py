from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
import tensorflow as tf

def conv2d(name, input, w, b, s, training, dropout):
    conv = tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, s, s, 1], padding='SAME'),b)
    bn = tf.layers.batch_normalization(conv, training=training)
    relu = tf.nn.relu(bn)
    return tf.nn.dropout(relu, dropout)

def gcnn2d(name, input, w, b, s, training, dropout, gconv_indices, gconv_shape_info):
    conv = gconv2d(input=input, filter=w, strides=[1, s, s, 1], padding='SAME',
                   gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    conv_with_bias = tf.nn.bias_add(conv, b)
    bn = tf.layers.batch_normalization(conv, training=training)
    relu = tf.nn.relu(bn)
    return tf.nn.dropout(relu, dropout)
    

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def fc_batch_relu(x, W, b, training, dropout):
    fc = tf.matmul(x, W) + b
    batch = tf.layers.batch_normalization(fc, training=training)
    relu = tf.nn.relu(batch)
    return tf.nn.dropout(relu, dropout)

def reg_loss_fn(W):
    return tf.nn.l2_loss(W['wd1']) + tf.nn.l2_loss(W['wd2']) + tf.nn.l2_loss(W['out'])

class ConvModelSmall(object):
    def __init__(self, x, y, settings, num_chan, num_features, lr, reg, dropout, training):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, num_chan, 16])),
            'wc2': tf.Variable(initializer([3, 3, 16, 16])),
            'wc3': tf.Variable(initializer([3, 3, 16, 32])),
            'wd1': tf.Variable(initializer([4*4*32, 256])),
            'wd2': tf.Variable(initializer([256, 64])),
            'out': tf.Variable(initializer([64, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([16])),
            'bc2': tf.Variable(initializer([16])),
            'bc3': tf.Variable(initializer([32])),
            'bd1': tf.Variable(initializer([256])),
            'bd2': tf.Variable(initializer([64])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d('conv1', self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out
        
class ConvModelMedium(object):
    def __init__(self, x, y, settings, num_chan, num_features, lr, reg, dropout, training):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, num_chan, 16])),
            'wc2': tf.Variable(initializer([3, 3, 16, 32])),
            'wc3': tf.Variable(initializer([3, 3, 32, 64])),
            'wc4': tf.Variable(initializer([3, 3, 64, 128])),
            'wd1': tf.Variable(initializer([4*4*128, 512])),
            'wd2': tf.Variable(initializer([512, 256])),
            'out': tf.Variable(initializer([256, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([16])),
            'bc2': tf.Variable(initializer([32])),
            'bc3': tf.Variable(initializer([64])),
            'bc4': tf.Variable(initializer([128])),
            'bd1': tf.Variable(initializer([512])),
            'bd2': tf.Variable(initializer([256])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d('conv1', self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout)
        conv3 = conv2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d('conv4', conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv4, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out
    
    
class ConvModelLarge(object):
    def __init__(self, x, y, settings, num_chan, num_features, lr, reg, dropout, training):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout    
        self.training = training
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, num_chan, 16])),
            'wc2': tf.Variable(initializer([3, 3, 16, 32])),
            'wc3': tf.Variable(initializer([3, 3, 32, 64])),
            'wc4': tf.Variable(initializer([3, 3, 64, 128])),
            'wc5': tf.Variable(initializer([3, 3, 128, 128])),
            'wd1': tf.Variable(initializer([4*4*128, 1028])),
            'wd2': tf.Variable(initializer([1028, 256])),
            'out': tf.Variable(initializer([256, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([16])),
            'bc2': tf.Variable(initializer([32])),
            'bc3': tf.Variable(initializer([64])),
            'bc4': tf.Variable(initializer([128])),
            'bc5': tf.Variable(initializer([128])),
            'bd1': tf.Variable(initializer([1028])),
            'bd2': tf.Variable(initializer([256])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = conv2d('conv1', self.x, self.weights['wc1'], self.biases['bc1'], 1, self.training, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 1, self.training, self.dropout)
        conv3 = conv2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout)
        conv4 = conv2d('conv4', conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout)
        conv5 = conv2d('conv5', conv4, self.weights['wc5'], self.biases['bc5'], 2, self.training, self.dropout)

        dense1_reshape = tf.reshape(conv5, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out
    
########################

class GCNNModelSmall(object):
    def __init__(self, x, y, settings, num_chan, num_features, lr, reg, dropout, training):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)        
        
        self.gconv_indices1, self.gconv_shape_info1, w_shape1 = gconv2d_util(
            h_input='Z2', h_output='D4', in_channels=num_chan, out_channels=16, ksize=3)
        self.gconv_indices2, self.gconv_shape_info2, w_shape2 = gconv2d_util(
            h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
        self.gconv_indices3, self.gconv_shape_info3, w_shape3 = gconv2d_util(
            h_input='D4', h_output='D4', in_channels=16, out_channels=32, ksize=3)
        
        self.weights = {
            'wc1': tf.Variable(initializer(w_shape1)),
            'wc2': tf.Variable(initializer(w_shape2)),
            'wc3': tf.Variable(initializer(w_shape3)),
            'wd1': tf.Variable(initializer([4*4*256, 512])),
            'wd2': tf.Variable(initializer([512, 64])),
            'out': tf.Variable(initializer([64, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([128])),
            'bc2': tf.Variable(initializer([128])),
            'bc3': tf.Variable(initializer([256])),
            'bd1': tf.Variable(initializer([512])),
            'bd2': tf.Variable(initializer([64])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d('conv1', self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout,
                       self.gconv_indices1, self.gconv_shape_info1)
        conv2 = gcnn2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout,
                       self.gconv_indices2, self.gconv_shape_info2)
        conv3 = gcnn2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices3, self.gconv_shape_info3)

        dense1_reshape = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out    
    
class GCNNModelMedium(object):
    def __init__(self, x, y, settings, num_chan, num_features, lr, reg, dropout, training):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.dropout = dropout
        self.training = training
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.gconv_indices1, self.gconv_shape_info1, w_shape1 = gconv2d_util(
            h_input='Z2', h_output='D4', in_channels=num_chan, out_channels=16, ksize=3)
        self.gconv_indices2, self.gconv_shape_info2, w_shape2 = gconv2d_util(
            h_input='D4', h_output='D4', in_channels=16, out_channels=16, ksize=3)
        self.gconv_indices3, self.gconv_shape_info3, w_shape3 = gconv2d_util(
            h_input='D4', h_output='D4', in_channels=16, out_channels=32, ksize=3)
        self.gconv_indices4, self.gconv_shape_info4, w_shape4 = gconv2d_util(
            h_input='D4', h_output='D4', in_channels=32, out_channels=64, ksize=3)
        
        self.weights = {
            'wc1': tf.Variable(initializer(w_shape1)),
            'wc2': tf.Variable(initializer(w_shape2)),
            'wc3': tf.Variable(initializer(w_shape3)),
            'wc4': tf.Variable(initializer(w_shape4)),
            'wd1': tf.Variable(initializer([4*4*512, 1024])),
            'wd2': tf.Variable(initializer([1024, 256])),
            'out': tf.Variable(initializer([256, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([128])),
            'bc2': tf.Variable(initializer([128])),
            'bc3': tf.Variable(initializer([256])),
            'bc4': tf.Variable(initializer([512])),
            'bd1': tf.Variable(initializer([1024])),
            'bd2': tf.Variable(initializer([256])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        conv1 = gcnn2d('conv1', self.x, self.weights['wc1'], self.biases['bc1'], 2, self.training, self.dropout,
                       self.gconv_indices1, self.gconv_shape_info1)
        conv2 = gcnn2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2, self.training, self.dropout,
                       self.gconv_indices2, self.gconv_shape_info2)
        conv3 = gcnn2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2, self.training, self.dropout,
                       self.gconv_indices3, self.gconv_shape_info3)
        conv4 = gcnn2d('conv4', conv3, self.weights['wc4'], self.biases['bc4'], 2, self.training, self.dropout,
                       self.gconv_indices4, self.gconv_shape_info4)

        dense1_reshape = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1_reshape, self.weights['wd1'], self.biases['bd1'], self.training, self.dropout)
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'], self.training, self.dropout)

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out    