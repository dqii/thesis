import tensorflow as tf

def conv2d(name, l_input, w, b, s):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def fc_batch_relu(x, W, b):
    return tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x, W) + b))

def reg_loss_fn(W):
    return tf.nn.l2_loss(W['wd1']) + tf.nn.l2_loss(W['wd2']) + tf.nn.l2_loss(W['out'])

class ConvModelSmall(object):
    def __init__(self, x, y, num_features, settings, lr, reg, dropout):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.num_features = num_features
        self.dropout = dropout        
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, 1, 16])),
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
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr)
            
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        # Reshape input picture
        input = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        conv1 = conv2d('conv1', input, self.weights['wc1'], self.biases['bc1'], 2)
        conv1 = tf.nn.dropout(conv1, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2)
        conv2 = tf.nn.dropout(conv2, self.dropout)
        conv3 = conv2d('conv3', conv2, self.weights['wc3'], self.biases['bc3'], 2)
        conv3 = tf.nn.dropout(conv3, self.dropout)

        dense1 = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1, self.weights['wd1'], self.biases['bd1'])
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'])

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out
    
class ConvModelMedium(object):
    def __init__(self, x, y, num_features, settings, lr, reg, dropout):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.num_features = num_features
        self.dropout = dropout        
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, 1, 16])),
            'wc2': tf.Variable(initializer([3, 3, 16, 32])),
            'wd1': tf.Variable(initializer([7*7*32, 512])),
            'wd2': tf.Variable(initializer([512, 128])),
            'out': tf.Variable(initializer([128, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([16])),
            'bc2': tf.Variable(initializer([32])),
            'bd1': tf.Variable(initializer([512])),
            'bd2': tf.Variable(initializer([128])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr)
            
        self.features = self.feature_model()
        self.acc = settings.acc(self.features, self.y)
        self.loss = settings.loss(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        # Reshape input picture
        input = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        conv1 = conv2d('conv1', input, self.weights['wc1'], self.biases['bc1'], 2)
        conv1 = tf.nn.dropout(conv1, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 2)
        conv2 = tf.nn.dropout(conv2, self.dropout)
        dense1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1, self.weights['wd1'], self.biases['bd1'])
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'])

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out
    
class ConvModelLarge(object):
    def __init__(self, x, y, num_features, loss_fn, acc_fn, lr, reg, dropout):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.num_features = num_features
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.dropout = dropout        
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.weights = {
            'wc1': tf.Variable(initializer([3, 3, 1, 16])),
            'wc2': tf.Variable(initializer([3, 3, 16, 32])),
            'wc3': tf.Variable(initializer([3, 3, 32, 64])),
            'wd1': tf.Variable(initializer([7*7*64, 1024])),
            'wd2': tf.Variable(initializer([1024, 128])),
            'out': tf.Variable(initializer([128, num_features]))
        }
        self.biases = {
            'bc1': tf.Variable(initializer([16])),
            'bc2': tf.Variable(initializer([32])),
            'bc3': tf.Variable(initializer([64])),
            'bd1': tf.Variable(initializer([1024])),
            'bd2': tf.Variable(initializer([128])),
            'out': tf.Variable(initializer([num_features]))
        }
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr)
            
        self.features = self.feature_model()
        self.acc = acc_fn(self.features, self.y)
        self.loss = loss_fn(self.features, self.y) + reg * reg_loss_fn(self.weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def feature_model(self):
        # Reshape input picture
        input = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        conv1 = conv2d('conv1', input, self.weights['wc1'], self.biases['bc1'], 1)
        conv1 = tf.nn.dropout(conv1, self.dropout)
        conv2 = conv2d('conv2', conv1, self.weights['wc2'], self.biases['bc2'], 1)
        pool2 = max_pool('pool2', conv2, k=2)
        pool2 = tf.nn.dropout(pool2, self.dropout)
        conv3 = conv2d('conv3', pool2, self.weights['wc3'], self.biases['bc3'], 1)
        pool3 = max_pool('pool3', conv3, k=2)
        pool3 = tf.nn.dropout(pool3, self.dropout)

        dense1 = tf.reshape(pool3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        dense1 = fc_batch_relu(dense1, self.weights['wd1'], self.biases['bd1'])
        dense2 = fc_batch_relu(dense1, self.weights['wd2'], self.biases['bd2'])

        out = tf.matmul(dense2, self.weights['out']) + self.biases['out']
        return out