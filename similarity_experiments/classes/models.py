import tensorflow as tf

def get_scope_variable(scope, var, shape=None, initializer=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        v = tf.get_variable(var, shape=shape, initializer=initializer)
    return v

# no batch normalization of features, softmax cross entropy
class LinearClassifier0(object):
    def __init__(self, features, y, training, num_features, num_classes, lr, reg, scope=""):
        """ init the model with hyper-parameters etc """
        self.features = features
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.W = get_scope_variable(scope=scope, var="W", shape=[num_features, num_classes], initializer=initializer)
        self.b = get_scope_variable(scope=scope, var="b", shape=[num_classes], initializer=initializer)

        scores = tf.matmul(self.features, self.W) + self.b
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=scores)) + reg * tf.nn.l2_loss(self.W)

        self.predictions = tf.argmax(scores, axis=1)        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.incorrect = tf.not_equal(self.predictions, self.y)

        self.metrics = self.loss, self.acc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
            

# batch normalization of features, softmax cross entropy
class LinearClassifier1(object):
    def __init__(self, features, y, training, num_features, num_classes, lr, reg, scope=""):
        """ init the model with hyper-parameters etc """
        self.features = features
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.W = get_scope_variable(scope=scope, var="W", shape=[num_features, num_classes], initializer=initializer)
        self.b = get_scope_variable(scope=scope, var="b", shape=[num_classes], initializer=initializer)

        scores = tf.matmul(tf.layers.batch_normalization(self.features, training=training), self.W) + self.b
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=scores)) + reg * tf.nn.l2_loss(self.W)

        self.predictions = tf.argmax(scores, axis=1)        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.incorrect = tf.not_equal(self.predictions, self.y)

        self.metrics = self.loss, self.acc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)        

# no batch normalization of features, logsistic loss
class LinearClassifier2(object):
    def __init__(self, features, y, training, num_features, num_classes, lr, reg, scope=""):
        """ init the model with hyper-parameters etc """
        self.features = features
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.W = get_scope_variable(scope=scope, var="W", shape=[num_features, num_classes], initializer=initializer)
        self.b = get_scope_variable(scope=scope, var="b", shape=[num_classes], initializer=initializer)

        scores = tf.matmul(self.features, self.W) + self.b
        self.loss = tf.losses.log_loss(tf.one_hot(self.y, num_classes), tf.nn.softmax(scores)) + reg * tf.nn.l2_loss(self.W)

        self.predictions = tf.argmax(scores, axis=1)        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.incorrect = tf.not_equal(self.predictions, self.y)

        self.metrics = self.loss, self.acc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

# batch normalization of features, logsistic loss
class LinearClassifier3(object):
    def __init__(self, features, y, training, num_features, num_classes, lr, reg, scope=""):
        """ init the model with hyper-parameters etc """
        self.features = features
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.W = get_scope_variable(scope=scope, var="W", shape=[num_features, num_classes], initializer=initializer)
        self.b = get_scope_variable(scope=scope, var="b", shape=[num_classes], initializer=initializer)

        scores = tf.matmul(tf.layers.batch_normalization(self.features, training=training), self.W) + self.b
        self.loss = tf.losses.log_loss(tf.one_hot(self.y, num_classes), tf.nn.softmax(scores)) + reg * tf.nn.l2_loss(self.W)

        self.predictions = tf.argmax(scores, axis=1)        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.incorrect = tf.not_equal(self.predictions, self.y)

        self.metrics = self.loss, self.acc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
            
class TwoLayerClassifier0(object):
    def __init__(self, features, y, training, num_features, num_classes, lr, reg, scope=""):
        """ init the model with hyper-parameters etc """
        self.features = features
        self.y = y
        self.training = training
        self.num_features = num_features
        self.num_classes = num_classes
        
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.W1 = get_scope_variable(scope=scope, var="W1", shape=[num_features, num_features], initializer=initializer)
        self.W2 = get_scope_variable(scope=scope, var="W2", shape=[num_features, num_classes], initializer=initializer)
        self.b1 = get_scope_variable(scope=scope, var="b1", shape=[num_features], initializer=initializer)
        self.b2 = get_scope_variable(scope=scope, var="b2", shape=[num_classes], initializer=initializer)

        scores = tf.matmul(tf.nn.relu(tf.matmul(self.features, self.W1) + self.b1), self.W2) + self.b2
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=scores)
                                  ) + reg * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2))

        self.predictions = tf.argmax(scores, axis=1)        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))
        self.incorrect = tf.not_equal(self.predictions, self.y)

        self.metrics = self.loss, self.acc
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(update_ops):
            self.optimize = tf.train.AdagradOptimizer(lr).minimize(self.loss)