import tensorflow as tf

class LinearClassifier(object):
    def __init__(self, x, y, num_features, num_classes, lr, reg):
        """ init the model with hyper-parameters etc """
        self.x = x
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes
        
        with tf.variable_scope("LinearClassifier"):
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            self.W = tf.Variable(initializer([num_features, num_classes]))
            self.b = tf.Variable(initializer([num_classes]))

            scores = tf.matmul(tf.layers.batch_normalization(self.x), self.W) + self.b
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=scores)) + reg * tf.nn.l2_loss(self.W)

            self.predictions = tf.argmax(scores, axis=1)        
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))

            self.metrics = self.loss, self.acc
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimize = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)