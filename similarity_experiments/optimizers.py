class SGDOptimizer(tf.train.Optimizer):
    def __init(self, lr, var_list=[]):
        self.lr = lr
        self.var_list = var_list
        
        
        
    return

class AMSGradOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, decay=False, beta1=0.9, beta2=0.99,
           epsilon=0.0, var_list=[]):
        self.learning_rate = learning_rate
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.var_list = var_list
        self.m = {}
        self.v = {}
        self.v_hat = {}
        self.t = tf.Variable(0.0, trainable=False)

        for var in self.var_list:
            self.m[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
            self.v[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
            self.v_hat[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)

    def apply_gradients(self, gradient_variables):
        with tf.control_dependencies([self.t.assign_add(1.0)]):
            learning_rate = self.learning_rate
        if self.decay:
            learning_rate /= tf.sqrt(self.t)
        update_ops = []

        for (g, var) in gradient_variables:
            m = self.m[var].assign(self.beta1 * self.m[var] + (1 - self.beta1) * g)
            v = self.v[var].assign(self.beta2 * self.v[var] + (1 - self.beta2) * g * g)
            v_hat = self.v_hat[var].assign(tf.maximum(self.v_hat[var], v))

            update = -learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
            update_ops.append(var.assign_add(update))

        return tf.group(*update_ops)