import tensorflow as tf
import time

def get_feature_model(feature_model, settings, num_features, data_format, f_lr, f_reg, f_scope):
    H, W, C = data_format
    x = tf.placeholder(tf.float32, shape=[None, H, W, C])
    y = tf.placeholder(tf.int64, shape=[None])
    dropout = tf.placeholder(tf.float32)
    f_training = tf.placeholder(tf.bool)
    f_model = feature_model(x=x, y=y, settings=settings, data_format=data_format, num_features=num_features, lr=f_lr, reg=f_reg, dropout=dropout, training=f_training, scope=f_scope)
    return f_model, (x, y, dropout, f_training)


def get_classifier(classifier, num_features, num_classes, c_lr, c_reg, c_scope, f_params):
    _, y, _, _ = f_params
    features = tf.placeholder(tf.float32, shape=[None, num_features])
    c_training = tf.placeholder(tf.bool)
    c_model = classifier(features=features, y=y, training=c_training, 
                         num_features=num_features, num_classes=num_classes, lr=c_lr, reg=c_reg, scope=c_scope)
    return c_model, (features, c_training)


def train_features(sess, feature_model, samplers, num_steps, keep_prob, f_params):  
    
    x, y, dropout, training = f_params
    train, valid, _ = samplers
    
    print("begin training features")
    train_time = time.time()
    for step in range(num_steps):        
        x_, y_ = train.sample(300)
        sess.run(feature_model.optimize, feed_dict={x:x_, y:y_, dropout:keep_prob, training:True})
        if step % 200 == 0:
            train_loss, train_acc = sess.run([feature_model.loss, feature_model.acc], 
                                             feed_dict={x:x_, y:y_, dropout:1.0, training:False})
            x_, y_ = valid.sample(300)
            valid_loss, valid_acc = sess.run([feature_model.loss, feature_model.acc], 
                                             feed_dict={x:x_, y:y_, dropout:1.0, training:False})
            print("\tstep %d: train loss %g, train error %g, valid loss %g, valid error %g" %
                  (step, train_loss, 1 - train_acc, valid_loss, 1 - valid_acc))          
    train_time = time.time() - train_time
    print("end training features // time elapsed: %.4f s"%(train_time))

    eval_time = time.time()
    valid_error = 0
    for _ in range(1000):
        x_, y_ = valid.sample(300)
        valid_error += 1 - sess.run(feature_model.acc, feed_dict={x:x_, y:y_, dropout:1.0, training:False})
    eval_time = time.time() - eval_time
    print("validation set error: %.4f // time elapsed: %.4f s"%(valid_error/1000, eval_time))  
    
    
def train_features_coarse(sess, feature_model, samplers, num_steps, keep_prob, f_params):  
    
    x, y, dropout, training = f_params
    train, coarse, valid, _ = samplers
    
    print("begin training features")
    train_time = time.time()
    for step in range(num_steps):
        if step % 2 == 0:
            x_, y_ = train.sample(300)
        else:
            x_, y_ = coarse.sample(300)
        sess.run(feature_model.optimize, feed_dict={x:x_, y:y_, dropout:keep_prob, training:True})
        if step % 200 == 0:
            train_loss, train_acc = sess.run([feature_model.loss, feature_model.acc], 
                                             feed_dict={x:x_, y:y_, dropout:1.0, training:False})
            x_, y_ = valid.sample(300)
            valid_loss, valid_acc = sess.run([feature_model.loss, feature_model.acc], 
                                             feed_dict={x:x_, y:y_, dropout:1.0, training:False})
            print("\tstep %d: train loss %g, train error %g, valid loss %g, valid error %g" %
                  (step, train_loss, 1 - train_acc, valid_loss, 1 - valid_acc))          
    train_time = time.time() - train_time
    print("end training features // time elapsed: %.4f s"%(train_time))

    eval_time = time.time()
    valid_error = 0
    for _ in range(1000):
        x_, y_ = valid.sample(300)
        valid_error += 1 - sess.run(feature_model.acc, feed_dict={x:x_, y:y_, dropout:1.0, training:False})
    eval_time = time.time() - eval_time
    print("validation set error: %.4f // time elapsed: %.4f s"%(valid_error/1000, eval_time))  
    
    
def test_features(sess, feature_model, samplers, f_params):  
    
    x, y, dropout, training = f_params
    _, valid, test = samplers
    
    test_time = time.time()
    valid_err = 0
    test_err = 0
    for step in range(1000):
        x_, y_ = test.sample(200)
        _, test_acc = sess.run([feature_model.loss, feature_model.acc], 
                                         feed_dict={x:x_, y:y_, dropout:1.0, training:False})
        test_err += 1 - test_acc
    test_time = time.time() - test_time
    print("test set error: %.4f // time elapsed: %.4f s"%(test_err/1000, test_time/2))
        
        
def train_classifier(sess, feature_model, classifier, data, samplers, num_steps, f_params, c_params):
    
    (x, y, dropout, f_training), (features, c_training) = f_params, c_params
    train, valid, _ = samplers
    _, (x_valid, y_valid), _ = data
        
    print("begin training classifier")
    train_time = time.time()
    for step in range(num_steps):        
        x_, y_ = train.sample(700)
        features_ = sess.run(feature_model.features, feed_dict={x:x_, dropout:1.0, f_training:False}) 
        sess.run(classifier.optimize, feed_dict={features:features_, y:y_, dropout:1.0, c_training:True})     
        if step % 100 == 0:
            train_loss, train_acc = sess.run([classifier.loss, classifier.acc], 
                                             feed_dict={features:features_, y:y_, c_training:False})
            x_, y_ = valid.sample(700)
            features_ = sess.run(feature_model.features, feed_dict={x:x_, dropout:1.0, f_training:False})
            valid_loss, valid_acc = sess.run([classifier.loss, classifier.acc], 
                                             feed_dict={features:features_, y:y_, c_training:False})
            print("\tstep %d: train loss %g, train error %g"
                  %(step, train_loss, 1 - train_acc))             
    train_time = time.time() - train_time
    print("end training classifier // time elapsed: %.4f s"%(train_time))

    eval_time = time.time()
    valid_error = 0
    for i in range(int(y_valid.shape[0]/1000)):
        features_ = sess.run(feature_model.features, feed_dict={x:x_valid[i:(i+1000)], dropout:1.0, f_training:False})
        valid_error += 1 - sess.run(classifier.acc, feed_dict={features:features_, y:y_valid[i:(i+1000)], c_training:False})
    eval_time = time.time() - eval_time
    print("validation set error: %.4f // time elapsed: %.4f s"%(valid_error/(y_valid.shape[0]/1000), eval_time))

    
def test_model(sess, feature_model, classifier, samplers, data, f_params, c_params):
    
    (x, y, dropout, f_training), (features, c_training) = f_params, c_params
    train, valid, _ = samplers
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data    
    
    eval_time = time.time()
    train_error = 0
    for i in range(int(y_train.shape[0]/1000)):
        features_ = sess.run(feature_model.features, feed_dict={x:x_train[i:(i+1000)], dropout:1.0, f_training:False})
        train_error += 1 - sess.run(classifier.acc, feed_dict={features:features_, y:y_train[i:(i+1000)], c_training:False})
    eval_time = time.time() - eval_time
    print("train set error: %.4f // time elapsed: %.4f s"%(train_error/(y_train.shape[0]/1000), eval_time))

    eval_time = time.time()
    valid_error = 0
    for i in range(int(y_valid.shape[0]/1000)):
        features_ = sess.run(feature_model.features, feed_dict={x:x_valid[i:(i+1000)], dropout:1.0, f_training:False})
        valid_error += 1 - sess.run(classifier.acc, feed_dict={features:features_, y:y_valid[i:(i+1000)], c_training:False})
    eval_time = time.time() - eval_time
    print("validation set error: %.4f // time elapsed: %.4f s"%(valid_error/(y_valid.shape[0]/1000), eval_time))

    eval_time = time.time()
    test_error = 0
    for i in range(int(y_test.shape[0]/1000)):
        features_ = sess.run(feature_model.features, feed_dict={x:x_test[i:(i+1000)], dropout:1.0, f_training:False})
        test_error += 1 - sess.run(classifier.acc, feed_dict={features:features_, y:y_test[i:(i+1000)], c_training:False})
    eval_time = time.time() - eval_time
    print("test set error: %.4f // time elapsed: %.4f s"%(test_error/(y_test.shape[0]/1000), eval_time))