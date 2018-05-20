import numpy as np
import scipy
from classes.samplers import Sampler
from features.numpy_samplers import PairSampler, TripletSampler
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import tensorflow as tf
import time

def augment_img(img, flip=False):
    # rotate image
    angle = np.random.uniform(low=-25, high=25)
    new_img = scipy.ndimage.rotate(img, angle, reshape=False)
    # flip image
    if flip and np.random.uniform(low=0, high=1) > 0.5:
        new_img = np.flip(new_img, axis=1)
    # shift image
    shift1, shift2 = np.random.randint(low=-4, high=4), np.random.randint(low=-4, high=4)
    new_img = scipy.ndimage.shift(new_img, (shift1,shift2,0))
    return new_img

def augment_data(x_train, y_train, factor=1, flip=True):
    N = y_train.shape[0]
    x_augment, y_augment = np.zeros(x_train.shape), np.zeros(y_train.shape, dtype='int8')    
    for i in range(int(factor*N)):
        j = np.random.randint(low=0, high=N)
        x_augment[i], y_augment[i] = augment_img(x_train[j], flip), y_train[j]
    new_x = np.concatenate((x_train, x_augment))
    new_y = np.concatenate((y_train, y_augment))
    return new_x, new_y    

def get_samplers(train, valid, test):
    data = train, valid, test
    samplers = Sampler(train), Sampler(valid), Sampler(test)
    pair_samplers = PairSampler(train), PairSampler(valid), PairSampler(test)
    triplet_samplers = TripletSampler(train), TripletSampler(valid), TripletSampler(test)
    return data, samplers, pair_samplers, triplet_samplers

def get_samplers_coarse(train, coarse, valid, test):
    data = train, valid, test
    samplers = Sampler(train), Sampler(coarse), Sampler(valid), Sampler(test)
    pair_samplers = PairSampler(train), PairSampler(coarse), PairSampler(valid), PairSampler(test)
    triplet_samplers = TripletSampler(train), TripletSampler(coarse), TripletSampler(valid), TripletSampler(test)
    return data, samplers, pair_samplers, triplet_samplers

def load_mnist(augment=False):
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_valid = np.copy(x_train[idx[:5000]])
    y_valid = np.copy(y_train[idx[:5000]])
    x_train = np.copy(x_train[idx[5000:]])
    y_train = np.copy(y_train[idx[5000:]])
    if augment:
        x_train, y_train = augment_data(x_train, y_train, 0.5, flip=False)
    train, valid, test = (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    print("MNIST loaded in %d seconds"%(time.time() - start_time))
    return get_samplers(train, valid, test)

def load_fashion_mnist(augment=False):    
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0 
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_valid = np.copy(x_train[idx[:5000]])
    y_valid = np.copy(y_train[idx[:5000]])
    x_train = np.copy(x_train[idx[5000:]])
    y_train = np.copy(y_train[idx[5000:]])
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    train, valid, test = (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    print("Fashion MNIST loaded in %d seconds"%(time.time() - start_time))
    return get_samplers(train, valid, test)

def load_cifar10(augment=False):
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_valid = np.copy(x_train[idx[:5000]])
    y_valid = np.copy(y_train[idx[:5000]])
    x_train = np.copy(x_train[idx[5000:]])
    y_train = np.copy(y_train[idx[5000:]])
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    train, valid, test = (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    print("CIFAR-10 loaded in %d seconds"%(time.time() - start_time))
    return get_samplers(train, valid, test)

def load_cifar100(augment=False):
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_valid = np.copy(x_train[idx[:5000]])
    y_valid = np.copy(y_train[idx[:5000]])
    x_train = np.copy(x_train[idx[5000:]])
    y_train = np.copy(y_train[idx[5000:]]) 
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    train, valid, test = (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    print("CIFAR-100 loaded in %d seconds"%(time.time() - start_time))
    return get_samplers(train, valid, test)

def load_cifar100_coarse(augment=False):
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_valid = np.copy(x_train[idx[:5000]])
    y_valid = np.copy(y_train[idx[:5000]])
    x_train = np.copy(x_train[idx[5000:]])
    y_train = np.copy(y_train[idx[5000:]]) 
    
    # get coarse label dataset
    x_coarse = np.copy(x_train)
    y_coarse = np.copy(y_train)
    indices = []
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [4, 30, 55, 72, 95] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [1, 32, 67, 73, 91] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [54, 62, 70, 82, 92] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [9, 10, 16, 28, 61] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [0, 51, 53, 57, 83] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [22, 39, 40, 86, 87] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [5, 20, 25, 84, 94] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [6, 7, 14, 18, 24] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [3, 42, 43, 88, 97] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [12, 17, 37, 68, 76] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [23, 33, 49, 60, 71] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [15, 19, 21, 31, 38] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [34, 63, 64, 66, 75] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [26, 45, 77, 79, 99] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [2, 11, 35, 46, 98] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [27, 29, 44, 78, 93] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [36, 50, 65, 74, 80] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [47, 52, 56, 59, 96] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [8, 13, 48, 58, 90] ]))
    indices.append(np.array([idx for idx, y_val in enumerate(y_train) if y_val in [41, 69, 81, 85, 89] ]))
    for count, idx in enumerate(indices, 0):
        y_coarse[idx] = count        
    
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
        x_coarse, y_coarse = augment_data(x_coarse, y_coarse)
    train, coarse, valid, test = (x_train, y_train), (x_coarse, y_coarse), (x_valid, y_valid), (x_test, y_test)
    print("CIFAR-100 coarse loaded in %d seconds"%(time.time() - start_time))
    return get_samplers_coarse(train, coarse, valid, test)

def load_datasets(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}
    
    data['mnist'], samplers['mnist'], pair_samplers['mnist'], triplet_samplers['mnist'] = load_mnist(augment)
    data['fashion_mnist'], samplers['fashion_mnist'], pair_samplers['fashion_mnist'], triplet_samplers['fashion_mnist'] = load_fashion_mnist(augment)
    data['cifar10'], samplers['cifar10'], pair_samplers['cifar10'], triplet_samplers['cifar10'] = load_cifar10(augment)
    data['cifar100'], samplers['cifar100'], pair_samplers['cifar100'], triplet_samplers['cifar100'] = load_cifar100(augment)

    return data, samplers, pair_samplers, triplet_samplers

def load_mnist_datasets(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}
    
    data['mnist'], samplers['mnist'], pair_samplers['mnist'], triplet_samplers['mnist'] = load_mnist(augment)
    data['fashion_mnist'], samplers['fashion_mnist'], pair_samplers['fashion_mnist'], triplet_samplers['fashion_mnist'] = load_fashion_mnist(augment)

    return data, samplers, pair_samplers, triplet_samplers

def load_cifar_datasets(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}
    
    data['cifar10'], samplers['cifar10'], pair_samplers['cifar10'], triplet_samplers['cifar10'] = load_cifar10(augment)
    data['cifar100'], samplers['cifar100'], pair_samplers['cifar100'], triplet_samplers['cifar100'] = load_cifar100(augment)

    return data, samplers, pair_samplers, triplet_samplers