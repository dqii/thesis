import numpy as np
import scipy
from classes.samplers import Sampler
from features.samplers import PairSampler, TripletSampler
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

def augment_data(x_train, y_train, flip=True):
    N = y_train.shape[0]
    x_augment, y_augment = np.zeros(x_train.shape), np.zeros(y_train.shape, dtype='int8')    
    for i in range(N):
        j = np.random.randint(low=0, high=N)
        x_augment[i], y_augment[i] = augment_img(x_train[j], flip), y_train[j]
    new_x = np.concatenate((x_train, x_augment))
    new_y = np.concatenate((y_train, y_augment))
    return new_x, new_y    

def load_datasets(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0
    if augment:
        x_train, y_train = augment_data(x_train, y_train, flip=False)
    data['mnist'] = (x_train, y_train), (x_test, y_test)
    samplers['mnist'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['mnist'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['mnist'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("MNIST loaded in %d seconds"%(time.time() - start_time))

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0 
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['fashion_mnist'] = (x_train, y_train), (x_test, y_test)
    samplers['fashion_mnist'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['fashion_mnist'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['fashion_mnist'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("Fashion MNIST loaded in %d seconds"%(time.time() - start_time))

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar10'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar10'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar10'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar10'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-10 loaded in %d seconds"%(time.time() - start_time))

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar100'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar100'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar100'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar100'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-100 loaded in %d seconds"%(time.time() - start_time))
    
    return data, pair_samplers, triplet_samplers

def load_mnist(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0 
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['mnist'] = (x_train, y_train), (x_test, y_test)
    samplers['mnist'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['mnist'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['mnist'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("MNIST loaded in %d seconds"%(time.time() - start_time))
    
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train.reshape(x_train.shape+(1,))/255.0, x_test.reshape(x_test.shape+(1,))/255.0 
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['fashion_mnist'] = (x_train, y_train), (x_test, y_test)
    samplers['fashion_mnist'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['fashion_mnist'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['fashion_mnist'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("Fashion MNIST loaded in %d seconds"%(time.time() - start_time))
    
    return data, samplers, pair_samplers, triplet_samplers


def load_cifar(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar10'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar10'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar10'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar10'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-10 loaded in %d seconds"%(time.time() - start_time))

    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar100'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar100'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar100'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar100'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-100 loaded in %d seconds"%(time.time() - start_time))
    
    return data, samplers, pair_samplers, triplet_samplers

def load_cifar10(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}
    
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar10'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar10'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar10'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar10'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-10 loaded in %d seconds"%(time.time() - start_time))
    
    return data, samplers, pair_samplers, triplet_samplers

def load_cifar100(augment=False):
    data, samplers, pair_samplers, triplet_samplers = {}, {}, {}, {}
    
    start_time = time.time()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    if augment:
        x_train, y_train = augment_data(x_train, y_train)
    data['cifar100'] = (x_train, y_train), (x_test, y_test)
    samplers['cifar100'] = Sampler((x_train, y_train)), Sampler((x_test, y_test))
    pair_samplers['cifar100'] = PairSampler((x_train, y_train)), PairSampler((x_test, y_test))
    triplet_samplers['cifar100'] = TripletSampler((x_train, y_train)), TripletSampler((x_test, y_test))
    print("CIFAR-100 loaded in %d seconds"%(time.time() - start_time))
    
    return data, samplers, pair_samplers, triplet_samplers