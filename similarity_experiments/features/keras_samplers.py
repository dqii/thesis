import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class PairSampler:
    
    def __init__(self, data, sample_size, flip=True):
        x, y = data
        datagen = ImageDataGenerator(        
            featurewise_center=True,
            featurewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=flip,
            vertical_flip=flip,
            channel_shift_range=0.1)
        datagen.fit(x)
        self.sampler = datagen.flow(x, y, batch_size=2*sample_size)
        
    def sample(self):
        return self.sampler.next()
    
class TripletSampler:
    
    def __init__(self, data, sample_size, flip=True):
        x, y = data
        self.num_classes = len(np.unique(y))
        datagen = ImageDataGenerator(        
            featurewise_center=True,
            featurewise_std_normalization=True,
            zca_whitening=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.05,
            zoom_range=0.1,
            horizontal_flip=flip,
            vertical_flip=flip,
            channel_shift_range=0.1)
        datagen.fit(x)        
        self.sampler = datagen.flow(x, y, batch_size=4*sample_size)
        self.sample_size = sample_size
        
    def sample(self):
        x, y = self.sampler.next()       
        x1, y1 = self.x[:self.sample_size], self.y[:self.sample_size]
        x_other, y_other = self.x[self.sample_size:], self.y[self.sample_size:]
        for c in range(self.num_classes):
            n = np.sum(y1 == c)
            idx_same = (x_other == c)
            idx_diff = (x_other != c)
            sample_same = np.random.random_integers(0,len(idx_same)-1,n)
            sample_diff = np.random.random_integers(0,len(idx_other)-1,n)
            x2 = idx_same[idx_same][sample_same]
            y2 = idx_same[idx_same][sample_same]
            x3 = x_other[idx_diff][sample_diff]
            y3 = y_other[idx_diff][sample_diff]
        sample_x = np.concatenate((x1, x2, x3))
        sample_y = np.concatenate((y1, y2, y3))
        return sample_x, sample_y