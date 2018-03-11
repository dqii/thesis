import numpy as np

class Sampler:
    
    def __init__(self, data):        
        self.x, self.y = data
        self.N = self.y.shape[0]
        self.start = 0
        self.shuffle = np.arange(self.N)
        np.random.shuffle(self.shuffle)
        self.x = self.x[self.shuffle]
        self.y = self.y[self.shuffle]
                
    def sample(self, s):
        start = self.start
        end = np.minimum(start+s, self.N)
        data = (self.x[start:end], self.y[start:end].flatten())
        self.start += s   
        if self.start >= self.N - 1:
            self.start = 0
            np.random.shuffle(self.shuffle)
            self.x = self.x[self.shuffle]
            self.y = self.y[self.shuffle]
        return data