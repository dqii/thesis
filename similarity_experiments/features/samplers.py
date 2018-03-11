import numpy as np

class PairSampler:
    
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
        end = np.minimum(start+2*s, self.N)
        data = (self.x[start:end], self.y[start:end].flatten())
        self.start += 2*s   
        if self.start >= self.N - 1:
            self.start = 0
            np.random.shuffle(self.shuffle)
            self.x = self.x[self.shuffle]
            self.y = self.y[self.shuffle]
        return data
    
    
class TripletSampler:
    
    def __init__(self, data):
        self.x, self.y = data
        self.N = self.y.shape[0]
        y_ind = self.y.argsort()        
        self.x = self.x[y_ind]
        self.y = self.y[y_ind]
        self.counts = np.bincount(self.y)
        self.first_index = np.roll(np.cumsum(self.counts),1)
        self.first_index[0] = 0
        
    def sample(self, s):
        idx1 = np.random.randint(low=0, high=self.N, size=s)
        x1, y1 = self.x[idx1], self.y[idx1]
        idx2 = np.floor(np.random.random_sample(s) * self.counts[y1] + self.first_index[y1]).astype(int)
        x2, y2 = self.x[idx2], self.y[idx2]
        x3, y3 = np.copy(x1), np.copy(y1)
        for i in range(s):
            idx = np.random.randint(low=0, high=self.N)
            while (self.y[idx] == y1[i]):
                idx = np.random.randint(low=0, high=self.N)
            x3[i] = self.x[idx]
            y3[i] = self.y[idx]
        sample_x = np.concatenate((x1, x2, x3))
        sample_y = np.concatenate((y1, y2, y3))
        return sample_x, sample_y