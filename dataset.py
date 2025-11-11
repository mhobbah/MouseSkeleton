import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Number of body parts
V = 10 

def masking(data, mask_length, mode, drop_rate, append):
    masked = np.empty((0, data.shape[1]))
    data_copy = data.copy(deep=True)
    
    for i in range(len(data)//(mask_length*V)):
        rng = np.random.default_rng(seed=42+append+i)
        seq = data_copy.iloc[i*mask_length*V:(i+1)*mask_length*V,:].values
        seq = seq.reshape(mask_length, V, -1)
    
        # Coordinates
        if mode == 0:
            mask = rng.random(seq.shape) < drop_rate
            seq[mask] = -1
    
        # Parts
        if mode == 1:
            mask_idx = rng.choice(np.arange(V), size=drop_rate, replace=False)
            seq[5:-5, mask_idx, :] = -1
    
        # Frames
        if mode == 2:
            length = drop_rate
            
            values = np.arange(0, mask_length - 2*length + 1)
            idx = rng.choice(values)
            
            seq[idx:idx+length, :, :] = -1
                
        seq = seq.reshape(-1, 3)
        masked = np.concatenate([masked, seq])

    return masked
            

class MouseTracks(Dataset):
    """ Mouse Tracking Dataset """
    def __init__(self, path, seq_length=100, stride=1, mode=0, drop_rate=0, norm=False, append=0):
        self.data = pd.read_csv(path)
        self.positions_d = self.data.iloc[:, 5:]
        self.seq_length = seq_length
        self.stride = stride
        self.mode = mode
        self.drop_rate = drop_rate

        # Translate to positive coordinates
        self.positions_d -= self.positions_d.min()

        # Normalize values to range [0, 1]
        if norm == True:
            self.maxima = self.positions_d.max()
            self.positions_d /= self.maxima

        
        self.masked = masking(self.positions_d, 100, self.mode, self.drop_rate, append)
            
        self.positions_d = self.positions_d.iloc[:len(self.masked), :]
        self.time = self.data["time"].iloc[:len(self.masked)]
        self.part = self.data["part"].iloc[:len(self.masked)]

    def __len__(self):
        return (len(self.positions_d) - self.seq_length * V) // (self.stride * V) + 1

    def __getitem__(self, i):
        idx_start = i * self.stride * V
        idx_end = idx_start + self.seq_length * V
        x = self.masked[idx_start:idx_end, :]
        y = self.positions_d.iloc[idx_start:idx_end, :].values
        x, y = x.reshape(self.seq_length, V, -1), y.reshape(self.seq_length, V, -1)
        x, y = torch.tensor(x).float(), torch.tensor(y).float()

        return x, y
