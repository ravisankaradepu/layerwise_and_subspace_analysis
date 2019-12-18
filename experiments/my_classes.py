import torch
from torch.utils import data
import numpy as np
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.list_IDs[index]

        # Load data and get label

        y = self.labels[index]
        X = X.astype(np.float)
        y = y.astype(np.float)
        X = torch.FloatTensor(X)
        if type(y) == np.float64:
                y = [y]        
        y = torch.FloatTensor(y)
        
        return X, y
