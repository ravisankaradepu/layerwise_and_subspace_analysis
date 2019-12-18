import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data

from my_classes import Dataset

df_red = pd.read_csv('wine-quality/winequality-red.csv', sep=';')
df_whi = pd.read_csv('wine-quality/winequality-white.csv', sep=';')
df_red["wine type"] = 'red'
df_whi["wine type"] = 'white'
df = pd.concat([df_red, df_whi], axis=0)
df = df.reset_index(drop=True)
df.head()
d = df.loc[:, ~df.columns.isin(['quality', 'wine type'])].values
target = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(d,
                                                    target,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)


# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets
partition = X_train
labels = y_train

# Generators
training_set = Dataset(partition, labels)
training_generator = data.DataLoader(training_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for l1,l2 in training_generator:
        print(l1,l2)  
