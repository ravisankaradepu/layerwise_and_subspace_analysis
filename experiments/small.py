# THis is to test hessian penality on a small dataset
import os
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import sys
import os.path as osp
sys.path.append(osp.dirname(os.getcwd()))
from hessian import FullHessian
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from my_classes import Dataset
from torch.utils import data

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-d', '--dataset', type=str, choices=['wine','breast_cancer','diabetic','ionospherre'], help='dataset to analyze' )
    parser.add_argument('--cuda', type=int, help='number of gpu to be used, default=None')    
    parser.add_argument('-n',
                        '--num_epochs',
                        type=int,
                        default=100,
                        help='number of training epochs')
                        
    parser.add_argument('-sp','--scaling_epoch',type=int,default=1000,help='after what epoch scaling comes into action')
    parser.add_argument('--abs',
                        action='store_true',
                        help='peanlize sum of absolute(H_{ii})')
                        
    parser.add_argument('--sf',
                        default = 1e-6,
                        help='regularization parameter'
                        )
    parser.add_argument('--pdb', action='store_true', help='run with debuggger')    
    parser.add_argument('-r', '--run', type=str, required=True, help='run directory to analyze')

    return parser.parse_args()

args = parse_args()
if args.pdb:
    import pdb
    pdb.set_trace()

if not osp.exists(args.run):
    os.makedirs(args.run)
ckpt_dir = osp.join(args.run, 'ckpt')
images_dir = osp.join(args.run, 'images')
log_dir = osp.join(args.run, 'logs')

if not osp.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not osp.exists(images_dir):
    os.makedirs(images_dir)
if not osp.exists(log_dir):
    os.makedirs(log_dir)

device = torch.device('cpu' if args.cuda is None else 'cuda:{}'.format(args.cuda))
if args.dataset == 'wine':
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
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=42)
elif args.dataset == 'breast_cancer':
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer()
    feature_names = dataset.feature_names
    target_names = dataset.target_names
    d = pd.DataFrame(np.c_[dataset['target'], dataset['data']], columns= np.append(['diagnosis'], feature_names ))
    X_train = np.float32(d.drop('diagnosis', axis=1).values)
    y_train = np.int64(d['diagnosis'].values)   # conversion to int64 for pytorch

class RegNN(nn.Module):
    def __init__(self):
        super(RegNN, self).__init__()
        self.l1 = nn.Linear(11,64)
        self.l2 = nn.Linear(64,1)
    
    def forward(self,x):
        out = torch.sigmoid(self.l1(x))
        return self.l2(out)

class Breast_Cancer(nn.Module):
    def __init__(self):
        super(Breast_Cancer, self).__init__()
        self.l1 = nn.Linear(30,64)
        self.l2 = nn.Linear(64,2)
    
    def forward(self,x):
        out = torch.sigmoid(self.l1(x))
        return self.l2(out)
if args.dataset == 'wine':
    model = RegNN()
elif args.dataset == 'breast_cancer':
    model = Breast_Cancer()

model.cuda(device)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.01)

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Datasets
partition = X_train
labels = y_train

# Generators
training_set = Dataset(partition, labels)
training_loader = data.DataLoader(training_set, **params)
lmax_lst = []
lmin_lst = []
loss_lst = []
for epoch in range(args.num_epochs):
    for local_batch, local_labels in training_loader:
        # Transfer to GPU
        X_torch, y_torch = local_batch.to(device), local_labels.to(device)
        optimiser.zero_grad()
        y_pred = model(X_torch)
        loss = criterion(y_pred, y_torch)
#        loss = criterion(y_pred, y_torch.squeeze().long())

        if epoch >= args.scaling_epoch:
            ones = [torch.ones_like(p) for p in model.parameters()]
            dl_dw = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            d2l_dw2 = torch.autograd.grad(dl_dw, model.parameters(), ones, create_graph=True)
            sum_dl2_dw2 = sum([torch.abs(g).sum() if args.abs else g.sum() for g in d2l_dw2])
            loss = loss + args.sf * sum_dl2_dw2

        loss.backward()
        optimiser.step()
        print(loss)
        loss_lst.append(loss)
        Hess = FullHessian(crit='MSELoss',
                            loader=training_loader,
                            device=device,
                            model=model,
                            double=False,
                            num_classes=10,
                            hessian_type='Hessian',
                            init_poly_deg=64,
                            poly_deg=128,
                            spectrum_margin=0.05,
                            poly_points=1024,
                            SSI_iters=128
                            )

        lmin, lmax = Hess.compute_lb_ub()
        lmax_lst.append(lmax)
        lmin_lst.append(lmin)

if args.num_epochs > args.scaling_epoch:
    f=osp.join(ckpt_dir , 'scaling_lmax_lmin.npz')
    f_loss = osp.join(images_dir,'scaling_loss.png')    
    f_lmax = osp.join(images_dir,'scaling_lmax.png')
    f_lmin = osp.join(images_dir,'scaling_lmin.png')

else:
    f=osp.join(ckpt_dir ,'lmax_lmin.npz')
    f_loss = osp.join(images_dir,'loss.png')    
    f_lmax = osp.join(images_dir,'lmax.png')
    f_lmin = osp.join(images_dir,'lmin.png')


np.savez(f,lmax=np.array(lmax),lmin=np.array(lmin))

import matplotlib.pyplot as plt
plt.plot(lmax_lst)
plt.xlabel('iterations')
plt.ylabel('lamda_max')
plt.savefig(f_lmax,dpi=1000)
plt.close()

plt.plot(lmin_lst)
plt.xlabel('iterations')
plt.ylabel('lamda_min')
plt.savefig(f_lmin,dpi=1000)
plt.close()

plt.plot(loss_lst)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig(f_loss,dpi=1000)
plt.close()

