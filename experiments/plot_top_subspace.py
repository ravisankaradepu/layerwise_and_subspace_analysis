import numpy as np
import argparse
from sklearn.preprocessing import normalize
import os.path as osp
parser = argparse.ArgumentParser(description='train')

parser.add_argument('-r',
                    type=str,
                    help='run directory prefix')
parser.add_argument('-n', type=int, help='no of models')


parser.add_argument('-pdb','--with_pdb',
                    action='store_true',
                    help='run with python debugger')

args = parser.parse_args()

if args.with_pdb:
	import pdb
	pdb.set_trace()

path=osp.join(args.r,'subspace')

####### calculating p-angles######
for i in range(1,args.n):
	if i == 1:
		eig_prev = np.load('{}/model_0.npz'.format(path))
		eig_prev = eig_prev['eigvecs_cur']
		eig_prev = normalize(eig_prev,axis=1,norm='l2')
		############ Normalize the eigvecs_cur########
		
	eig_cur = np.load('{}/model_{}.npz'.format(path,i))
	eig_cur = eig_cur['eigvecs_cur']
	_,s,_ = np.svd(np.matmul(eig_prev.transpose(),eig_cur))
	print(np.norm(s,2))
