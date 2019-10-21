import numpy as np
from tqdm import tqdm
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

dist = []

####### calculating p-angles######
for i in tqdm(range(1,args.n)):
	if i == 1:
		eig_prev = np.load('{}/model_0.npz'.format(path))
		eig_prev = eig_prev['eigvecs_cur']
		eig_prev = normalize(eig_prev,axis=1,norm='l2')
		############ Normalize the eigvecs_cur########
		
	eig_cur = np.load('{}/model_{}.npz'.format(path,i))
	eig_cur = eig_cur['eigvecs_cur']
	eig_cur = normalize(eig_cur, axis =1, norm ='l2')
	_,s,_ = np.linalg.svd(np.matmul(eig_prev.transpose(),eig_cur))
	dist.append(np.linalg.norm(s,2))
np.save(f'{args.r}/subspace/dist.npy',np.log(dist))

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(np.log(dist))
plt.savefig(f'{args.r}/images/dist_log.png',dpi=1000)
plt.close()

