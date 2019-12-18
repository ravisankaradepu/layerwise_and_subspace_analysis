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


import math
def sin(x):
    return math.sin(x)**2
def cos(x):
    return math.cos(x)**2

c=np.vectorize(cos)
s=np.vectorize(sin)


dist_grassman = []
dist_binet = []
dist_chorda = []
dist_martin = []
dist_proc = []
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
	_,v,_ = np.linalg.svd(np.matmul(eig_prev.transpose(),eig_cur))

	dist_grassman.append(np.linalg.norm(v,2))
	dist_binet.append(np.sqrt(1-np.prod(c(v))))
	dist_chorda.append(np.sqrt(np.sum(s(v))))
	dist_martin.append(np.sqrt(math.log(np.prod(1.0/c(v)))))
	dist_proc.append(np.sqrt(np.sum(s(v/2.0))))

np.savez(f'{args.r}/subspace/all_dist.npz', dist_grassman = np.log(dist_grassman), dist_binet =np.log(dist_binet), dist_chorda = np.log(dist_chorda), dist_martin = np.log(dist_martin), dist_proc = np.log(dist_proc))

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(np.log(dist_grassman))
plt.savefig(f'{args.r}/images/dist_grassman.png',dpi=1000)
plt.close()

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(dist_binet)
plt.savefig(f'{args.r}/images/dist_binet.png',dpi=1000)
plt.close()

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(dist_chorda)
plt.savefig(f'{args.r}/images/dist_chorda.png',dpi=1000)
plt.close()

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(dist_martin)
plt.savefig(f'{args.r}/images/dist_martin.png',dpi=1000)
plt.close()

import matplotlib.pyplot as plt
plt.ylabel('Distance',size=20)
plt.xlabel('Iteration',size=20)
plt.plot(dist_grassman)
plt.savefig(f'{args.r}/images/dist_log.png',dpi=1000)
plt.close()


