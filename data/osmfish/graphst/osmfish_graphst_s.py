import scanpy as sc
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle

from GraphST import GraphST

datadir = '/data1/shahs3/users/mezallj1/data/osmfish'#dataset path

# the location of R, which is necessary for mclust algorithm. Please replace it with local R installation path
os.environ['R_HOME'] ='/home/mezallj1/miniconda3/envs/graphst/lib/R'

ari_dic = {}

for i in range(10):
    
    adata = sc.read_h5ad(f'{datadir}/raw/osmfish_remove_excluded.h5ad')

    # the number of clusters
    n_clusters = 11

    adata.var_names_make_unique()
    
    # define and train model
    model = GraphST.GraphST(adata)
    adata = model.train()
    
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    
    # clustering
    from GraphST.utils import clustering
    clustering(adata, n_clusters, radius=radius, refinement=False) #For DLPFC dataset, we use optional refinement step.

    obs_df = adata.obs.dropna()
    
    ari = metrics.adjusted_rand_score(obs_df['domain'], obs_df['Region'])

    print('Iteration, %d, ARI, %.2f'%(i+1, ari))
    
    ari_dic[i+1] = ari
        
    adata.write(f'{datadir}/graphst/osmfish_graphst_s_{i+1}.h5ad')

with open(f'{datadir}/graphst/ari_graphst_s.pkl', 'wb') as f:
    pickle.dump(ari_dic, f)
