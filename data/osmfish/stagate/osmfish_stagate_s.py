import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import pickle

from sklearn.metrics.cluster import adjusted_rand_score

import warnings
warnings.filterwarnings("ignore")

import STAGATE_pyG

datadir= '/data1/shahs3/users/mezallj1/data/osmfish'

def mclust_P(adata, num_cluster, used_obsm='STAGATE', modelNames='EEE'):
     from sklearn import mixture
     g = mixture.GaussianMixture(n_components=num_cluster, covariance_type='diag')
     res = g.fit_predict(adata.obsm[used_obsm])
     adata.obs['mclust'] = res
     return adata

ari_dic = {}

for i in range(10):

    adata = sc.read_h5ad(f'{datadir}/raw/osmfish_remove_excluded.h5ad')
    
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    #Constructing the spatial network
    STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE_pyG.Stats_Spatial_Net(adata)

    #Running STAGATE
    adata = STAGATE_pyG.train_STAGATE(adata)

    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    #adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=11)
    adata = mclust_P(adata, used_obsm='STAGATE', num_cluster=11)
    
    obs_df = adata.obs.dropna()

    ari = adjusted_rand_score(obs_df['mclust'], obs_df['Region'])

    print('Iteration, %d, ARI, %.2f'%(i+1, ari))
    
    ari_dic[i+1] = ari
        
    adata.write(f'{datadir}/stagate/osmfish_stagate_s_{i+1}.h5ad')

with open(f'{datadir}/stagate/ari_stagate_s.pkl', 'wb') as f:
    pickle.dump(ari_dic, f)
