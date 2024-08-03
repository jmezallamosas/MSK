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

datadir= '/data1/shahs3/users/mezallj1/data/merfish'

def mclust_P(adata, num_cluster, used_obsm='STAGATE', modelNames='EEE'):
     from sklearn import mixture
     np.random.seed(2020)
     g = mixture.GaussianMixture(n_components=num_cluster, covariance_type='diag')
     res = g.fit_predict(adata.obsm[used_obsm])
     adata.obs['mclust'] = res
     return adata

ari_dic = {}

for filename in os.listdir(f'{datadir}/raw'):

    file_path = os.path.join(f'{datadir}/raw', filename)

    if os.path.isfile(file_path):

        adata = sc.read_h5ad(file_path)

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
        #adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=8)
        adata = mclust_P(adata, used_obsm='STAGATE', num_cluster=8)

        obs_df = adata.obs.dropna()

        ari = adjusted_rand_score(obs_df['mclust'], obs_df['Region'])

        print('File, %s, ARI, %.2f'%(filename, ari))

        ari_dic[filename[8:11]] = ari
        
        adata.write(f'{datadir}/stagate/'+filename[8:11]+'_stagate_seeded.h5ad')

with open(f'{datadir}/stagate/ari_stagate_seeded.pkl', 'wb') as f:
    pickle.dump(ari_dic, f)
