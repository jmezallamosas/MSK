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

datadir = '/data1/shahs3/users/mezallj1/data/merfish'#dataset path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# the location of R, which is necessary for mclust algorithm. Please replace it with local R installation path
os.environ['R_HOME'] ='/home/mezallj1/miniconda3/envs/graphst/lib/R'

ari_dic = {}

for filename in os.listdir(f'{datadir}/raw'):

    file_path = os.path.join(f'{datadir}/raw', filename)

    if os.path.isfile(file_path):

        adata = sc.read_h5ad(file_path)

        # the number of clusters
        n_clusters = 8

        adata.var_names_make_unique()

        # define and train model
        model = GraphST.GraphST(adata, device=device)
        adata = model.train()

        # set radius to specify the number of neighbors considered during refinement
        radius = 50

        # clustering
        from GraphST.utils import clustering
        clustering(adata, n_clusters, radius=radius, refinement=False) #For DLPFC dataset, we use optional refinement step.

        obs_df = adata.obs.dropna()

        ari = metrics.adjusted_rand_score(obs_df['domain'], obs_df['Region'])

        print('File, %s, ARI, %.2f'%(filename, ari))

        ari_dic[filename[8:11]] = ari
        
        adata.write(f'{datadir}/graphst/'+filename[8:11]+'_graphst_seeded.h5ad')

with open(f'{datadir}/graphst/ari_graphst_seeded.pkl', 'wb') as f:
    pickle.dump(ari_dic, f)
