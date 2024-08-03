import grafiti as gf
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from sklearn import metrics
import seaborn as sns
import pandas as pd
import os
import pickle
import torch

import warnings
warnings.filterwarnings('ignore')

datadir = "/data1/shahs3/users/mezallj1/data/dlpfc"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ari_dic = {}

for filename in os.listdir(f'{datadir}/raw'):

    file_path = os.path.join(f'{datadir}/raw', filename)

    if os.path.isfile(file_path):

        adata = sc.read_h5ad(file_path)

        #Normalization
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #sc.pp.scale(adata, zero_center=False, max_value=10)
        #adata = adata[:,adata.var['highly_variable']]

        sq.gr.spatial_neighbors(adata,n_rings=1,coord_type='grid',delaunay=False) # Creates spatial_connectivities and spatial_distances in 'obsp' from spatial location (x,y) in 'obsm'

        gae = gf.ml.GAE(adata, layers=[50,50], lr=0.0001, device=device, alpha=5, beta=5, gamma=0.995)#, exponent=2, distance_scale=10)

        gae.train(10000, update_interval=100, threshold=1e-3, patience=10)

        gae.load_embedding(adata, encoding_key="X_grafiti") # Load features into the z latent space

        gf.tl.umap(adata) # Embed grafiti latent space into umap latent space (2 dimensions)

        gf.tl.find_motifs_gmm(adata,k=7)

        obs_df = adata.obs.dropna()

        ari = metrics.adjusted_rand_score(obs_df['grafiti_motif'], obs_df['Region'])

        print('File, %s, ARI, %.2f'%(filename, ari))

        ari_dic[filename[:6]] = ari
        
        adata.write(f'{datadir}/grafiti/{filename[:6]}_grafiti_dcl_{gae.encoder_layers[0]}_{gae.encoder_layers[1]}_{gae.lr}_{gae.alpha}_{gae.beta}_{gae.gamma}.h5ad')

with open(f'{datadir}/grafiti/ari_grafiti_dcl_{gae.encoder_layers[0]}_{gae.encoder_layers[1]}_{gae.lr}_{gae.alpha}_{gae.beta}_{gae.gamma}.h5ad', 'wb') as f:
    pickle.dump(ari_dic, f)
