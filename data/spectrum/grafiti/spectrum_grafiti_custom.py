import grafiti as gf
import scanpy as sc
import seaborn as sns
import torch

import warnings
warnings.filterwarnings('ignore')

datadir = "/data1/shahs3/users/mezallj1/data/spectrum"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(f'{datadir}/raw/spectrum_after_links.h5ad') # SPECTRUN_squidpy.h5ad after cc.gr.remove_long_links()

features = ['panCK', 'CD8', 'CD68', 'PD1', 'PDL1', 'TOX']
new_adata = adata[:, features].copy()

gae = gf.ml.GAE(new_adata, layers=[50,7], lr=0.001, device=device, alpha=5, beta=10, gamma=0.995)#, exponent=2, distance_scale=10)
gae.train(10000, update_interval=100, threshold=1e-3, patience=10)
gae.load_embedding(adata, encoding_key="X_grafiti") # Load features into the z latent space
gf.tl.umap(adata) # Embed grafiti latent space into umap latent space (2 dimensions)
gf.tl.find_motifs_gmm(adata,k=15)
adata.write(f'{datadir}/grafiti/spectrum_grafiti_dcl_{gae.encoder_layers[0]}_{gae.encoder_layers[1]}_{gae.lr}_{gae.alpha}_{gae.beta}_{gae.gamma}.h5ad')

