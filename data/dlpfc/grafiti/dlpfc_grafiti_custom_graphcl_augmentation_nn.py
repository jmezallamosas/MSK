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

############################ Custom Grafiti
import scanpy as sc
import scipy.sparse
import numpy as np
import seaborn as sns
import umap
import torch.nn.functional as F
import torch
from torch import Tensor
import torch_scatter
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch.nn as nn
from torch_geometric.nn import models
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import aggr
from torch_geometric.nn import MessagePassing
from sklearn import preprocessing
import random

class GrafitiEncoderLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GrafitiEncoderLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def message(self, x_j, edge_attr):
        edge_attr = edge_attr.to(x_j.dtype) 
        return x_j / edge_attr.unsqueeze(-1) 

    def forward(self, x, edge_index, edge_attr):
        ret = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        ret = self.lin(ret) 
        return F.leaky_relu(ret, negative_slope=0.01)
    
class GrafitiDecoderLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(GrafitiDecoderLayer, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def message(self, x_j, edge_attr): 
        edge_attr = edge_attr.to(x_j.dtype)
        degree = x_j.size(0) 
        degree_normalized_message = x_j / edge_attr.unsqueeze(-1) 
        res = degree_normalized_message / degree
        return res

    def aggregate(self, inputs, index, dim_size=None):
        res = torch_scatter.scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        return res

    def forward(self, x, edge_index, edge_attr):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        transformed_features = x - aggr_out
        transformed_features = self.lin(transformed_features) 
        return F.leaky_relu(transformed_features, negative_slope=0.01)
    

class GrafitiEncoderModule(torch.nn.Module):
    def __init__(self, in_dim, layers=[10,10]):
        super(GrafitiEncoderModule, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleList()
        lhidden_dim = self.layers[0]
        self.conv.append(GrafitiEncoderLayer(in_dim, lhidden_dim))
        for hidden_dim in self.layers[1:]:
            self.conv.append(GrafitiEncoderLayer(lhidden_dim, hidden_dim))
            lhidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        for conv in self.conv:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return x

class GrafitiDecoderModule(torch.nn.Module):
    def __init__(self, in_dim, layers=[30,30]):
        super(GrafitiDecoderModule, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleList()
        lhidden_dim = self.layers[0]
        self.conv.append(GrafitiDecoderLayer(in_dim, lhidden_dim))
        for hidden_dim in self.layers[1:]:
            self.conv.append(GrafitiDecoderLayer(lhidden_dim, hidden_dim))
            lhidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        for conv in self.conv:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return x

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return torch.mean(global_emb,0)

class Discriminator(nn.Module):
    def __init__(self, n_hidden_layers, device):
        super(Discriminator, self).__init__()
        self.bilin = nn.Bilinear(n_hidden_layers, n_hidden_layers, 1).to(device)  # Bilinear layer to compute similarity

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)  # Initialize weights
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # Initialize bias

    def forward(self, s, h, h_a, s_bias1=None, s_bias2=None):
        s_x = torch.unsqueeze(s,0)
        s_x = s.expand_as(h)  # Expand s to match the size of h

        sc_1 = self.bilin(h, s_x)  # Compute similarity score for positive samples
        sc_2 = self.bilin(h_a, s_x)  # Compute similarity score for negative samples

        if s_bias1 is not None:
            sc_1 += s_bias1  # Add bias to positive scores
        if s_bias2 is not None:
            sc_2 += s_bias2  # Add bias to negative scores

        logits = torch.cat((sc_1, sc_2), 1)  # Concatenate scores to form logits

        return logits
        
def corrupt_features(x):
    """Randomly permute the node features to create corrupted features."""
    perm = torch.randperm(x.size(0))
    return x[perm] 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def aug_random_mask(x, drop_rate=0.2):
    # Determine device
    device = x.device

    # Determine the number of features to mask
    num_features = x.size(1)
    
    # Generate a mask for the features to keep
    mask = torch.rand(num_features, device=device) > drop_rate

    # Create the augmented feature matrix by masking selected features
    aug_x = x.clone()
    aug_x[:, ~mask] = 0  # Mask the selected features by setting them to zero

    return aug_x

def aug_edge_perturbation(edge_index, edge_attr, drop_rate=0.2):
    
    # Determine device
    device =  edge_index.device
    
    # Generate mask for nodes to keep
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=device) > drop_rate
    idx_nondrop = mask.nonzero(as_tuple=False).squeeze()

    # Create a mapping from old to new indices
    idx_map = -torch.ones(num_edges, dtype=torch.long, device=device)
    idx_map[idx_nondrop] = torch.arange(idx_nondrop.size(0), dtype=torch.long, device=device)

    # Filter edges and edge attributes based on the new mask
    mask_src = mask[edge_index[0]]
    mask_tgt = mask[edge_index[1]]
    valid_edges = mask_src & mask_tgt
    new_edge_index = idx_map[edge_index[:, valid_edges]]
    new_edge_attr = edge_attr[valid_edges]

    return new_edge_index, new_edge_attr


def aug_drop_node(x, edge_index, edge_attr, drop_rate=0.2):
    
    # Determine device
    device =  x.device
    
    # Generate mask for nodes to keep
    num_nodes = x.size(0)
    mask = torch.rand(num_nodes, device=device) > drop_rate
    idx_nondrop = mask.nonzero(as_tuple=False).squeeze()

    # Filter node features
    x_dropped = x[idx_nondrop]

    # Create a mapping from old to new indices
    idx_map = -torch.ones(num_nodes, dtype=torch.long, device=device)
    idx_map[idx_nondrop] = torch.arange(idx_nondrop.size(0), dtype=torch.long, device=device)

    # Filter edges and edge attributes based on the node mask
    mask_src = mask[edge_index[0]]
    mask_tgt = mask[edge_index[1]]
    valid_edges = mask_src & mask_tgt
    new_edge_index = idx_map[edge_index[:, valid_edges]]
    new_edge_attr = edge_attr[valid_edges]

    return x_dropped, new_edge_index, new_edge_attr


def aug_subgraph(x, edge_index, edge_attr, drop_rate=0.2):

    # Determine device
    device = x.device

    # Convert edge_index to adjacency matrix
    num_nodes = x.size(0)
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1

    # Calculate the number of nodes to keep
    s_node_num = int(num_nodes * (1 - drop_rate))
    center_node_id = random.randint(0, num_nodes - 1)

    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        all_neighbor_list += torch.nonzero(adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if n not in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    # Generate the drop_node_list
    all_node_list = list(range(num_nodes))
    drop_node_list = sorted([i for i in all_node_list if i not in sub_node_id_list])

    # Filter the node features
    aug_x = torch.index_select(x, 0, torch.tensor(sub_node_id_list, device=device))

    # Update edge_index and edge_attr
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[drop_node_list] = False

    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    new_edge_index = edge_index[:, edge_mask]
    new_edge_attr = edge_attr[edge_mask]

    # Map old indices to new indices
    idx_map = -torch.ones(num_nodes, dtype=torch.long, device=device)
    idx_map[torch.tensor(sub_node_id_list, device=device)] = torch.arange(len(sub_node_id_list), device=device)

    new_edge_index = idx_map[new_edge_index]

    return aug_x, new_edge_index, new_edge_attr

class GAE(object):

    def __init__(self, adata, layers=[50,50], lr=0.00001, distance_threshold=None, exponent=2, distance_scale=None, device='cpu', aug_type='edge', drop_rate=0.2, alpha=10, beta=1, gamma=1, seed=42):
        seed_everything(seed)  # Seed everything for reproducibility
        self.lr = lr
        self.device = torch.device(device)
        print("Generating PyTorch Geometric Dataset...")
        if distance_threshold != None:
            distances = adata.obsp["spatial_distances"]
            connectiv = adata.obsp["spatial_connectivities"]
            rows, cols = distances.nonzero()
            for row, col in zip(rows, cols):
                if distances[row, col] > distance_threshold:
                    connectiv[row, col] = 0
            adata.obsp["spatial_connectivities"] = connectiv
        self.adj = adata.obsp["spatial_connectivities"]
        edges = self.adj.nonzero()

        # Check if adata.X is a scipy.sparse.csr_matrix
        if scipy.sparse.issparse(adata.X):
            x = torch.from_numpy(adata.X.toarray())
        else:
            x = torch.from_numpy(adata.X)

        x = x.float().to(self.device)
        e = torch.from_numpy(np.array(edges)).type(torch.int64).to(self.device)
        self.graph_neigh = torch.from_numpy(self.adj.toarray()).type(torch.int64).to(self.device) + torch.eye(torch.from_numpy(self.adj.toarray()).type(torch.int64).shape[0]).to(self.device)
        attrs = [adata.obsp["spatial_distances"][x,y] for x,y in zip(*edges)]
        if distance_scale!=None:
            scaler = preprocessing.MinMaxScaler(feature_range=(0,distance_scale))
            attrs = scaler.fit_transform(np.array(attrs).reshape(-1,1)).reshape(1,-1)
            attrs = 1. / (np.array(attrs)**exponent)
            attrs = attrs[0]
        else:
            attrs = np.array(attrs)
        data = Data(x=x, edge_index=e, edge_attr=attrs)
        self.adata = adata
        data.edge_attr = torch.from_numpy(data.edge_attr).to(self.device)
        self.encoder_layers = layers
        self.decoder_layers = list(reversed(layers[1:])) + [data.num_features]
        print("Setting up Model...")
        self.encoder = GrafitiEncoderModule(data.num_features,layers=self.encoder_layers).to(self.device)
        self.decoder = GrafitiDecoderModule(layers[-1],layers=self.decoder_layers).to(self.device)
        self.gae = models.GAE(encoder=self.encoder,decoder=self.decoder).to(self.device)
        self.optimizer = torch.optim.Adam(self.gae.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma)
        self.contrastive_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.losses = []
        self.global_epoch = 0
        self.data = data
        self.aug_type = aug_type
        self.drop_rate = drop_rate
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(layers[-1], self.device)
        self.alpha = alpha # Importance parameter for reconstruction loss
        self.beta = beta # Importance parameter for contrastive loss
        self.gamma = gamma # Parameter for learning rate scheduler
        print("Ready to train!")

    def train(self, epochs, update_interval=5, threshold=0.001, patience=10):
        prev_loss = np.inf
        best_loss = np.inf
        patience_counter = 0 # Counter to track the number of epochs without improvement

        for i in range(epochs):
            self.optimizer.zero_grad()

            # Preparing corrupt data
            self.data.xc = corrupt_features(self.data.x) # Dynamic augmentation
            
            # Preparing augmented data
            #self.data.x1, self.data.edge_index1, self.data.edge_attr1 = graph_augmentation(self.data.x, self.data.edge_index, self.data.edge_attr, device=self.device)
            #self.data.x2, self.data.edge_index2, self.data.edge_attr2 = graph_augmentation(self.data.x, self.data.edge_index, self.data.edge_attr, device=self.device)

            if self.aug_type == 'node':
                
                self.data.x1, self.data.edge_index1, self.data.edge_attr1 = aug_drop_node(self.data.x, self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate)
                self.data.x2, self.data.edge_index2, self.data.edge_attr2 = aug_drop_node(self.data.x, self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate)
            
            elif self.aug_type == 'edge':

                self.data.edge_index1, self.data.edge_attr1 = aug_edge_perturbation(self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate) # random drop edges
                self.data.edge_index2, self.data.edge_attr2 = aug_edge_perturbation(self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate) # random drop edges

                self.data.x1 = self.data.x
                self.data.x2 = self.data.x
                
            elif self.aug_type == 'mask':
            
                self.data.x1 = aug_random_mask(self.data.x,  drop_rate=self.drop_rate)
                self.data.x2 = aug_random_mask(self.data.x,  drop_rate=self.drop_rate)

                self.data.edge_index1, self.data.edge_attr1 = self.data.edge_index, self.data.edge_attr
                self.data.edge_index2, self.data.edge_attr2 = self.data.edge_index, self.data.edge_attr

            elif self.aug_type == 'subgraph':
                
                self.data.x1, self.data.edge_index1, self.data.edge_attr1 = aug_subgraph(self.data.x, self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate)
                self.data.x2, self.data.edge_index2, self.data.edge_attr2 = aug_subgraph(self.data.x, self.data.edge_index, self.data.edge_attr, drop_rate=self.drop_rate)
            
            
            # Encoding original, corrupted and augmented graphs to latent space
            
            h = self.gae.encode(self.data.x, self.data.edge_index, self.data.edge_attr)
            hc = self.gae.encode(self.data.xc, self.data.edge_index, self.data.edge_attr)
            h1 = self.gae.encode(self.data.x1, self.data.edge_index1, self.data.edge_attr1)
            h2 = self.gae.encode(self.data.x2, self.data.edge_index2, self.data.edge_attr2)

            # Summarizing latent embeddings of the two augmented graphs to capture local context
            s1 = self.read(h1, to_dense_adj(self.data.edge_index1, max_num_nodes=self.data.x1.size(0))[0].fill_diagonal_(1)).to(self.device)
            s1 = self.sigm(s1).to(self.device) # Normalize to 0-1 probabilities
    
            s2 = self.read(h2, to_dense_adj(self.data.edge_index2, max_num_nodes=self.data.x2.size(0))[0].fill_diagonal_(1)).to(self.device)
            s2 = self.sigm(s2).to(self.device) # Normalize to 0-1 probabilities

            # Construction of logits (raw scores that represent the similarity between node embeddings and the summary vector)
            logits1 = self.disc(s1, h, hc)
            logits2 = self.disc(s2, h, hc)
            logits = logits1 + logits2
            
            # Constrastive Loss
            labels = torch.cat([torch.ones(logits.shape[0], 1), torch.zeros(logits.shape[0], 1)], dim=1).to(logits.device)   
            contrastive_loss = self.contrastive_loss(logits, labels)
            
            # Reconstruction Loss
            reconstruction = self.gae.decode(h, self.data.edge_index, self.data.edge_attr)
            reconstruction_loss = self.reconstruction_loss(reconstruction, self.data.x)

            # Total Loss with the importance parameters
            loss = self.alpha * reconstruction_loss + self.beta * contrastive_loss
            loss.backward()
            self.optimizer.step() 
            self.scheduler.step() # Step the learning rate scheduler
            self.losses.append(loss.item())
            if i % update_interval == 0:
                print("Epoch {} ** iteration {} ** Loss: {}".format(self.global_epoch, i, np.mean(self.losses[-update_interval:])))
            self.global_epoch += 1
            curr_loss = loss.item()

            # Check for improvement
            if curr_loss < best_loss - threshold:
                best_loss = curr_loss
                patience_counter = 0 # Reset the counter if there is an improvement
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print("Early stopping due to no improvement over {} epochs.".format(patience))
                break

            prev_loss = curr_loss # Update previous loss

        print("Training Complete.")

    def __str__(self):
        fmt = "Pytorch Dataset\n\n"
        fmt += str(self.data) + "\n\n"
        fmt += "GAE Architecture\n\n"
        fmt += str(self.gae) + "\n"
        return fmt

    def plot(self):
        sns.lineplot(self.losses)

    def save(self, path):
        torch.save(self.gae.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path)
        self.gae.load_state_dict(state_dict)

    def load_embedding(self, adata, encoding_key="X_grafiti"):
        with torch.no_grad():
            z = self.gae.encode(self.data.x, self.data.edge_index, self.data.edge_attr)
            #h = self.gae.decode(z, self.data.edge_index, self.data.edge_attr)
            #hcpu = h.detach().cpu().numpy()
            #adata.obsm[encoding_key] = hcpu
            zcpu = z.detach().cpu().numpy()
            adata.obsm[encoding_key] = zcpu


#################### Pipeline
ari_dic = {}

for iteration, filename in enumerate(os.listdir(f'{datadir}/raw')):

    file_path = os.path.join(f'{datadir}/raw', filename)

    if os.path.isfile(file_path):

        adata = sc.read_h5ad(file_path)

        #Normalization
        #sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        #sc.pp.normalize_total(adata, target_sum=1e4)
        #sc.pp.log1p(adata)
        #sc.pp.scale(adata, zero_center=False, max_value=10)
        #adata = adata[:,adata.var['highly_variable']]

        print(f'{filename}')
        
        sq.gr.spatial_neighbors(adata,n_rings=1,coord_type='grid',delaunay=False) # Creates spatial_connectivities and spatial_distances in 'obsp' from spatial location (x,y) in 'obsm'

        gae = GAE(adata, layers=[50,50], lr=0.0001, device=device, aug_type='node', alpha=10, beta=5, gamma=1)#, exponent=2, distance_scale=10)

        gae.train(10000, update_interval=100, threshold=1e-3, patience=10)

        gae.load_embedding(adata, encoding_key="X_grafiti") # Load features into the z latent space

        gf.tl.umap(adata) # Embed grafiti latent space into umap latent space (2 dimensions)

        gf.tl.find_motifs_gmm(adata,k=7)

        obs_df = adata.obs.dropna()

        ari = metrics.adjusted_rand_score(obs_df['grafiti_motif'], obs_df['Region'])

        print(f'{iteration+1},{filename},{gae.aug_type},{gae.encoder_layers[0]},{gae.encoder_layers[1]},{gae.lr},{gae.alpha},{gae.beta},{gae.gamma},{ari}')

        ari_dic[filename[:6]] = ari
        
        adata.write(f'{datadir}/grafiti/{filename[:6]}_grafiti_gcl_nn_{gae.aug_type}_{gae.encoder_layers[0]}_{gae.encoder_layers[1]}_{gae.lr}_{gae.alpha}_{gae.beta}_{gae.gamma}.h5ad')

with open(f'{datadir}/grafiti/ari_grafiti_gcl_nn_{gae.aug_type}_{gae.encoder_layers[0]}_{gae.encoder_layers[1]}_{gae.lr}_{gae.alpha}_{gae.beta}_{gae.gamma}.h5ad', 'wb') as f:
    pickle.dump(ari_dic, f)
