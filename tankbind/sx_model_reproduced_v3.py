# sx_model_reproduced_v3.py
# Compare with v2:
# modification: NCIYesLoss Function:

import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch import Tensor
from typing import List
from torch.nn import Linear
import sys
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from GATv2 import GAT
from GINv2 import GIN
import traceback
import pickle

import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):

        super(GVP_embedding, self).__init__()
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        seq = self.W_s(seq)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

class TriangleProteinToCompound(torch.nn.Module):
    def __init__(self, embedding_channels=256, c=128, hasgate=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)
        self.hasgate = hasgate
        if hasgate:
            self.gate_linear = Linear(embedding_channels, c)
        self.linear = Linear(embedding_channels, c)
        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j, 1
        z = self.layernorm(z)
        if self.hasgate:
            ab = self.gate_linear(z).sigmoid() * self.linear(z) * z_mask
        else:
            ab = self.linear(z) * z_mask
        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab)
        block2 = torch.einsum("bikc,bjkc->bijc", ab, compound_pair)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class TriangleProteinToCompound_v2(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        # pdrint(g.shape, block1.shape, block2.shape)
        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class Self_Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dp(attention_probs)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            # if self.training:
            #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
            # else:
            top_mask = (attention_weight_sorted_sorted<32)
            attention_probs = attention_probs * top_mask
            # attention_probs = attention_probs * attention_weight
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # pridnt(attention_probs.shape,v.shape)
        # attention_probs = self.dp(attention_probs)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.ln(outputs)
        return outputs

class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # pdrint(g.shape, block1.shape, block2.shape)
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        return z


class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, embedding_channels)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z


        

class sx_IaBNet_with_affinity(torch.nn.Module):
    def __init__(self, hidden_channels=128, embedding_channels=128, c=128, mode=0, protein_embed_mode=1, 
                 compound_embed_mode=1, n_trigonometry_module_stack=5, protein_bin_max=30, readout_mode=2, 
                 nciyes=False):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.protein_bin_max = protein_bin_max
        self.mode = mode
        self.protein_embed_mode = protein_embed_mode
        self.compound_embed_mode = compound_embed_mode
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.readout_mode = readout_mode
        self.nciyes = nciyes
        self.loss = NCIYesLoss()


        if protein_embed_mode == 0:
            self.conv_protein = GNN(hidden_channels, embedding_channels)
            self.conv_compound = GNN(hidden_channels, embedding_channels)
            # self.conv_protein = SAGEConv((-1, -1), embedding_channels)
            # self.conv_compound = SAGEConv((-1, -1), embedding_channels)
        if protein_embed_mode == 1:
            self.conv_protein = GVP_embedding((6, 3), (embedding_channels, 16), 
                                              (32, 1), (32, 1), seq_in=True)
            

        if compound_embed_mode == 0:
            self.conv_compound = GNN(hidden_channels, embedding_channels)
        elif compound_embed_mode == 1:
            self.conv_compound = GIN(input_dim = 56, hidden_dims = [128,56,embedding_channels], edge_input_dim = 19, concat_hidden = False)

        if mode == 0:
            self.protein_pair_embedding = Linear(16, c)
            self.compound_pair_embedding = Linear(16, c)
            self.protein_to_compound_list = []
            self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=embedding_channels, c=c) for _ in range(n_trigonometry_module_stack)])
            self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=embedding_channels) for _ in range(n_trigonometry_module_stack)])
            self.tranistion = Transition(embedding_channels=embedding_channels, n=4)

        self.linear = Linear(embedding_channels, 1)
        self.linear_energy = Linear(embedding_channels, 1)
        if readout_mode == 2:
            self.gate_linear = Linear(embedding_channels, 1)
        # self.gate_linear = Linear(embedding_channels, 1)
        self.bias = torch.nn.Parameter(torch.ones(1))
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.25)
    def forward(self, data):
        if self.protein_embed_mode == 0:
            x = data['protein'].x.float()
            edge_index = data[("protein", "p2p", "protein")].edge_index
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(x, edge_index)
        if self.protein_embed_mode == 1:
            nodes = (data['protein']['node_s'], data['protein']['node_v'])
            edges = (data[("protein", "p2p", "protein")]["edge_s"], data[("protein", "p2p", "protein")]["edge_v"])
            protein_batch = data['protein'].batch
            protein_out = self.conv_protein(nodes, data[("protein", "p2p", "protein")]["edge_index"], edges, data.seq)

        if self.compound_embed_mode == 0:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index
            compound_batch = data['compound'].batch
            compound_out = self.conv_compound(compound_x, compound_edge_index)
        elif self.compound_embed_mode == 1:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index.T
            compound_edge_feature = data[("compound", "c2c", "compound")].edge_attr
            edge_weight = data[("compound", "c2c", "compound")].edge_weight
            compound_batch = data['compound'].batch
            compound_out = self.conv_compound(compound_edge_index,edge_weight,compound_edge_feature,compound_x.shape[0],compound_x)['node_feature']

        # protein_batch version could further process b matrix. better than for loop.
        # protein_out_batched of shape b, n, c
        protein_out_batched, protein_out_mask = to_dense_batch(protein_out, protein_batch)
        compound_out_batched, compound_out_mask = to_dense_batch(compound_out, compound_batch)
    
        node_xyz = data.node_xyz

        p_coords_batched, p_coords_mask = to_dense_batch(node_xyz, protein_batch)
        # c_coords_batched, c_coords_mask = to_dense_batch(coords, compound_batch)

        protein_pair = get_pair_dis_one_hot(p_coords_batched, bin_size=2, bin_min=-1, bin_max=self.protein_bin_max)
        # compound_pair = get_pair_dis_one_hot(c_coords_batched, bin_size=1, bin_min=-0.5, bin_max=15)
        compound_pair_batched, compound_pair_batched_mask = to_dense_batch(data.compound_pair, data.compound_pair_batch)
        batch_n = compound_pair_batched.shape[0]
        max_compound_size_square = compound_pair_batched.shape[1]
        max_compound_size = int(max_compound_size_square**0.5)
        assert (max_compound_size**2 - max_compound_size_square)**2 < 1e-4
        compound_pair = torch.zeros((batch_n, max_compound_size, max_compound_size, 16)).to(data.compound_pair.device)
        for i in range(batch_n):
            one = compound_pair_batched[i]
            compound_size_square = (data.compound_pair_batch == i).sum()
            compound_size = int(compound_size_square**0.5)
            compound_pair[i,:compound_size, :compound_size] = one[:compound_size_square].reshape(
                                                                (compound_size, compound_size, -1))
        protein_pair = self.protein_pair_embedding(protein_pair.float())
        compound_pair = self.compound_pair_embedding(compound_pair.float())
        # b = torch.einsum("bik,bjk->bij", protein_out_batched, compound_out_batched).flatten()

        protein_out_batched = self.layernorm(protein_out_batched)
        compound_out_batched = self.layernorm(compound_out_batched)
        # z of shape, b, protein_length, compound_length, channels.
        z = torch.einsum("bik,bjk->bijk", protein_out_batched, compound_out_batched)
        z_mask = torch.einsum("bi,bj->bij", protein_out_mask, compound_out_mask)
        # z = z * z_mask.unsqueeze(-1)
        if self.mode == 0:
            for _ in range(1):
                for i_module in range(self.n_trigonometry_module_stack):
                    z = z + self.dropout(self.protein_to_compound_list[i_module](z, protein_pair, compound_pair, z_mask.unsqueeze(-1)))
                    z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
                    z = self.tranistion(z)
        # batch_dim = z.shape[0]

        b = self.linear(z).squeeze(-1)
        y_pred = b[z_mask]
        y_pred = y_pred.sigmoid() * 10   # normalize to 0 to 10.
        if self.readout_mode == 0:
            pair_energy = self.linear_energy(z).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        if self.readout_mode == 1:
            # valid_interaction_z = (z * z_mask.unsqueeze(-1)).mean(axis=(1, 2))
            valid_interaction_z = (z * z_mask.unsqueeze(-1)).sum(axis=(1, 2)) / z_mask.sum(axis=(1, 2)).unsqueeze(-1)
            affinity_pred = self.linear_energy(valid_interaction_z).squeeze(-1)
            # pdrint("z shape", z.shape, "z_mask shape", z_mask.shape,   "valid_interaction_z shape", valid_interaction_z.shape, "affinity_pred shape", affinity_pred.shape)
        if self.readout_mode == 2:
            pair_energy = (self.gate_linear(z).sigmoid() * self.linear_energy(z)).squeeze(-1) * z_mask
            affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        if self.nciyes:
            return y_pred, affinity_pred, z, z_mask
        else:
            return y_pred, affinity_pred






class NCIClassifier(nn.Module):
    def __init__(self, hidden_channels=128, mid_channels=128, output_dimension = 1, dropout_rate=0.1):
        super().__init__()
        self.mlp_1 = nn.Linear(hidden_channels, mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mlp_2 = nn.Linear(mid_channels, output_dimension)
    def forward(self, z, z_mask, pair_shape):
        #pdrint(f"PAIRSHAPE{pair_shape}")
        #prdint(f"ZSHAPE {z.shape}")
        #prdint(f"ZMASK_UNIQUE {z_mask.unique()}")
        z = self.mlp_1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.mlp_2(z)
        nci_pred = z.squeeze(-1)
        #prdint(f"NCICLASSIFIER ZSHAPE {z.shape} ZMASKSHAPE {z_mask.shape} NCIPREDSHAPE {nci_pred.shape}")
        return nci_pred
        
class FFN(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        y = self.layer1(x)
        y = self.gelu(y)
        y = self.dropout(y)
        return y

class NCIYes_IaBNet(torch.nn.Module):
    def __init__(self, hidden_channels=128, embedding_channels=128, c=128, mode=0, protein_embed_mode=1, 
                 compound_embed_mode=1, n_trigonometry_module_stack=5, 
                 protein_bin_max=30, readout_mode=2, output_classes=1, nciyes=False, class_weight=None,
                 margin=1, margin_weight=1, dist_weight=1, nci_weight=1):
        super().__init__()
        self.nciyes = nciyes
        self.class_weight = class_weight
        self.IaBNet = sx_IaBNet_with_affinity(hidden_channels, embedding_channels, c, 
                                              mode, protein_embed_mode, compound_embed_mode, 
                                              n_trigonometry_module_stack, protein_bin_max, 
                                              readout_mode, nciyes)
        self.NCIClassifier = NCIClassifier(hidden_channels, hidden_channels, output_classes) if self.nciyes else None
        self.loss = NCIYesLoss(margin, margin_weight, dist_weight, nci_weight, nciyes = self.nciyes, class_weight = self.class_weight)

    def forward(self, data, i):
        try:
            if self.nciyes:
                y_pred, affinity_pred, z, z_mask = self.IaBNet(data)
                nci_pred = self.NCIClassifier(z, z_mask, data.pair_shape)                
                return y_pred, affinity_pred, nci_pred
            else:
                return self.IaBNet(data)
        except Exception as e:
            print("\n")
            print(f" Error in <NCIYes_IaBNet> with batch {i}. Exception info: {e.__class__.__name__, e}")
            print(traceback.format_exc())
            print("\n")
            return None, None
        
    def calculate_loss(self, aff_pred, y_pred, nci_pred, aff_true, y_true, nci_true, right_pocket, i, y_batch, pair_shape):
        return self.loss(aff_pred, y_pred, nci_pred, aff_true, y_true, nci_true, right_pocket, i, y_batch, pair_shape)


    
class NCIYesLoss(nn.Module):
    def __init__(self, margin=1, margin_weight=1, dist_weight=1, nci_weight=1, nciyes=False, class_weight=None, nci_under_sampling=True):
        super().__init__()
        self.margin_weight = margin_weight 
        self.dist_weight = dist_weight
        self.nci_weight = nci_weight
        self.MarginLoss = TBMarginLoss(margin)
        self.DistLoss = TBDistLoss()
        self.nci_under_sampling = nci_under_sampling
        #class_weight = torch.tensor([1,1],dtype=torch.float32) if (class_weight is None) else class_weight
        if nciyes and (class_weight is None):
            self.NCILoss = nn.BCELoss()
            print(self.NCILoss)
        elif nciyes and (class_weight is not None):
            self.NCILoss = NCIClassifierLoss(class_weight, self.nci_under_sampling)
            print(self.NCILoss)
        print("$ NCIYesLoss.nciyes = ", nciyes)
        self.nciyes = nciyes
    
    def forward(self, aff_pred, y_pred, nci_pred, aff_true, y_true, nci_true, right_pocket, i, y_batch, pair_shape):
        try:
            loss = self.margin_weight * self.MarginLoss(aff_pred, aff_true, right_pocket)
            if sum(right_pocket):
                loss += self.dist_weight * self.DistLoss(y_pred, y_true, right_pocket, y_batch)
                loss += self.nci_weight * self.NCILoss(nci_pred, nci_true.long(), right_pocket, y_batch, pair_shape) if self.nciyes else 0
            return loss
        except Exception as e:
            print("\n")
            print(f" Error in <NCIYesLoss> with batch {i}. Exception info: {e.__class__.__name__, e}")
            print(traceback.format_exc())
            print("\n")
            return None #, None


class NCIClassifierLoss(nn.Module):
    def __init__(self,class_weight, nci_under_sampling):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weight)
        self.under_sampling = nci_under_sampling
    def forward(self, nci_pred, nci_true, right_pocket, y_batch, pair_shape):
        device = nci_pred.device
        samples = []
        shapes = []
        for i, (_right, _shape) in enumerate(zip(right_pocket, pair_shape)):
            if _right == True:
                samples.append(i)
                shapes.append(_shape)
        samples = torch.tensor(samples).to(device)
        _nci_pred = None
        for i, _shape in zip(samples, shapes):
            if _nci_pred is None:
                _nci_pred = nci_pred[i, 0:_shape[0], 0:_shape[1]].flatten(end_dim=-2)
            else:
                _nci_pred = torch.cat(
                    (_nci_pred, nci_pred[i,0:_shape[0],0:_shape[1]].flatten(end_dim=-2))
                )
        index = torch.isin(y_batch, samples)
        nci_true = nci_true[index]
        trues = torch.where(nci_true == True)[0].data
        falses = torch.where(nci_true == False)[0].data
        falses = torch.tensor(
            np.random.choice(
                np.array(falses.to("cpu")), 
                size = 2*len(trues), 
                replace=False)
        ).to(device)
        tf = torch.cat((trues, falses))
        return self.loss(_nci_pred[tf], nci_true[tf]) if len(tf) else 0


class TBDistLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, y_pred, y_true, right_pocket, y_batch):
        device = y_pred.device
        samples = []
        for i, _right in enumerate(right_pocket):
            if _right == True:
                samples.append(i)
        samples = torch.tensor(samples).to(device)
        index = torch.isin(y_batch, samples)
        return self.loss(y_pred[index], y_true[index])



class TBMarginLoss(nn.Module):
    """Loss module equivalent to `tankbind.utils.my_affinity_criterion`\n
    Args:
        margin (int): the margin value.
    """
    def __init__(self, margin=1):
        super().__init__()
        self.margin=margin
    def forward(self, aff_pred, aff_true, right_pocket):
        #mask = torch.tensor(right_pocket).to(aff_pred.device)
        right_pocket = np.array(right_pocket)
        affinity_loss = torch.zeros(aff_pred.shape, dtype=torch.float64).to(aff_pred.device)
        #print(aff_pred, aff_true, right_pocket)
        affinity_loss[right_pocket] = ((aff_pred-aff_true)**2)[right_pocket]
        affinity_loss[~right_pocket] = (((aff_pred-(aff_true-self.margin).relu()**2)))[~right_pocket]
        return affinity_loss.mean()




def sx_get_model(mode, logging, device, **kwargs):
    if mode == 0:
        logging.info("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN")
        model = NCIYes_IaBNet(**kwargs).to(device)
    return model







# == reproduced ====================================================================================================================
# 用于旧版训练脚本的 新Loss
class NFTLoss(nn.Module):
    """Loss function for tankbind, nci and frag.
    Args:
        margin (int | float): value of margin for margin loss.
        margin_weight (int) : weight of margin loss.
        dist_weight (int) : weight of dist loss.
        nci_weight (int) : weight of nci classification loss.
        with_nci (bool) : whether nci classifier is used.
        nci_under_sampling (bool) : if with_nci == True, whether under sampling strategy is applied.
        nci_class_weight (Tuple[int, int]) : weight of loss from positive/negative samples in NCI classification loss (MSELoss).
    """
    def __init__(self, margin=1, margin_weight=1, dist_weight=1, nci_weight=1, with_nci=True, nci_under_sampling=True, nci_class_weight=None):
        super().__init__()
        self.margin_weight = margin_weight
        self.dist_weight = dist_weight
        
        self.with_nci = with_nci
        self.nci_weight = nci_weight
        
        self.MarginLoss = NFTMarginLoss(margin=margin)
        self.DistLoss = nn.MSELoss() 
        self.NCILoss = NFTNCILoss(nci_class_weight, nci_under_sampling) if with_nci else None


    def forward(self, aff_pred, aff_true, y_pred, y_true, right_pocket, batch_index, y_batch, pair_shape=None, nci_pred=None, nci_true=None):
        try:
            loss_m, loss_m_seq = self.MarginLoss(aff_pred, aff_true, right_pocket)
            return (self.margin_weight * self.MarginLoss(aff_pred, aff_true, right_pocket) + 
                    self.margin_weight * self.DistLoss(y_pred, y_true)+ 
                    (self.NCILoss(nci_pred, nci_true.long(), right_pocket, y_batch, pair_shape)
                     if (sum(right_pocket) and self.with_nci) else 0))
        except Exception as e:
            print(f"Exception of batch {i} in NFTLoss(): Info:\n{e.__class__.__name__, e}")
            return None
    
    def loss_calcul_score(self, aff_pred, aff_true): #TODO: by sample as a list
        return self.DistLoss(aff_pred, aff_true) if self.return_mean else self.DistLoss(aff_pred, aff_true) * len(aff_true)
    
class NFTMarginLoss(nn.Module):
    """Class of the marginal loss function for afinity prediction.
        Loss function instantiated with this class returns a marginal MSE Loss depending on the pocket's nature (right/decoys)
    Args:
        margin (int | float): value of margin for decoys.
    """
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
    def forward(self, aff_pred, aff_true, right_pocket):
        loss = torch.zeros(1).to(aff_pred.device)[0]
        aff_diff = aff_pred - aff_true
        for _a, _p in zip(aff_diff, right_pocket):
            _loss = _a**2 if _p else (max(0, (_a + self.margin).relu()))**2
            loss += _loss
            loss_list.append(_loss.detach().cpu().item())
        return loss
    
class NFTNCILoss(nn.Module):
    def __init__(self, class_weight, nci_under_sampling):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weight)
        self.under_sampling = nci_under_sampling
    def forward(self, nci_pred, nci_true, right_pocket, y_batch, pair_shape):
        device = nci_pred.device
        samples = []
        shapes = []
        for i, (_right, _shape) in enumerate(zip(right_pocket, pair_shape)):
            if _right == True:
                samples.append(i)
                shapes.append(_shape)
        samples = torch.tensor(samples).to(device)
        _nci_pred = None
        for i, _shape in zip(samples, shapes):
            if _nci_pred is None:
                _nci_pred = nci_pred[i, 0:_shape[0], 0:_shape[1]].flatten(end_dim=-2)
            else:
                _nci_pred = torch.cat(
                    (_nci_pred, nci_pred[i,0:_shape[0],0:_shape[1]].flatten(end_dim=-2))
                )
        index = torch.isin(y_batch, samples)
        
        # np.random.seed(hash(str(nci_pred)))
        
        nci_true = nci_true[index]
        trues = torch.where(nci_true == True)[0].data
        falses = torch.where(nci_true == False)[0].data
        falses = torch.tensor(
            np.random.choice(
                np.array(falses.to("cpu")), 
                size = 2*len(trues), 
                replace=False)
        ).to(device)
        tf = torch.cat((trues, falses))
        return self.loss(_nci_pred[tf], nci_true[tf]) if len(tf) else 0
    
    
class TBMarginLoss_OLD(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin=margin
    def forward(self, aff_pred, aff_true, right_pocket):
        loss = None
        print("AFSAGSA", aff_pred, right_pocket)
        aff_pred = aff_pred - aff_true
        for _a, _p in zip(aff_pred, right_pocket):
            if _p:
                loss = _a**2 if loss is None else loss + _a**2
            else:
                loss = (torch.max(0, (_a + self.margin).relu()))**2 if loss is None else loss + (max(0, (_a + self.margin).relu()))**2
        print(loss)
        return (loss / len(aff_pred))