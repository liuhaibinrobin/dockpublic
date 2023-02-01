import torch
module = torch.nn.Module
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch.nn import Linear
import sys
from torch_cluster import radius, radius_graph
from e3nn.nn import BatchNorm
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from spyrmsd import rmsd, molecule
from torch_scatter import scatter, scatter_mean
from torch.nn import functional as F
from torch_geometric.utils import degree
from torsion_geometry import *
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from GATv2 import GAT
from GINv2 import GIN
from utils import OptimizeConformer

import time
import logging

logger=logger = logging.getLogger()
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
        # print(g.shape, block1.shape, block2.shape)
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
        # print(attention_probs.shape,v.shape)
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
        # print(g.shape, block1.shape, block2.shape)
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

class TensorProductConvLayer(torch.nn.Module):
    #use diffdock TensorProductConvLayer to predict tr,rot,tor angles
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp =  o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out

class MultiHeadAttention_dis_bias(nn.Module):
    def __init__(self, embedding_channels, attention_dropout_rate, z_channel):
        super(MultiHeadAttention_dis_bias, self).__init__()

        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.linear = nn.Linear(embedding_channels, z_channel) #z:[b, pocket_len, ligand_len, 128] -> [b, pocket_len, ligand_len, z_channel]
        self.output_layer = nn.Linear(embedding_channels * z_channel, embedding_channels)
        self.bucket_num = 20
        self.dis_attn_linear=nn.Linear(self.bucket_num, z_channel)
        self.linear_energy = nn.Linear(z_channel, 1)
        self.gate_linear = Linear(z_channel, 1)
        self.leaky = torch.nn.LeakyReLU()
        self.bias = torch.nn.Parameter(torch.ones(1))
        
        self.linear_energy_prmsd = nn.Linear(z_channel, 1)
        self.gate_linear_prmsd = Linear(z_channel, 1)
        self.leaky_prmsd = torch.nn.LeakyReLU()
        self.bias_prmsd = torch.nn.Parameter(torch.ones(1))

        # self.confidence_predictor = nn.Sequential(
        #         nn.Linear(embedding_channels,ns),
        #         nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
        #         nn.ReLU(),
        #         nn.Dropout(confidence_dropout),
        #         nn.Linear(ns, ns),
        #         nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
        #         nn.ReLU(),
        #         nn.Dropout(confidence_dropout),
        #         nn.Linear(ns, num_confidence_outputs)
        #     )
        

        
    def forward(self, z, z_mask, pocket_rep, candicate_dis_matrix_batched ,edge_type_matrix=None, attn_bias=None, subsq_mask = None, valid=False,check=False):
        #z shape: [b, pocket_len, ligand_len, embedding_channels]
        #pocket_rep shape : [b, pocket_len, embedding_channels]
        #candicate_dis_matrix_batched shape: [b, ligand_len, pocket_len]
        #edge_type_matrix shape: [b, pocket_len, ligand_len,edge_type_pair_channels]
        
        
        z = self.linear(z).transpose(3,1)  #z shape :[b, z_channel, ligand_len, pocket_len]
        ligand_len, pocket_len = z.shape[2], z.shape[3]
        embedding_channels = pocket_rep.shape[-1]
        batch_size = z.shape[0]
        head_size = z_channel = z.shape[1] 

        #z=z*self.scale
        if attn_bias is not None:
            z = z + attn_bias
       
        z = torch.softmax(z, dim=-1)
        if not valid:
            z = self.att_dropout(z)
        # dis_attn_bias=get_dist_features(self,candicate_dis_matrix_batched, edge_type_matrix)
        edge_length_embedding = soft_one_hot_linspace(candicate_dis_matrix_batched,start=0.0,end=15,number=self.bucket_num,basis='smooth_finite',cutoff=True)
        z += self.dis_attn_linear(edge_length_embedding).view(batch_size, z_channel, ligand_len, pocket_len)
        z_mask_repeat = z_mask.repeat(z_channel, 1, 1).view([batch_size, -1, ligand_len, pocket_len])
        z = z.masked_fill(z_mask_repeat, 1e-9)
        pair_energy = (self.gate_linear(z.transpose(3,1)).sigmoid() * self.linear_energy(z.transpose(3,1))).squeeze(-1) * z_mask
        affinity_pred = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        pair_energy_prmsd = (self.gate_linear_prmsd(z.transpose(3,1)).sigmoid() * self.linear_energy_prmsd(z.transpose(3,1))).squeeze(-1) * z_mask
        prmsd_pred = self.leaky_prmsd(self.bias_prmsd + ((pair_energy_prmsd).sum(axis=(-1, -2))))
        z_ = z.view([batch_size,-1,z.shape[-1]])
        ligand_rep = torch.bmm(z_, pocket_rep).view([batch_size, head_size, -1, embedding_channels])  #[b, z_channel, ligand_len, embedding_channels]                              
        ligand_rep = ligand_rep.transpose(1, 2).contiguous()  # [b, ligand_len, z_channel, embedding_channels]
        ligand_rep = ligand_rep.view(batch_size, -1, head_size * embedding_channels) # [b, ligand_len, z_channel * embedding_channels]
        ligand_rep = self.output_layer(ligand_rep)
        return ligand_rep, affinity_pred, prmsd_pred

class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class IaBNet_with_affinity(torch.nn.Module):
    def __init__(self, hidden_channels=128, embedding_channels=128, c=128, mode=0, protein_embed_mode=1, compound_embed_mode=1,
                 n_trigonometry_module_stack=5, protein_bin_max=30, readout_mode=2, finetune=False, recycling_num=1,
                 logging=None): #TODO: recycling_num=1
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.protein_bin_max = protein_bin_max
        self.mode = mode
        self.protein_embed_mode = protein_embed_mode
        self.compound_embed_mode = compound_embed_mode
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.readout_mode = readout_mode
        self.finetune = finetune
        self.recycling_num = recycling_num
        self.lig_max_radius = 5
        self.logging = logging
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
            self.MultiHeadAttention_dis_bias = MultiHeadAttention_dis_bias(embedding_channels = 128, attention_dropout_rate = 0.1, z_channel = 5)
            #步骤三声明
            torch.nn.Module = module
            self.ns = ns = 24 #small score model 参数
            dropout = 0.1
            distance_embed_dim = 32
            self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns), #去除sigma_embed_dim维度的影响
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps='128x0e',
                # in_irreps=f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=ns + 128,
                residual=False,
                dropout=dropout,
                batch_norm=True
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps='128x0e',
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=128*2 + ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=True
                )
            self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            self.tr_final_layer = nn.Sequential(nn.Linear(1, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1)) #去除sigma_embed_dim维度的影响
            self.rot_final_layer = nn.Sequential(nn.Linear(1, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1)) #去除sigma_embed_dim维度的影响
            self.lig_distance_expansion = GaussianSmearing(0.0, 5, distance_embed_dim)
            self.center_distance_expansion = GaussianSmearing(0.0, 30, distance_embed_dim)
            self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )

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
            #这种GIN 不接受小分子键信息
            compound_out = self.conv_compound(compound_x, compound_edge_index)
        elif self.compound_embed_mode == 1:
            compound_x = data['compound'].x.float()
            compound_edge_index = data[("compound", "c2c", "compound")].edge_index.T
            compound_edge_feature = data[("compound", "c2c", "compound")].edge_attr
            edge_weight = data[("compound", "c2c", "compound")].edge_weight
            compound_batch = data['compound'].batch
            #GIN 包含了小分子键的信息
            compound_out = self.conv_compound(compound_edge_index,edge_weight,compound_edge_feature,compound_x.shape[0],compound_x)['node_feature']


        # protein_batch version could further process b matrix. better than for loop.
        # protein_out_batched of shape b, n, c
        protein_out_batched, protein_out_mask = to_dense_batch(protein_out, protein_batch)
        compound_out_batched, compound_out_mask = to_dense_batch(compound_out, compound_batch)  #转换为batch pad形式

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
        # print(protein_pair.shape, compound_pair.shape, b.shape)
        if self.mode == 0:
            for _ in range(1):
                for i_module in range(self.n_trigonometry_module_stack):
                    z = z + self.dropout(self.protein_to_compound_list[i_module](z, protein_pair, compound_pair, z_mask.unsqueeze(-1)))
                    z = z + self.dropout(self.triangle_self_attention_list[i_module](z, z_mask))
                    z = self.tranistion(z)
        # return z,z_mask, protein_out_batched,compound_out_batched, compound_out, compound_batch
        pair_energy = (self.gate_linear(z).sigmoid() * self.linear_energy(z)).squeeze(-1) * z_mask
        affinity_pred_A = self.leaky(self.bias + ((pair_energy).sum(axis=(-1, -2))))
        #self.logging.info(f"after point A, z shape: {z.shape}, compound_out_batched shape: {compound_out_batched.shape}, protein_out_batched shape: {protein_out_batched.shape}, affinity_pred_A shape: {affinity_pred_A.shape}")
        # 步骤三：torsional 

        affinity_pred_B_list = []
        prmsd_pred_list = []
        rmsd_loss_list = []
        tr_tor_rot_loss_list=[]

        data_new_list=[copy.deepcopy(data.detach().to("cpu")),]
        rmsd_loss_list.append(self.get_symmetry_rmsd(data).to("cpu"))

        for _ in range(self.recycling_num):
            protein_num_batch = degree(data['protein'].batch, dtype=torch.long).tolist()
            compound_num_batch = degree(data['compound'].batch, dtype=torch.long).tolist()
            batch_size, ligand_len, pocket_len = z.shape[0], z.shape[2], z.shape[1]
            candicate_dis_matrix_batched = data.candicate_dis_matrix.new_full([batch_size, ligand_len, pocket_len], 0.)
            for i in range(batch_size):
                candicate_dis_matrix_batched[i, :compound_num_batch[i], :protein_num_batch[i]] = self.unbatch(data.candicate_dis_matrix, data.candicate_dis_matrix_batch)[i].view(compound_num_batch[i], protein_num_batch[i]) #让candicate_dis_matrix的维度与z一致
            #self.logging.info("3 z.shape:%s,protein_out_batched.shape:%s,  candicate_dis_matrix_batched.shape:%s "%(z.shape,protein_out_batched.shape,  candicate_dis_matrix_batched.shape))
            compound_out_batched_new, affinity_pred_B, prmsd_pred = self.MultiHeadAttention_dis_bias(z, z_mask, protein_out_batched, candicate_dis_matrix_batched) #获得包含protein和compound交互信息的compound single representation
            


            affinity_pred_B = affinity_pred_B.sigmoid() * 15
            prmsd_pred = prmsd_pred.sigmoid() * 20
            affinity_pred_B_list.append(affinity_pred_B)
            prmsd_pred_list.append(prmsd_pred)
            #self.logging.info(f"in recycling {_}, affinity_pred_B shape: {affinity_pred_B.shape}, prmsd_pred shape: {prmsd_pred.shape}, compound_out_batched_new shape: {compound_out_batched_new.shape}")
            #先算true_rmsd_上一轮的，再跟上面的算 prmsd_loss
            compound_out_new = self.rebatch(compound_out_batched_new, compound_batch)
            center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
            center_edge_attr = self.center_edge_embedding(center_edge_attr)
            # lig_node_attr 和 compound_out_new 都是node embedding 有问题，compound_out_new也包含protein信息（z）
            center_edge_attr = torch.cat([center_edge_attr, compound_out_new[center_edge_index[0], :]], -1)  #注意node_embedding初始维度为128维，只取了前24维？
            # in_irreps=f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            # compound_out_new = compound_out_new[:,:o3.Irreps(in_irreps).dim]  #保证node_embedding的维度是84维，但是原来是128维，需要调整的是in_irreps 
            global_pred = self.final_conv(compound_out_new, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
            tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
            rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
            tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
            tr_pred = tr_pred / tr_norm * self.tr_final_layer(tr_norm)
            rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
            rot_pred = rot_pred / rot_norm * self.rot_final_layer(rot_norm)
            # torsional components
            tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
            tor_bond_vec = data.candicate_conf_pos[tor_bonds[1]] - data.candicate_conf_pos[tor_bonds[0]]
            tor_bond_attr = compound_out_new[tor_bonds[0]] + compound_out_new[tor_bonds[1]]
            tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
            tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])
            tor_edge_attr = torch.cat([tor_edge_attr, compound_out_new[tor_edge_index[1], :],  #注意node_embedding初始维度为128维，只取了前24维？TODO
                                    tor_bond_attr[tor_edge_index[0], :]], -1)
            tor_pred = self.tor_bond_conv(compound_out_new, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                    out_nodes=data['compound'].edge_mask.sum(), reduce='mean')
            tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
            # batch_size = int(max(data_new['compound'].batch)) + 1
            # 步骤四
            next_candicate_conf_pos, next_candicate_dis_matrix = self.modify_conformer(data, tr_pred, rot_pred, tor_pred, batch_size)

            tmp_loss_list=self.get_tr_tor_rot_loss( data,tr_pred, rot_pred, tor_pred, batch_size)
            tr_tor_rot_loss_list.append(tmp_loss_list)

            data.candicate_conf_pos = next_candicate_conf_pos
            data.candicate_dis_matrix = next_candicate_dis_matrix
            data_new_list.append(copy.deepcopy(data.detach()))

            rmsd_loss_list.append(self.get_symmetry_rmsd(data).to(prmsd_pred.device))

        #self.logging.info(f"after point B, affinity_pred_A shape: {affinity_pred_A.shape}, affinity_pred_B_list len: {len(affinity_pred_B_list)}, prmsd_pred_list len: {len(prmsd_pred_list)}, rmsd_list len: {len(rmsd_list)}")
        return data_new_list, affinity_pred_A, affinity_pred_B_list, prmsd_pred_list, rmsd_loss_list,tr_tor_rot_loss_list

    def get_tr_tor_rot_loss(self, data, tr_update, rot_update, torsion_updates, batch_size):
        # 计算本轮recycling 初始构象到真实构象的tr,tor,rot，计算loss

        #根据tr,rot,tor update data的candicate_conf_pos，然后return
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())
        data_pos_batched = self.unbatch(data.candicate_conf_pos, data['compound'].batch)
        true_data_pos_batched=self.unbatch(data['compound'].pos,data['compound'].batch)
        compound_edge_index_batched = self.unbatch(data['compound', 'compound'].edge_index.T, data.compound_compound_edge_attr_batch)
        compound_rotate_edge_mask_batched = self.unbatch(data['compound'].edge_mask,data.compound_compound_edge_attr_batch)


        loss_list=[]
        for i in range(batch_size):

            opt_obj=OptimizeConformer(data_pos_batched[i], true_data_pos_batched[i],
                              compound_edge_index_batched[i],compound_rotate_edge_mask_batched[i])

            opt_tr,opt_torsion,opt_rotate,opt_rmsd=opt_obj.run()

            loss_list.append((nn.MSELoss(opt_torsion,torsion_updates),nn.MSELoss(opt_rotate,rot_update),nn.MSELoss(opt_tr,tr_update)))

        return loss_list




    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['compound'].batch.unsqueeze(0), torch.arange(len(data['compound'].batch)).to(data['compound'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['compound'].x.device), torch.zeros((data.num_graphs, 3)).to(data['compound'].x.device)
        center_pos.index_add_(0, index=data['compound'].batch, source=data.candicate_conf_pos)
        center_pos = center_pos / torch.bincount(data['compound'].batch).unsqueeze(1)

        edge_vec = data.candicate_conf_pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        #因为与时间无关，所以不需要sigma embedding
        # edge_sigma_emb = data['compound'].node_sigma_emb[edge_index[1].long()]
        # edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['compound', 'compound'].edge_index[:, data['compound'].edge_mask].long()
        bond_pos = (data.candicate_conf_pos[bonds[0]] + data.candicate_conf_pos[bonds[1]]) / 2
        bond_batch = data['compound'].batch[bonds[0]]
        edge_index = radius(data.candicate_conf_pos, bond_pos, self.lig_max_radius, batch_x=data['compound'].batch, batch_y=bond_batch)

        edge_vec = data.candicate_conf_pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
    #步骤四声明
    def rebatch(self, src_batch, batch):
        # 将分粒后padding的data根据batch聚为一体
        _size = degree(batch, dtype=torch.long).tolist() #获得split
        return torch.concat([src_batch[i].split(_size[i])[0] for i in range(len(_size))])

    def unbatch(self, src, batch):
        #将聚为一体的data根据batch split
        sizes = degree(batch, dtype=torch.long).tolist()
        return src.split(sizes)


    def modify_conformer(self, data, tr_update, rot_update, torsion_updates, batch_size):
        #根据tr,rot,tor update data的candicate_conf_pos，然后return
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())
        data_pos_batched = self.unbatch(data.candicate_conf_pos, data['compound'].batch)
        compound_edge_index_batched = self.unbatch(data['compound', 'compound'].edge_index.T, data.compound_compound_edge_attr_batch)
        compound_rotate_edge_mask_batched = self.unbatch(data['compound'].edge_mask,data.compound_compound_edge_attr_batch)


        new_pos_t=[]
        for i in range(batch_size):

            if torsion_updates is not None:
                rotate_edge_index=compound_edge_index_batched[i][compound_rotate_edge_mask_batched[i]]
                flexible_new_pos = modify_conformer_torsion_angles(data_pos_batched[i],
                                                                   rotate_edge_index,
                                                                   data['compound'].mask_rotate[i],
                                                                   torsion_updates)
                # TODO:这里先删掉原版diffdock代码中的align
                # R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
                # aligned_flexible_pos = flexible_new_pos @ R.T + t.T
                # candicate_conf_pos = aligned_flexible_pos

            else:
                flexible_new_pos=data_pos_batched[i]

            lig_center = torch.mean(flexible_new_pos, dim=0, keepdim=True)
            _new_pos = (flexible_new_pos - lig_center) @ rot_mat.permute(0, 2, 1)[i] + tr_update[i,:] + lig_center
            new_pos_t.append(_new_pos)

        candicate_conf_pos = torch.concat(new_pos_t)


        #更新candicate_dis_matrix
        data_pos_batched_new = self.unbatch(candicate_conf_pos, data['compound'].batch)
        protein_pos_batched = self.unbatch(data.node_xyz, data['protein'].batch)
        candicate_dis_matrix = torch.concat([torch.cdist(data_pos_batched_new[i], protein_pos_batched[i]).flatten() for i in range(batch_size)])
        return candicate_conf_pos,candicate_dis_matrix

    def get_symmetry_rmsd(self, data):

        # TODO:目前用一般的rmsd替代，由于对称rmsd有时计算会卡住
        tmp_rmsd_value = torch.sqrt(torch.average(torch.sum((data.candicate_conf_pos -data['compound'].pos) ** 2, axis=-1)))


        #calculate symmetry rmsd with true pocket
        data_candicate_pos_batched = self.unbatch(data.candicate_conf_pos, data['compound'].batch)
        data_compound_pos_batched = self.unbatch(data['compound'].pos, data['compound'].batch)
        RMSD = []
        for i in range(len(data.atomicnums)):
            tmp_time=time.time()
            if data.is_equivalent_native_pocket[i]:
                #rmsd_symm_value=rmsd.symmrmsd(
                #    data_compound_pos_batched[i].detach().cpu().numpy(),
                #    data_candicate_pos_batched[i].detach().cpu().numpy(),
                #    data.atomicnums[i],
                #    data.atomicnums[i],
                #    data.adjacency_matrix[i],
                #    data.adjacency_matrix[i],
                #)
                #print(tmp_rmsd_value,tmp_rmsd_value)
                RMSD.append(tmp_rmsd_value)
                #RMSD.append(rmsd_symm_value)
            #spend_time=time.time()-tmp_time
            #logger.info("atom num : %s"%(len(data.atomicnums[i])))
            #if spend_time>1:
            #    logger.info(spend_time)
            #if spend_time>9:
            #    import pdb
            #    pdb.set_trace()
        return torch.tensor(RMSD, dtype=torch.float).requires_grad_()

def get_model(mode, logging, device):
    if mode == 0:
        logging.info("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN")
        model = IaBNet_with_affinity(logging=logging).to(device)
    elif mode == 1:
        print("5 stack, readout2, pred dis map add self attention and GVP embed, compound model GIN, use fragmentation and return z")
        model = IaBNet_with_affinity(logging=logging, finetune=True).to(device)
    return model
