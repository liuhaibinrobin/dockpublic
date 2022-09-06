import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch.nn import Linear
import sys
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from GATv2 import GAT
from GINv2 import GIN

class TBMarginLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.register_buffer('margin', torch.tensor(margin, dtype=torch.float64))
    def forward(self, aff_pred, aff_true, right_pocket):
        if right_pocket:
            return (aff_pred-aff_true)**2
        else:
            return max(0, aff_pred-(aff_true-self.margin)**2)            


class NCIYesLoss(nn.Module):
    def __init__(self, is_eval, margin, margin_weight=1, dist_weight=1, nci_weight=1):
        super().__init__()
        self.eval = is_eval
        self.margin_weight = margin_weight 
        self.dist_weight = dist_weight
        self.nci_weight = nci_weight
        self.MarginLoss = TBMarginLoss(margin)
        self.DistLoss = nn.MSELoss()
        self.NCILoss = nn.BCELoss()
    
    def forward(self, aff_pred, y_pred, aff_true, y_true, right_pocket):
        loss = (
            self.margin_weight * self.MarginLoss(aff_pred, aff_true, right_pocket)+
            self.dist_weight * self.DistLoss(y_pred, y_true)+
            self.nci_weight * self.NCILoss(y_pred, y_true)        
        )
        return loss
        


class PairwiseLoss(nn.Module):
    def __init__(self, keep_rate=1, sigmoid_lambda=1,
                 ingrp_thr=0.3, outgrp_thr=9999, eval=False):
        super(PairwiseLoss, self).__init__()
        self.eval = eval
        self.register_buffer('keep_rate', torch.tensor(keep_rate, dtype=torch.float64))
        self.register_buffer('sigmoid_lambda', torch.tensor(sigmoid_lambda, dtype=torch.float64))
        self.register_buffer('ingrp_thr', torch.tensor(ingrp_thr, dtype=torch.float64))
        self.register_buffer('outgrp_thr', torch.tensor(outgrp_thr, dtype=torch.float64))

    def forward(self, pred, true, groupid):

        """
        Customized pairwise ranking loss.

        """
        # print('pred',pred)
        # print('true',true)
        # print('group_id',groupid)
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)
        if len(true.shape) == 1:
            true = true.unsqueeze(1)
        drop_rate = 1 - self.keep_rate
        true_tile_row = true.repeat((1, true.shape[0]))
        true_tile_col = torch.t(true_tile_row)
        assert (true_tile_row.shape == true_tile_col.shape)

        groupid_row = groupid.repeat((groupid.shape[0], 1))
        groupid_col = torch.t(groupid_row)

        pred_tile_row = pred.repeat((1, pred.shape[0]))
        pred_tile_col = torch.t(pred_tile_row)
        # diff = (true_tile_row - true_tile_col) / (torch.abs(true_tile_col) + 1e-4)
        diff = (true_tile_row - true_tile_col)

        pred_pair = torch.stack([pred_tile_row, pred_tile_col], dim=0)
        valid_ind = torch.logical_or(torch.logical_and(groupid_row == groupid_col, diff > self.ingrp_thr),
                                     diff > self.outgrp_thr)
        # print("len(valid_ind)",len(valid_ind))
        pred_pair_valid = pred_pair.masked_select(valid_ind).reshape(2, -1)
        pred_pair_diff = pred_pair_valid[0] - pred_pair_valid[1]
        reverse = torch.sum(pred_pair_diff > 0)
        ntotal = pred_pair_valid.shape[1] + 1e-8
        print("Total valid pairs: {:.3f}, reversed: {:.3f}, reverse ratio: {:.3f}".format(ntotal, reverse, reverse/ntotal))
        if drop_rate > 1e-4:
            pred_pair_valid_dropout = torch.nn.functional.dropout(pred_pair_valid[0], drop_rate) * self.keep_rate
            pred_pair_valid_ind = torch.logical_or(pred_pair_valid_dropout == pred_pair_valid[0],
                                                   pred_pair_valid_dropout != 0.0)

            pred_pair_valid = pred_pair_valid.masked_select(pred_pair_valid_ind).reshape(2, -1)
        # print(pred_pair_valid[1] - pred_pair_valid[0])
        loss = torch.sum(torch.log(1. + torch.exp(self.sigmoid_lambda * (pred_pair_valid[1] - pred_pair_valid[0]))))
        num = pred_pair_valid.shape[1] + 1e-8
        loss = torch.div(loss, num)
        return loss, reverse/ntotal