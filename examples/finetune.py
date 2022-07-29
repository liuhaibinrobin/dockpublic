# %% [markdown]
# # Overview
# 
# In this example, we are going to predict the binding pose of a small molecule to a target protein.
# 
# PDB 6HD6 is an interesting case, in which two small molecule drugs bind to the same protein target on different binding sites.
# 
# One small molecule, STI-571, is the famous drug Imatinib. The other small molecule, FYH, is asciminib.
# 
# This PDB is also used as an exmaple by Regina Barzilay and Hannes Stärk in MLDD workshop https://www.mldd-workshop.org/home
# 
# https://www.rcsb.org/structure/6HD6

tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB.PDBList import PDBList   # pip install biopython if import failure
import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from IPython import embed
import argparse
from Bio.PDB import PDBParser
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data import HeteroData, Data
import rdkit.Chem as Chem    # conda install rdkit -c rdkit if import failure.
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.

class TBDataset(Dataset):
    def __init__(self,root, df_tr=None, df_te=None, df_va=None,
                 transform=None, pre_transform=None, pre_filter=None):
        '''
        '''
        self.compound_name = args.compound_name
        self.trainindx = 0
        self.testindx = 0
        self.valindx = 0
        self.test_df = df_te
        self.train_df = df_tr
        self.val_df = df_va
        self.testindx = len(self.test_df)
        self.expected_test_count = len(self.test_df)
        self.trainindx = len(self.train_df) +  self.testindx
        self.expected_train_count = len(self.train_df)
        self.valindx = len(self.val_df) + self.trainindx
        self.expected_val_count = len(self.val_df) 
        super(TBDataset, self).__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_dir)

    @property
    def processed_file_names(self):
        return ['data_0.pt']
    def process(self):
        tasks = 'label'
        idx_list = []
        i = 0
        for df in [self.test_df, self.train_df, self.val_df]:
            # print(df)
            smiles_list = df.loc[:, 'smiles'].tolist() if 'smiles' in set(df.columns) else df.loc[:, 'mol'].tolist() if 'mol' in set(df.columns) else None
            assert smiles_list is not None, "Cannot find the smiles column in the data file"
            assert tasks is not None, "Cannot determine the target tasks due to unsupported dataset"
            self.ntasks = len(tasks)
            labels_list = df.loc[:, tasks]
            group_id_list = df.loc[:, 'group']
            # embed_list = df.loc[:, 'embed']
            z_list = df.loc[:, 'z']
            z_mask_list = df.loc[:, 'z_mask']
            
            labels_list = labels_list.to_numpy()
            group_id_list = group_id_list.to_numpy()
            # embed_list = embed_list.to_numpy()
            z_list = z_list.to_numpy()
            z_mask_list = z_mask_list.to_numpy()
            group_dic = {}
            tmp = 0
            for idx in tqdm(range(len(smiles_list))):
                label, group_id_str, z, z_mask = labels_list[idx], group_id_list[idx], z_list[idx], z_mask_list[idx]
                # embed, label, group_id_str = embed_list[idx], labels_list[idx], group_id_list[idx]
                data = HeteroData()
                # data.x = embed
                # data.x = data.x.reshape(1,128)
                data.z = z.view(-1,128)
                data.z_size = data.z.shape[0]
                data.z_mask = z_mask.view(-1)
                data.y = torch.from_numpy(np.array(label)).to(torch.double)
                data.reverse = 0
                data.reverse = torch.tensor([data.reverse],dtype = torch.int64) 
                group_id = group_dic.get(group_id_str)
                if group_id is None:
                    group_id = tmp
                    group_dic[group_id_str] = tmp
                    tmp += 1
                data.group_id =  torch.from_numpy(np.array(group_id)).to(torch.double)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
            idx_list.append(i)

            
        self.testindx, self.trainindx, self.valindx = idx_list
        self.expected_test_count = self.testindx
        self.trainindx = len(self.train_df) +  self.testindx
        self.expected_train_count = len(self.train_df)
        self.valindx = len(self.val_df) + self.trainindx
        self.expected_val_count = len(self.val_df)
        
    def len(self):
        return self.expected_test_count + self.expected_train_count \
            + self.expected_val_count
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        data.idx = idx
        return  data
    def get_idx_split(self):
        return self.expected_test_count, \
            self.expected_test_count + self.expected_train_count, \
            self.expected_test_count + self.expected_train_count + self.expected_val_count

class TBNet(nn.Module):
    def __init__(self, args):
        super(TBNet, self).__init__()
        self.downstream_n_layers = args.layer_n
        self.lr = args.lr
        self.dropout = args.dropout_rate
        self.compound_name = args.compound_name
        self.loss_fn = PairwiseLoss()
        self.gate_linear = nn.Linear(128, 1)
        self.linear_energy = nn.Linear(128, 1)
        self.leaky = torch.nn.LeakyReLU()
        self.bias = torch.nn.Parameter(torch.ones(1))

        
    def forward(self, z, z_mask, z_size):
        pair_energy = (self.gate_linear(z).sigmoid() * self.linear_energy(z)).squeeze(-1) * z_mask
        pair_energy_split = pair_energy.split(z_size.tolist())
        pair_energy_split_sum = torch.stack([i.sum(axis=-1) for i in pair_energy_split])
        affinity_pred = self.leaky(self.bias + (pair_energy_split_sum))
        return affinity_pred

    def my_loss(self, pred, target, groupid):
        is_valid = torch.logical_not(torch.isnan(target))
        return self.loss_fn(pred[is_valid], target[is_valid], groupid)

class GEMNet(nn.Module):
    def __init__(self, args):
        super(TBNet, self).__init__()
        self.downstream_n_layers = args.layer_n
        self.lr = args.lr
        self.dropout = args.dropout_rate
        self.compound_name = args.compound_name
        self.loss_fn = PairwiseLoss()
        self.downstream_out = nn.Linear(128, 1)
        ffn_layers=[FFN(128, self.dropout) for _ in range(args.layer_n)]
        self.layers = nn.ModuleList(ffn_layers)

        
    def forward(self, x):
        for ffn_layer in self.layers:
            x = ffn_layer(x)
        x = self.downstream_out(x)
        return x

    def my_loss(self, pred, target, groupid):
        is_valid = torch.logical_not(torch.isnan(target))
        return self.loss_fn(pred[is_valid], target[is_valid], groupid)

class PairwiseLoss(nn.Module):
    def __init__(self, keep_rate=1., sigmoid_lambda=0.05,
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

def main(args):
    """
    finetune main function
    """
    ###load embedding
    cp_name = args.compound_name
    data_path = args.data_path
    with open(f'embedding/{cp_name}/{cp_name}_z_dic.pickle', 'rb') as f:
        z_dic = pickle.load(f)
    with open(f'embedding/{cp_name}/{cp_name}_z_mask_dic.pickle', 'rb') as f1:
        z_mask_dic = pickle.load(f1)
    # with open(f'embedding/{cp_name}/{cp_name}_sum_embedding_dic.pickle', 'rb') as f1:
    #     SOS1_sum_embedding_dic = pickle.load(f1)
    df_tr = pd.read_csv(f'{data_path}/{cp_name}/train.csv')
    df_te = pd.read_csv(f'{data_path}/{cp_name}/test.csv')
    df_va = pd.read_csv(f'{data_path}/{cp_name}/valid.csv')
    # df_tr['embed'] = df_te['embed'] = df_va['embed'] = ''
    df_tr['z'] = df_te['z'] = df_va['z'] = ''
    df_tr['z_mask'] = df_te['z_mask'] = df_va['z_mask'] = ''
    for index, row in df_tr.iterrows():
        # row['embed'] = SOS1_sum_embedding_dic[row['smiles']]
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_tr.iloc[index] = row
    for index, row in df_te.iterrows():
        # row['embed'] = SOS1_sum_embedding_dic[row['smiles']]
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_te.iloc[index] = row
    for index, row in df_va.iterrows():
        # row['embed'] = SOS1_sum_embedding_dic[row['smiles']]
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_va.iloc[index] = row
    if '' in df_tr.values or '' in df_te.values or '' in df_va.values:
        raise ValueError("some smiles don't have embeddings")

    dataset = TBDataset(root='data/SOS1',df_tr=df_tr, df_te=df_te, df_va=df_va)
    te, tr, va = dataset.get_idx_split()
    dataset_test = dataset[:te]
    dataset_train = dataset[te:tr]
    dataset_val = dataset[tr:va]
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = TBNet(args).to(device)
    ######将TB的最后两层模型参数放入finetune模型中
    model_dict = model.state_dict()
    pretrainedd_dict = torch.load("../saved_models/self_dock.pt", map_location=device)
    model_dict['linear_energy.weight'] = pretrainedd_dict['linear_energy.weight']
    model_dict['linear_energy.bias'] = pretrainedd_dict['linear_energy.bias']
    model_dict['gate_linear.weight'] = pretrainedd_dict['gate_linear.weight']
    model_dict['gate_linear.bias'] = pretrainedd_dict['gate_linear.bias']
    model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_loader_tr = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=3)
    data_loader_te = DataLoader(dataset_test, batch_size=120, shuffle=False, num_workers=3)
    data_loader_va = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=3)
    list_val_metric = []

    ######## start train
    # for epoch_id in range(args.max_epoch):
    #     model.train()
    #     scores = []
    #     for data in tqdm(data_loader_tr):
    #         y = data.y.to(device)
    #         z = data.z.to(device)
    #         z_mask = data.z_mask.to(device)
    #         data.group_id = data.group_id.to(device)
    #         outputs = model(z, z_mask)
    #         loss, score = model.my_loss(outputs, y, data.group_id)
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step() 
    #         scores.append(score)
    #     score = torch.cat([torch.tensor([[i]]) for i in scores])
    #     score = torch.mean(score, dim=0)
    #     print('TRAIN----> Epoch [{}/{}], score: {}, Loss: {:.4f}' .format(epoch_id+1, args.max_epoch, score, loss.item()))
    #     model.eval()
    #     with torch.no_grad():
    #         scores_va = []
    #         for data in tqdm(data_loader_va):
    #             y = data.y.to(device)
    #             z = data.z.to(device)
    #             z_mask = data.z_mask.to(device)
    #             data.group_id = data.group_id.to(device)
    #             outputs = model(z, z_mask)
    #             _, score = model.my_loss(outputs, y, data.group_id)
    #             scores_va.append(score)
    #         score = torch.cat([torch.tensor([[i]]) for i in scores_va])
    #         score = torch.mean(score, dim=0)
    #         print('VALID----> Epoch [{}/{}], score: {}, Loss: {:.4f}' .format(epoch_id+1, args.max_epoch, score, loss.item()))
    #         list_val_metric.append(score)
    #         state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch_id}
    #         model_dir = args.model_dir + '/%s/lr%s-drop%s-config%s-epoch%s' % (args.compound_name, args.lr, args.dropout_rate, args.layer_n, epoch_id)
    #         torch.save(state, model_dir)
    # best_epoch_id = np.argmax(list_val_metric)
    # print('-----------------------')
    # print('BEST_epoch_id:', best_epoch_id)
    # model_dir = args.model_dir + '/%s/lr%s-drop%s-config%s-epoch%s' % (args.compound_name, args.lr, args.dropout_rate, args.layer_n, best_epoch_id+1)
    # checkpoint = torch.load(model_dir)
    # model.load_state_dict(checkpoint['net'])

    ######## start test
    model.eval() 
    with torch.no_grad():
        scores_te = []
        pred = []
        for data in tqdm(data_loader_te):
            y = data.y.to(device)
            batch_z = data.z.to(device)
            batch_z_size = data.z_size.to(device)
            batch_z_mask = data.z_mask.to(device)
            data.group_id = data.group_id.to(device)
            outputs = model(batch_z, batch_z_mask, batch_z_size)
            _, score = model.my_loss(outputs, y, data.group_id)
            pred.append(outputs)
            scores_te.append(score)
        score = torch.cat([torch.tensor([[i]]) for i in scores_te])
        score = torch.mean(score, dim=0)
        pred = torch.cat(pred, dim=0)
        pred_cpu = pred.cpu()
        pred_numpy = pred_cpu.numpy()
        pred_list = []
        for i in pred_numpy:
            pred_list.append(i)
        smile_path = './data/%s/test.csv' % (args.compound_name)
        df = pd.read_csv(smile_path,sep=',')
        df['score'] = pred_list
        # df = df.rename(columns={'smiles': 'SMILES'})
        df = df.drop(columns=['group'])
        df = df.drop(columns=['label'])
        df.to_csv('./SAR-interface/output/%s.csv'%(args.compound_name), index=False,encoding='utf-8')
        print('TEST----> score: {}' .format(score))
        

    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=15)
    parser.add_argument("--gpu_id", type=int, default=6)
    parser.add_argument("--compound_name", 
            choices=['PDL1','HPK1_3','HPK1_4','KRAS_G12D','MAT2A','SOS1','ALK', 'BRAF', 'BTK', 'CDK4', 'EGFR', 'FGFR1', 'JAK2', 'NTRK1', 'VEGFR2','PRMT5'])
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str, default='model_dir')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--layer_n", type=int, default=3)
    args = parser.parse_args()
    
    main(args)
# %%
