import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
# from helper_functions import *
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import glob
import torch
# %matplotlib inline
from data import get_data
from torch_geometric.loader import DataLoader
from metrics import *
from utils import *
from datetime import datetime
import logging
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB.PDBList import PDBList   # pip install biopython if import failure
import os.path as osp
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
import random
from model import *
from torch.utils.tensorboard import SummaryWriter
from data import TankBindDataSet_qsar

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# model.train()
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
                data.smiles = smiles_list[idx]
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
        self.lr = args.lr
        self.dropout = args.dropout_rate 
        self.compound_name = args.compound_name
        if args.model_mode == 'init' or args.model_mode == 'Halfbind':
            self.loss_fn = PairwiseLoss(sigmoid_lambda=0.001)
        else:
            self.loss_fn = PairwiseLoss()
        # self.loss_fn = PairwiseLoss()
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

class PairwiseLoss(nn.Module):
    def __init__(self, keep_rate=1., sigmoid_lambda=1.,
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

def main(args, _model_epoch_id):
    """
    finetune main function
    """
    ###load embedding
    Seed_everything(seed=42)

    torch.set_num_threads(1)
    # # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
    torch.multiprocessing.set_sharing_strategy('file_system')
    pre = f"/home/jovyan/TankBind/data/{args.compound_name}"
    addNoise=args.addNoise
    print(f"re-docking, using dataset: apr22_pdbbind_gvp_pocket_radius20 pred distance map.")
    print(f"compound feature based on torchdrug")
    add_noise_to_com = float(addNoise) if addNoise else None

    # compoundMode = 1 is for GIN model.
    new_dataset = TankBindDataSet_qsar(f"{pre}/dataset", add_noise_to_com=add_noise_to_com)
    # load compound features extracted using torchdrug.
    # new_dataset.compound_dict = torch.load(f"{pre}/compound_dict.pt")
    true_pocket_dict = {'SOS1':4, 'HPK1_3':1, 'HPK1_4':1, 'PRMT5':1}
    pdb_dict = {'SOS1':'6scm', 'HPK1_3':'7kac', 'HPK1_4':'7kac', 'PRMT5':'7s1s'}
    true_pocket = true_pocket_dict[args.compound_name] - 1
    pdb_name = pdb_dict[args.compound_name]
    new_dataset.data = new_dataset.data.query(f"pdb == '{pdb_name}_{true_pocket}'").reset_index(drop=True)
    d = new_dataset.data
    train_index = d.query("group =='train'").index.values
    train_after_warm_up = new_dataset[train_index]


    # all_pocket_test.compound_dict = torch.load(f"{pre}/compound_dict.pt")
    # info is used to evaluate the test set. 
    print(f"data point train: {len(new_dataset)}, train_after_warm_up: {len(train_after_warm_up)}")

    data_loader = DataLoader(new_dataset, batch_size=1, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=10)
    # import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
    device = 'cuda'
    model = get_model(1, logging, device)
    print('use baseline optimal validation model')
    if args.iter == 1:
        # modelFile = "/home/jovyan/TankBind/tankbind/result/2022_10_14_03_02/models/epoch_156.pt"
        modelFile = f"/home/jovyan/reproduce/train3times/2022_10_27_06_21_58/models/epoch_{_model_epoch_id}.pt"
    elif args.iter == 2:
        modelFile = f"/home/jovyan/reproduce/train3times/2022_10_31_02_27_31/models/epoch_{_model_epoch_id}.pt"
    elif args.iter == 3:
        modelFile = "/home/jovyan/TankBind/tankbind/result/2022_10_14_03_02/models/epoch_149.pt"
    args.init_model = modelFile
    model.load_state_dict(torch.load(modelFile, map_location=device))
    z_list = []
    z_mask_list = []
    z_list_t = []
    z_mask_list_t = []
    affinity_pred_list = []
    y_pred_list = []
    for data in tqdm(data_loader):
        data = data.to(device)
        # y_pred, affinity_pred, sum_embed, protein_out_batched, compound_out_batched = model(data)
        y_pred, affinity_pred, z_mask, z = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        z_mask_list.append(z_mask.detach().cpu()) #正确口袋！！！！非最优口袋
        z_list.append(z.detach().cpu())


    z_mask_dic = {}
    for i, line in tqdm(d.iterrows(), total=d.shape[0]):
        smiles = line['smiles']
        z_mask_dic[smiles] = z_mask_list[i]
    
    z_dic = {}
    for i, line in tqdm(d.iterrows(), total=d.shape[0]):
        smiles = line['smiles']
        z_dic[smiles] = z_list[i]


    cp_name = args.compound_name
    data_path = args.data_path
    embedding_dir = args.embedding_dir
    # with open(f'{embedding_dir}/{cp_name}_origin/{cp_name}_z_dic.pickle', 'rb') as f:
    #     z_dic = pickle.load(f)
    # with open(f'{embedding_dir}/{cp_name}_origin/{cp_name}_z_mask_dic.pickle', 'rb') as f1:
    #     z_mask_dic = pickle.load(f1)
    df_tr = pd.read_csv(f'{data_path}/{cp_name}/train.csv')
    df_te = pd.read_csv(f'{data_path}/{cp_name}/test.csv')
    df_va = pd.read_csv(f'{data_path}/{cp_name}/valid.csv')
    df_tr.drop_duplicates(['smiles'],inplace = True, ignore_index=True)
    df_te.drop_duplicates(['smiles'],inplace = True, ignore_index=True)
    df_va.drop_duplicates(['smiles'],inplace = True, ignore_index=True)
    df_tr['z'] = df_te['z'] = df_va['z'] = ''
    df_tr['z_mask'] = df_te['z_mask'] = df_va['z_mask'] = ''
    for index, row in df_tr.iterrows():
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_tr.iloc[index] = row
    for index, row in df_te.iterrows():
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_te.iloc[index] = row
    for index, row in df_va.iterrows():
        row['z'] = z_dic[row['smiles']]
        row['z_mask'] = z_mask_dic[row['smiles']]
        df_va.iloc[index] = row

    if '' in df_tr.values or '' in df_te.values or '' in df_va.values:
        raise ValueError("some smiles don't have embeddings")

    dataset = TBDataset(root=f'data/{args.compound_name}',df_tr=df_tr, df_te=df_te, df_va=df_va)
    te, tr, va = dataset.get_idx_split()
    dataset_test = dataset[:te]
    dataset_train = dataset[te:tr]
    dataset_val = dataset[tr:va]
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = TBNet(args).to(device)
    ######将TB的最后两层模型参数放入finetune模型中
    if args.model_mode == 'Tankbind':
        model_dict = model.state_dict()
        pretrainedd_dict = torch.load(args.init_model, map_location=device)
        model_dict['linear_energy.weight'] = pretrainedd_dict['linear_energy.weight']
        model_dict['linear_energy.bias'] = pretrainedd_dict['linear_energy.bias']
        model_dict['gate_linear.weight'] = pretrainedd_dict['gate_linear.weight']
        model_dict['gate_linear.bias'] = pretrainedd_dict['gate_linear.bias']
        model_dict['bias'] = pretrainedd_dict['bias']
        model.load_state_dict(model_dict)
    elif args.model_mode == 'Halfbind':
        model_dict = model.state_dict()
        pretrainedd_dict = torch.load(args.init_model, map_location=device)
        model_dict['linear_energy.weight'] = pretrainedd_dict['linear_energy.weight']
        model_dict['linear_energy.bias'] = pretrainedd_dict['linear_energy.bias']
        model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_loader_tr = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=3)
    data_loader_te = DataLoader(dataset_test, batch_size=150, shuffle=False, num_workers=3)
    data_loader_va = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=3)
    list_val_metric = []

    ######## start train
    # for epoch_id in range(args.max_epoch):
    #     model.train()
    #     scores = []
    #     list_loss = []
    #     for data in tqdm(data_loader_tr):
    #         y = data.y.to(device)
    #         batch_z = data.z.to(device)
    #         batch_z_size = data.z_size.to(device)
    #         batch_z_mask = data.z_mask.to(device)
    #         data.group_id = data.group_id.to(device)
    #         outputs = model(batch_z, batch_z_mask, batch_z_size)
    #         loss, score = model.my_loss(outputs, y, data.group_id)
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step() 
    #         scores.append(score)
    #         list_loss.append(loss)
    #     losst = torch.cat([torch.tensor([i]) for i in list_loss])
    #     losst = torch.mean(losst, dim=0)
    #     score = torch.cat([torch.tensor([[i]]) for i in scores])
    #     score = torch.mean(score, dim=0)
    #     print('TRAIN----> Epoch [{}/{}], score: {}, Loss: {:.4f}' .format(epoch_id+1, args.max_epoch, score, losst.item()))
    #     model.eval()
    #     with torch.no_grad():
    #         scores_va = []
    #         list_loss_va = []
    #         for data in tqdm(data_loader_va):
    #             y = data.y.to(device)
    #             batch_z = data.z.to(device)
    #             batch_z_size = data.z_size.to(device)
    #             batch_z_mask = data.z_mask.to(device)
    #             data.group_id = data.group_id.to(device)
    #             outputs = model(batch_z, batch_z_mask, batch_z_size)
    #             loss, score = model.my_loss(outputs, y, data.group_id)
    #             scores_va.append(score)
    #             list_loss_va.append(loss)
    #         losst_va = torch.cat([torch.tensor([i]) for i in list_loss_va])
    #         losst_va = torch.mean(losst_va, dim=0)
    #         score = torch.cat([torch.tensor([[i]]) for i in scores_va])
    #         score = torch.mean(score, dim=0)
    #         print('VALID----> Epoch [{}/{}], score: {}, Loss: {:.4f}' .format(epoch_id+1, args.max_epoch, score, losst_va.item()))
    #         list_val_metric.append(score)
    #         state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch_id}
    #         model_dir = args.model_dir + '/%s/%s/lr%s-batchsize%s-%s' % (args.compound_name, args.iter, args.lr, args.batch_size, args.model_mode)
    #         if not os.path.exists(model_dir):
    #             os.system(f"mkdir -p {model_dir}")
    #         epoch_model_dir = '%s/epoch-%s' % (model_dir, epoch_id + 1)
    #         torch.save(state, epoch_model_dir)
    #     model.eval() 
    #     with torch.no_grad():
    #         scores_te = []
    #         pred = []
    #         for data in tqdm(data_loader_te):
    #             y = data.y.to(device)
    #             batch_z = data.z.to(device)
    #             batch_z_size = data.z_size.to(device)
    #             batch_z_mask = data.z_mask.to(device)
    #             data.group_id = data.group_id.to(device)
    #             outputs = model(batch_z, batch_z_mask, batch_z_size)
    #             _, score = model.my_loss(outputs, y, data.group_id)
    #             pred.append(outputs)
    #             scores_te.append(score)
    #         score = torch.cat([torch.tensor([[i]]) for i in scores_te])
    #         score = torch.mean(score, dim=0)
    #         print('TE_EPOCH----> score: {}' .format(score))
    # best_epoch_id = np.argmax(list_val_metric)
    # print('-----------------------')
    # print('BEST_epoch_id:', best_epoch_id + 1)
    # model_dir = args.model_dir + '/%s/%s/lr%s-batchsize%s-%s/epoch-%s' % (args.compound_name, args.iter, args.lr, args.batch_size, args.model_mode, best_epoch_id + 1)
    # # model_dir = 'model_dir/PRMT5/1/lr0.001-batchsize32-init/epoch-3'
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
        df = pd.DataFrame()
        df['score'] = pred_list
        df['smiles'] = ''
        for index, row in df.iterrows():
            row['smiles'] = dataset_test[index].smiles
            df.iloc[index] = row
        csv_name = f'/home/jovyan/TankBind/outputs/{args.compound_name}'
        if not os.path.exists(csv_name):
                os.system(f"mkdir -p {csv_name}")
        # df.to_csv(f'{csv_name}/{args.lr}-{args.batch_size}-{args.model_mode}-{args.iter}.csv', index=False,encoding='utf-8')
        print('TEST----> score: {}' .format(score))
    return score
        

    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train your own TankBind model.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=25)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--compound_name", 
            choices=['PDL1','HPK1_3','HPK1_4','KRAS_G12D','MAT2A','SOS1','ALK', 'BRAF', 'BTK', 'CDK4', 'EGFR', 'FGFR1', 'JAK2', 'NTRK1', 'VEGFR2','PRMT5','PRMT5_ly'])
    parser.add_argument("--data_path", type=str, default='/home/jovyan/TankBind/data')
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--model_dir", type=str, default='model_dir')
    parser.add_argument("--embedding_dir", type=str, default='embedding')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--model_mode", choices=['init', 'Halfbind', 'Tankbind'], default='Tankbind')
    parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
    parser.add_argument("-d", "--data", type=str, default="0",
                        help="data specify the data to use.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size.")
    parser.add_argument("--sample_n", type=int, default=20000,
                        help="number of samples in one epoch.")
    parser.add_argument("--restart", type=str, default=None,
                        help="continue the training from the model we saved.")
    parser.add_argument("--addNoise", type=str, default=None,
                        help="shift the location of the pocket center in each training sample \
                        such that the protein pocket encloses a slightly different space.")

    pair_interaction_mask = parser.add_mutually_exclusive_group()
    # use_equivalent_native_y_mask is probably a better choice.
    pair_interaction_mask.add_argument("--use_y_mask", action='store_true',
                        help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                        real_y_mask=True if it's the native pocket that ligand binds to.")
    pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true',
                        help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                        real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")

    parser.add_argument("--use_affinity_mask", type=int, default=0,
                        help="mask affinity in loss evaluation based on data.real_affinity_mask")
    parser.add_argument("--affinity_loss_mode", type=int, default=1,
                        help="define which affinity loss function to use.")
    parser.add_argument("--decoy_gap", type=int, default=1,
                        help="define deocy gap used in args.affinity_loss_mode=1")

    parser.add_argument("--pred_dis", type=int, default=1,
                        help="pred distance map or predict contact map.")
    parser.add_argument("--posweight", type=int, default=8,
                        help="pos weight in pair contact loss, not useful if args.pred_dis=1")

    parser.add_argument("--relative_k", type=float, default=0.01,
                        help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
    parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                        help="define how the relative_k changes over epochs")
    parser.add_argument("--warm_up_epochs", type=int, default=15,
                        help="used in combination with relative_k_mode.")
    parser.add_argument("--data_warm_up_epochs", type=int, default=0,
                        help="option to switch training data after certain epochs.")

    # parser.add_argument("--resultFolder", type=str, default="../",
    #                     help="information you want to keep a record.")
    parser.add_argument("--resultFolder", type=str, default="./result/",
                        help="information you want to keep a record.")
    parser.add_argument("--label", type=str, default="",
                        help="information you want to keep a record.")
    args = parser.parse_args()
    dic = {}
    for i in range(100, 200):
        dic[i] = float(np.array(main(args, i)))
    df_t = pd.DataFrame.from_dict(dic, orient='index',columns=['score'])
    df_t['epoch_index'] = [i for i in range(100, 200)]
    if args.iter == 1:
        df_t['model_dir'] = [f"/home/jovyan/reproduce/train3times/2022_10_27_06_21_58/models/epoch_{i}.pt" for i in range(100, 200)]
    elif args.iter == 2:
        df_t['model_dir'] = [f"/home/jovyan/reproduce/train3times/2022_10_31_02_27_31/models/epoch_{i}.pt" for i in range(100, 200)]
    csv_name = f'/home/jovyan/TankBind/outputs/{args.compound_name}'
    df_t.to_csv(f'{csv_name}/{args.compound_name}-{args.iter}.csv', index=False,encoding='utf-8')

