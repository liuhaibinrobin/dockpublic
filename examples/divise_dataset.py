#!/usr/bin/python                                                                                                                                  
#-*-coding:utf-8-*- 
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetune:to do some downstream task
"""

import os
from os.path import join, exists, basename
import argparse
import numpy as np
import copy
from sklearn.model_selection import GroupKFold
from utils_dataset import get_dataset_rk_list
import pandas as pd







def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    ### config for the body

    dataset_t = get_dataset_rk_list(args.dataset_name, args.data_path, ['pvalues'])
    status_list = dataset_t[1]
    data_list = dataset_t[2]
    N = len(data_list)
    dataset_tr, dataset_te = [], []
    for i in range(N):
        if status_list[i]['status'] == 'train':
            dataset_tr.append(data_list[i])
        else:
            dataset_te.append(data_list[i])
            
    label_list, smiles_l, group_l = [], [], []
    for i in range(len(dataset_tr)):
        label_list.append(dataset_tr[i]['label'])
        smiles_l.append(dataset_tr[i]['smiles'])
        group_l.append(dataset_tr[i]['group'])
    label_np = np.array(label_list)
    smiles_np = np.array(smiles_l)
    group_np = np.array(group_l)
    kf = GroupKFold(n_splits=3)
    train_dataset, valid_dataset = [], []

    if len(set(group_l)) == 1 or len(set(group_l)) == 2:
        leng = len(dataset_tr)
        print('length',leng)
        train_index = np.arange(leng).tolist()
        if 0 in train_index: #第0个是没用SMILE，不计入任何数据集中
            train_index.remove(0)
        for i in train_index:
            train_dataset.append(dataset_tr[i])
            valid_dataset.append(dataset_tr[i])
    else:
        if args.dataset_name == 'CDK4':
            leng = len(dataset_tr)
            print('length',leng)
            index = np.arange(leng).tolist()
            train_index = index[:400]
            val_index = index[400:]
            if 0 in train_index: #第0个是没用SMILE，不计入任何数据集中
                train_index.remove(0)
            else:
                val_index.remove(0)
            for i in train_index:
                train_dataset.append(dataset_tr[i])
            for j in val_index:
                valid_dataset.append(dataset_tr[j])
        else:
            for train_index , val_index in kf.split(smiles_np,label_np,group_np):
                ratio = int(len(train_index)/len(val_index))
                if ratio >= 2 and ratio <= 12:
                    print('ratio of train_index by test_index: %s ' %(len(train_index)/len(val_index)))
                    train_index = train_index.tolist()
                    val_index = val_index.tolist()
                    if 0 in train_index: #第0个是没用SMILE，不计入任何数据集中
                        train_index.remove(0)
                    else:
                        val_index.remove(0)
                    for i in train_index:
                        train_dataset.append(dataset_tr[i])
                    for j in val_index:
                        valid_dataset.append(dataset_tr[j])
                    break
    dic_tr, dic_te, dic_va = {}, {}, {}
    label_list_tr, smiles_l_tr, group_l_tr = [], [], []
    for i in range(len(train_dataset)):
        label_list_tr.append(train_dataset[i]['label'])
        smiles_l_tr.append(train_dataset[i]['smiles'])
        group_l_tr.append(train_dataset[i]['group'])
    dic_tr['smiles'] = smiles_l_tr
    dic_tr['label'] = label_list_tr
    dic_tr['group'] = group_l_tr
    label_list_va, smiles_l_va, group_l_va = [], [], []
    for i in range(len(valid_dataset)):
        label_list_va.append(valid_dataset[i]['label'])
        smiles_l_va.append(valid_dataset[i]['smiles'])
        group_l_va.append(valid_dataset[i]['group'])
    dic_va['smiles'] = smiles_l_va
    dic_va['label'] = label_list_va
    dic_va['group'] = group_l_va
    label_list_te, smiles_l_te, group_l_te = [], [], []
    for i in range(len(dataset_te)):
        label_list_te.append(dataset_te[i]['label'])
        smiles_l_te.append(dataset_te[i]['smiles'])
        group_l_te.append(dataset_te[i]['group'])
    dic_te['smiles'] = smiles_l_te
    dic_te['label'] = label_list_te
    dic_te['group'] = group_l_te
    df_tr = pd.DataFrame(dic_tr)
    df_va = pd.DataFrame(dic_va)
    df_te = pd.DataFrame(dic_te)
    df_tr.to_csv("%s/train.csv" % (args.out_data_path),sep=',',index=None,encoding='utf-8')
    df_te.to_csv("%s/test.csv" % (args.out_data_path),sep=',',index=None,encoding='utf-8')
    df_va.to_csv("%s/valid.csv" % (args.out_data_path),sep=',',index=None,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", 
            choices=['PDL1','HPK1_3','HPK1_4','KRAS_G12D','MAT2A','SOS1','ALK', 'BRAF', 'BTK', 'CDK4', 'EGFR', 'FGFR1', 'JAK2', 'NTRK1', 'VEGFR2','PRMT5'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--out_data_path", type=str, default=None)
    args = parser.parse_args()
    
    main(args)
