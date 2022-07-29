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
utils
"""

from __future__ import print_function
import sys
import os
from os.path import exists, dirname
import numpy as np
import pickle
import json
import time
import six

from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import roc_auc_score
import os
from os.path import join, exists
import pandas as pd
import numpy as np


 

def get_dataset_rk_list(dataset_name, data_path, task_names):
    """Return dataset according to the ``dataset_name``"""
    if dataset_name == 'PDL1':
        dataset = load_PDL1_dataset_list(data_path, task_names)
    elif dataset_name == 'KRAS_G12D':
        dataset = load_KRAS_G12D_dataset_list(data_path, task_names)
    elif dataset_name == 'MAT2A':
        dataset = load_MAT2A_dataset_list(data_path, task_names)
    elif dataset_name == 'HPK1_3':
        dataset = load_HPK1_3_dataset_list(data_path, task_names)
    elif dataset_name == 'HPK1_4':
        dataset = load_HPK1_4_dataset_list(data_path, task_names)
    elif dataset_name == 'SOS1':
        dataset = load_SOS1_dataset_list(data_path, task_names)
    elif dataset_name == 'ALK':
        dataset = load_ALK_dataset_list(data_path, task_names)
    elif dataset_name == 'BRAF':
        dataset = load_BRAF_dataset_list(data_path, task_names)
    elif dataset_name == 'BTK':
        dataset = load_BTK_dataset_list(data_path, task_names)
    elif dataset_name == 'CDK4':
        dataset = load_CDK4_dataset_list(data_path, task_names)
    elif dataset_name == 'EGFR':
        dataset = load_EGFR_dataset_list(data_path, task_names)
    elif dataset_name == 'FGFR1':
        dataset = load_FGFR1_dataset_list(data_path, task_names)
    elif dataset_name == 'JAK2':
        dataset = load_JAK2_dataset_list(data_path, task_names)
    elif dataset_name == 'NTRK1':
        dataset = load_NTRK1_dataset_list(data_path, task_names)
    elif dataset_name == 'VEGFR2':
        dataset = load_VEGFR2_dataset_list(data_path, task_names)
    elif dataset_name == 'PRMT5':
        dataset = load_PRMT5_dataset_list(data_path, task_names)
    else:
        raise ValueError('%s not supported' % dataset_name)
    return dataset

def load_KRAS_G12D_dataset_list(data_path, task_names=None):


    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='KRAS-G12D')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='KRAS-G12D')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        # data['label'] = labels[i]
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i] 
        # data['label'] = labels_1[i]       
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_HPK1_4_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='HPK1_4')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='HPK1_4')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_HPK1_3_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='HPK1_3')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='HPK1_3')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_SOS1_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='SOS1')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='SOS1')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_PDL1_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='PDL1')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='PDL1')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_MAT2A_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='MAT2A')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='MAT2A')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_ALK_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='ALK')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='ALK')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_BRAF_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='BRAF')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='BRAF')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_BTK_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='BTK')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='BTK')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_CDK4_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='CDK4')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='CDK4')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_EGFR_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='EGFR')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='EGFR')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_FGFR1_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='FGFR1')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='FGFR1')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_JAK2_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='JAK2')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='JAK2')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_NTRK1_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='NTRK1')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='NTRK1')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_VEGFR2_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='VEGFR2')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='VEGFR2')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)

def load_PRMT5_dataset_list(data_path, task_names=None):
    task_names = 'pvalue'
    # pvalue
    raw_path = join(data_path, '')
    te_csv_file = 'all_ndcg.test.csv'
    tr_csv_file = 'all_ndcg.train.csv'
    input_df = pd.read_csv(join(raw_path, te_csv_file), sep=',')
    input_df = input_df.loc[(input_df['target']=='PRMT5')]
    smiles_list = input_df['smiles']
    smiles_list = smiles_list.reset_index(drop = True)
    # input_df[task_names] = 9 - input_df[task_names].apply(np.log10)
    labels = input_df[task_names]
    labels = labels.reset_index(drop = True)
    group = input_df['group']
    group = group.reset_index(drop = True)
    status = input_df['split']
    status = status.reset_index(drop = True)
    input_df_1 = pd.read_csv(join(raw_path, tr_csv_file), sep=',')
    input_df_1 = input_df_1.loc[(input_df_1['target']=='PRMT5')]
    smiles_list_1 = input_df_1['smiles']
    smiles_list_1 = smiles_list_1.reset_index(drop = True)
    # input_df_1[task_names] = 9 - input_df_1[task_names].apply(np.log10)
    labels_1 = input_df_1[task_names]
    labels_1 = labels_1.reset_index(drop = True)
    group_1 = input_df_1['group']
    group_1 = group_1.reset_index(drop = True)
    status_1 = input_df_1['split']
    status_1 = status_1.reset_index(drop = True)
    #由于测试集不参与split，训练集和验证集split，所以在data_list中新加“status”列，包含train和test，在finetune_regr.py中将测试集单独分开
    data_list = []
    status_list = []
    data = {} #创建一个没用的SMILE用来填充batch，使得同一batch只包含一个group
    data_1 = {}
    data['smiles'] = 'CC'      
    data['label'] = np.array(-1)
    data['group'] = group_1[0]
    data_1['status'] = 'train'
    data_list.append(data)
    status_list.append(data_1)
    for i in range(len(smiles_list)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data['group'] = group[i]
        data_1['status'] = status[i]
        data_list.append(data)
        status_list.append(data_1)
    for i in range(len(smiles_list_1)):
        data = {}
        data_1 = {}
        data['smiles'] = smiles_list_1[i]        
        data['label'] = labels_1.values[i]
        data['group'] = group_1[i]
        data_1['status'] = status_1[i]
        data_list.append(data)
        status_list.append(data_1)
    return (data,status_list,data_list)