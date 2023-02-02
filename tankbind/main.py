import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
# from helper_functions import *
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import glob
import torch
torch_module = torch.nn.Module 
# %matplotlib inline
from model import *    ## add to avoid torchdrug & e3nn bug
torch.nn.Module = torch_module
from data import get_data
from torch_geometric.loader import DataLoader
from metrics import *
from utils import *
from datetime import datetime
import logging
import sys
import argparse
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler
import random
import math
from torch.utils.tensorboard import SummaryWriter
from utils import OptimizeConformer






def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

Seed_everything(seed=42)

parser = argparse.ArgumentParser(description='Train your own TankBind model.')
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
parser.add_argument("--use_contact_loss", type=int, default=0,
                    help="whether to upgrade contact loss during training, 0 means both, other means only contact loss")
parser.add_argument("--contact_loss_mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                    help="choose contact loss mode, 0 means dis^2, 1 means e^dis, 2 means 2^dis, 3,4,5 means dis^3,4,5")

parser.add_argument("--use_weighted_rmsd_loss", type=bool, default=False,
                    help="whether to change contact weight according to distance")
args = parser.parse_args()


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

arg_hash = str(abs(hash(str(args))))[0:10]

writer = SummaryWriter(f"./logs/{timestamp}_{args.label}")

logger = logging.getLogger()
# 指定logger输出格式
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
handler = logging.FileHandler(f'{timestamp}.log')
#handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式


# 为logger添加的日志处理器
logger.addHandler(handler)
# 指定日志的最低输出级别，默认为WARN级别
logger.setLevel(logging.INFO)


logging.info(f'''\
{' '.join(sys.argv)}
{timestamp}
{args.label}
--------------------------------
''')
pre = f"{args.resultFolder}/{timestamp}"
os.system(f"mkdir -p {pre}/models")
os.system(f"mkdir -p {pre}/results")
os.system(f"mkdir -p {pre}/src")
os.system(f"cp *.py {pre}/src/")
os.system(f"cp -r gvp {pre}/src/")


torch.set_num_threads(1)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')




train, train_after_warm_up, valid, test, all_pocket_test, all_pocket_valid, info, info_va = get_data(args.data, logging, addNoise=args.addNoise)
logging.info(f"data point train: {len(train)}, train_after_warm_up: {len(train_after_warm_up)}, valid: {len(valid)}, test: {len(test)}")

num_workers = 10
# sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
sampler = RandomSampler(train, replacement=False, num_samples=len(train)) #训练数据不足2w，全部口袋时要换回来 TODO
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair','candicate_dis_matrix',"compound_compound_edge_attr"], sampler=sampler, pin_memory=False, num_workers=num_workers)
sampler2 = RandomSampler(train_after_warm_up, replacement=False, num_samples=args.sample_n)
train_after_warm_up_loader = DataLoader(train_after_warm_up, batch_size=args.batch_size, follow_batch=['x', 'compound_pair','candicate_dis_matrix'], sampler=sampler2, pin_memory=False, num_workers=num_workers)
valid_batch_size = test_batch_size = 4
valid_loader = DataLoader(valid, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair','candicate_dis_matrix'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=test_batch_size, follow_batch=['x', 'compound_pair','candicate_dis_matrix'], shuffle=False, pin_memory=False, num_workers=num_workers)
all_pocket_test_loader = DataLoader(all_pocket_test, batch_size=2, follow_batch=['x', 'compound_pair','candicate_dis_matrix'], shuffle=False, pin_memory=False, num_workers=4)
all_pocket_valid_loader = DataLoader(all_pocket_valid, batch_size=2, follow_batch=['x', 'compound_pair','candicate_dis_matrix'], shuffle=False, pin_memory=False, num_workers=4)
# import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
from model import *
device = 'cuda'
model = get_model(args.mode, logger, device)
if args.restart:
    model.load_state_dict(torch.load(args.restart))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# model.train()
if args.pred_dis:
    if args.use_weighted_rmsd_loss:
        contact_criterion = weighted_rmsd_loss
        print('use weighted rmsd loss!!!')
        pred_dis = True
    else:   
        contact_criterion = nn.MSELoss()
        pred_dis = True       
else:
    contact_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))

affinity_criterion = nn.MSELoss()
# contact_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))

metrics_list = []
valid_metrics_list = []
test_metrics_list = []

best_auroc = 0
best_f1_1 = 0
epoch_not_improving = 0
data_warmup_epochs = args.data_warm_up_epochs
warm_up_epochs = args.warm_up_epochs
logging.info(f"warming up epochs: {warm_up_epochs}, data warming up epochs: {data_warmup_epochs}")


global_steps_train = 0
global_steps_val = 0
global_steps_test = 0
global_samples_train = 0
global_samples_val = 0
global_samples_test = 0
for epoch in range(10000):
    model.train()
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_A_pred_list = []
    affinity_B_pred_list = []
    rmsd_pred_list = []
    prmsd_pred_list = []
    train_result_list = []
    epoch_loss_contact = 0.0
    epoch_loss_contact_5A = 0.0
    epoch_num_nan_contact_5A = 0
    epoch_loss_contact_10A = 0.0
    epoch_num_nan_contact_10A = 0
    epoch_loss_affinity_A = 0.0
    epoch_loss_affinity_B = 0.0
    epoch_loss_rmsd = 0.0
    epoch_loss_prmsd = 0.0

    epoch_rmsd_recycling_0_loss=0
    epoch_rmsd_recycling_1_loss=0
    epoch_rmsd_recycling_9_loss=0
    epoch_rmsd_recycling_19_loss=0
    epoch_rmsd_recycling_39_loss=0

    epoch_tr_loss =0
    epoch_rot_loss =0
    epoch_tor_loss=0
    #先修改数据为只有真口袋的，验证我们的方法，后续改回全部口袋 TODO
    # if epoch < data_warmup_epochs:
    #     data_it = tqdm(train_loader)
    # else:
    #     data_it = tqdm(train_after_warm_up_loader)
    data_it = tqdm(train_loader)
    for data in data_it:
        data = data.to(device)
        optimizer.zero_grad()
        affinity_pred_A, affinity_pred_B_list, prmsd_list,pred_result_list= model(data)
        sample_num=len(data.pdb)

        #记录每个样本的学习信息
        data_new_pos_batched_list=[]
        for i in range(sample_num):# data_new_pos_batched_list=[[]]*sample_num  #这么写会有严重的bug  所有[]其实都指向了一个[]
            data_new_pos_batched_list.append([])
        candicate_conf_pos_batched=pred_result_list[0][5] #初始坐标
        for i in range(len(candicate_conf_pos_batched)):
            data_new_pos_batched_list[i].append(candicate_conf_pos_batched[i].detach().cpu().numpy().tolist())
        for pred_result in pred_result_list:#每个pred_result是一个迭代轮次的batch中的全部样本
            #data_new_pos_batched:  bs*pos..
            next_candicate_conf_pos_batched=pred_result[3]
            for i in range(len(next_candicate_conf_pos_batched)):
                data_new_pos_batched_list[i].append(next_candicate_conf_pos_batched[i].detach().cpu().numpy().tolist())
        # print(data.y.sum(), y_pred.sum())
        # print(data_new.is_equivalent_native_pocket, rmsd_list[2].shape) 训练时出现晶体构象不是is_equivalent_native_pocket情况，暂时无法打印rmsd_list
        for i in range(len(data_new_pos_batched_list)): #记录每个样本的后面recycling构象更新情况，第0个构象是原始构象 全口袋时删除  TODO
            train_result_list.append([data.pdb[i], data_new_pos_batched_list[i], affinity_pred_A[i].detach().cpu().numpy(), affinity_pred_B_list[-1][i].detach().cpu().numpy() , prmsd_list[-1][i].detach().cpu().numpy()])

        # RMSD loss
        #dis_map
        y = data.y
        affinity = data.affinity
        dis_map = data.dis_map
        y_pred = candicate_dis_matrix=pred_result_list[-1][4]
        if args.use_equivalent_native_y_mask:
            y_pred = y_pred[data.equivalent_native_y_mask]
            y = y[data.equivalent_native_y_mask]
            dis_map = dis_map[data.equivalent_native_y_mask]
            for i in range(len(prmsd_list)):
                prmsd_list[i] = prmsd_list[i][data.is_equivalent_native_pocket]
        elif args.use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
            dis_map = dis_map[data.real_y_mask]
        if args.pred_dis:
            import pdb
            pdb.set_trace()
            #tr,rot,tor loss
            tr_loss=0
            rot_loss=0
            tor_loss=0
            for pred_result in pred_result_list:

                tr_pred, rot_pred, tor_pred, _, _, current_candicate_conf_pos_batched = pred_result

                data_groundtruth_pos_batched = model.unbatch(data['compound'].pos, data['compound'].batch)
                compound_edge_index_batched = self.unbatch(data['compound', 'compound'].edge_index.T,data.compound_compound_edge_attr_batch)
                compound_rotate_edge_mask_batched = self.unbatch(data['compound'].edge_mask,data.compound_compound_edge_attr_batch)

                for i in range(len(data_groundtruth_pos_batched)):
                    rotate_edge_index = compound_edge_index_batched[i][compound_rotate_edge_mask_batched[i]] - sum(
                        ligand_atom_sizes[:i + 1])  # 把edge_id 从batch计数转换为样本内部计数
                    OptimizeConformer_obj = OptimizeConformer( current_pos=current_candicate_conf_pos_batched[i],
                                                               ground_truth_pos=data_groundtruth_pos_batched[i],
                                                               rotate_edge_index=rotate_edge_index,
                                                               mask_rotate=data['compound'].mask_rotate[i])


                    opt_tr,opt_rotate, opt_torsion, opt_rmsd=OptimizeConformer_obj.run()
                    tr_loss+=F.mse_loss(tr_pred,opt_tr)
                    rot_loss+=F.mse_loss(rot_pred,opt_rotate)
                    tor_loss+=F.mse_loss(rot_pred, opt_torsion)


            #rmsd_loss
            rmsd_list=[]
            for pred_result in pred_result_list:
                next_candicate_conf_pos_batched=pred_result[3]
                data_groundtruth_pos_batched = data.unbatch(data['compound'].pos, data['compound'].batch)
                tmp_list=[]
                for i in range(len(data_groundtruth_pos_batched)):
                    tmp_list.append(torch.sqrt(F.mse_loss(next_candicate_conf_pos_batched[i],data_groundtruth_pos_batched[i], reduction="mean")))
                rmsd_list.append(torch.tensor(tmp_list, dtype=torch.float).requires_grad_())
            rmsd_loss = torch.stack(rmsd_list).mean() #TODO:
            candicate_conf_pos = pred_result_list[0][5]  # 初始坐标
            rmsd_recycling_0_loss = torch.sqrt(F.mse_loss(candicate_conf_pos, data['compound'].pos, reduction="mean"))
            rmsd_recycling_1_loss = rmsd_list[1] if len(rmsd_list) >= 1 else torch.tensor([0]).to(y_pred.device)
            rmsd_recycling_9_loss = rmsd_list[9] if len(rmsd_list) >=9 else torch.tensor([0]).to(y_pred.device)
            rmsd_recycling_19_loss = rmsd_list[19] if len(rmsd_list) >= 19 else torch.tensor([0]).to(y_pred.device)
            rmsd_recycling_39_loss = rmsd_list[39] if len(rmsd_list) >= 39 else torch.tensor([0]).to(y_pred.device)

            if args.use_weighted_rmsd_loss:
                prmsd_loss = torch.stack([contact_criterion(rmsd_list[i], prmsd_list[i], args.contact_loss_mode) for i in range(len(prmsd_list))]).mean() if len(prmsd_list) > 0 else torch.tensor([0]).to(y_pred.device)
                contact_loss = contact_criterion(y_pred, dis_map, args.contact_loss_mode) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)
            else:
                prmsd_loss = torch.stack([contact_criterion(rmsd_list[i], prmsd_list[i]) for i in range(len(prmsd_list))]).mean() if len(prmsd_list) > 0 else torch.tensor([0]).to(y_pred.device)
                contact_loss = contact_criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)

            with torch.no_grad():
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=5)):
                    epoch_num_nan_contact_5A += len(y_pred)
                    contact_loss_cat_off_rmsd_5 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_5 = cut_off_rmsd(y_pred, dis_map, cut_off=5)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=10)):
                    epoch_num_nan_contact_10A += len(y_pred)
                    contact_loss_cat_off_rmsd_10 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_10 = cut_off_rmsd(y_pred, dis_map, cut_off=10)
        else:
            contact_loss = contact_criterion(y_pred, y) if len(y) > 0 else torch.tensor([0]).to(y.device)
            y_pred = y_pred.sigmoid()


        #relative_k
        if args.restart is None:
            base_relative_k = args.relative_k
            if args.relative_k_mode == 0:
                # increase exponentially. reach base_relative_k at epoch = warm_up_epochs.
                relative_k = min(base_relative_k * (2**epoch) / (2**warm_up_epochs), base_relative_k)
            if args.relative_k_mode == 1:
                # increase linearly
                relative_k = min(base_relative_k / warm_up_epochs * epoch, base_relative_k)
        else:
            relative_k = args.relative_k


        #affinity_loss
        if args.use_affinity_mask:
            affinity_pred_A = affinity_pred_A[data.real_affinity_mask]
            affinity = affinity[data.real_affinity_mask]
        if args.affinity_loss_mode == 0:
            affinity_loss_A = relative_k * affinity_criterion(affinity_pred_A, affinity)
            affinity_loss_B = relative_k * torch.stack([affinity_criterion(affinity_pred_B_list[i], affinity) for i in range(len(affinity_pred_B_list))],0).mean()
        elif args.affinity_loss_mode == 1:
            native_pocket_mask = data.is_equivalent_native_pocket
            affinity_loss_A = relative_k * my_affinity_criterion(affinity_pred_A,
                                                                affinity, 
                                                                native_pocket_mask, decoy_gap=args.decoy_gap)
            affinity_loss_B = relative_k * torch.stack([my_affinity_criterion(affinity_pred_B_list[i], affinity, native_pocket_mask, decoy_gap=args.decoy_gap) for i in range(len(affinity_pred_B_list))],0).mean()



        # print(contact_loss.item(), affinity_loss_A.item())

        #total loss
        if args.use_contact_loss == 0:
            loss=tr_loss+tor_loss+rot_loss #TODO:debug阶段
            #loss = prmsd_loss.double() + rmsd_loss.double() + affinity_loss_A + affinity_loss_B
        else:
            loss = contact_loss.float()
            loss = loss.requires_grad_(True)
        # logging.info(f"prmsd_loss: {prmsd_loss.detach().cpu()}, rmsd_loss: {rmsd_loss.detach().cpu()}, affinity_loss_A: {affinity_loss_A.detach().cpu()}, affinity_loss_B: {affinity_loss_B.detach().cpu()}")
        loss.backward()
        optimizer.step()

        #记录日志
        epoch_tr_loss+=len(rmsd_list[0]) * tr_loss.item()
        epoch_rot_loss += len(rmsd_list[0]) * rot_loss.item()
        epoch_tor_loss += len(rmsd_list[0]) * tor_loss.item()


        epoch_loss_contact += len(y_pred) * contact_loss.item()
        epoch_loss_contact_5A += len(y_pred) * contact_loss_cat_off_rmsd_5.item()
        epoch_loss_contact_10A += len(y_pred) * contact_loss_cat_off_rmsd_10.item()
        epoch_loss_affinity_A += len(affinity_pred_A) * affinity_loss_A.item()
        epoch_loss_affinity_B += len(affinity_pred_B_list[0]) * affinity_loss_B.item()
        epoch_loss_rmsd += len(rmsd_list[0]) * rmsd_loss.item()
        epoch_loss_prmsd += len(prmsd_list[0]) * prmsd_loss.item()

        epoch_rmsd_recycling_0_loss +=len(rmsd_list[0]) * rmsd_recycling_0_loss.item()
        epoch_rmsd_recycling_1_loss +=len(rmsd_list[0]) * rmsd_recycling_1_loss.item()
        epoch_rmsd_recycling_9_loss  +=len(rmsd_list[0]) * rmsd_recycling_9_loss.item()
        epoch_rmsd_recycling_19_loss +=len(rmsd_list[0]) * rmsd_recycling_19_loss.item()
        epoch_rmsd_recycling_39_loss +=len(rmsd_list[0]) * rmsd_recycling_39_loss.item()

        # print(f"{loss.item():.3}")
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_A_pred_list.append(affinity_pred_A.detach())
        affinity_B_pred_list.append(affinity_pred_B_list[-1].detach()) #只取最后一个pred做pearson， TODO
        rmsd_pred_list.append(rmsd_list[-1].detach())
        prmsd_pred_list.append(prmsd_list[-1].detach())
        # torch.cuda.empty_cache()

        writer.add_scalar(f'batchLoss.Total/train', loss.item(), global_steps_train)
        writer.add_scalar(f'sampleLoss.Total/train', loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact/train', contact_loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_5A/train', contact_loss_cat_off_rmsd_5.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_10A/train', contact_loss_cat_off_rmsd_10.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Affinity_A/train', affinity_loss_A.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Affinity_B/train', affinity_loss_B.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.RMSD/train', rmsd_loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Pred_RMSD/train', prmsd_loss.item(), global_samples_train)
        global_steps_train+=1
        global_samples_train+=len(data)

    train_result = pd.DataFrame(train_result_list, columns=['compound_name', 'candicate_conf_pos', 'affinity_pred_A', 'affinity_pred_B', 'prmsd_pred'])
    save_path = f"{pre}/results/train_result_{epoch}.csv"
    train_result.to_csv(save_path)
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    # print(y.min(), y.max())
    # print(y_pred.min(), y_pred.max())
    if args.pred_dis:
        y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
        # we define 8A as the cutoff for contact, therefore, contact_threshold will be 1 - 8/10 = 0.2
        contact_threshold = 0.2
    else:
        contact_threshold = 0.5

    affinity = torch.cat(affinity_list)
    affinity_pred_A = torch.cat(affinity_A_pred_list)
    affinity_pred_B = torch.cat(affinity_B_pred_list)
    RMSD_pred = torch.cat(rmsd_pred_list)
    PRMSD_pred = torch.cat(prmsd_pred_list)
    metrics = {
        "loss": epoch_loss_rmsd / len(RMSD_pred) + epoch_loss_affinity_A / len(affinity_pred_A) + epoch_loss_affinity_B / len(affinity_pred_B) + epoch_loss_prmsd / len(PRMSD_pred), 
        "loss_affinity_A": epoch_loss_affinity_A / len(affinity_pred_A),
        "loss_affinity_B": epoch_loss_affinity_B / len(affinity_pred_B),
        "loss_rmsd": epoch_loss_rmsd / len(RMSD_pred),
        "loss_prmsd": epoch_loss_prmsd / len(PRMSD_pred),
        "loss_contact_5A": epoch_loss_contact_5A / (len(y_pred) - epoch_num_nan_contact_5A),
        "loss_contact_10A": epoch_loss_contact_10A / (len(y_pred) - epoch_num_nan_contact_10A),
    }

    # torch.cuda.empty_cache()
    metrics.update(myMetric(y_pred, y, threshold=contact_threshold))
    metrics.update(affinity_metrics(affinity_pred_A, affinity))
    logging.info(f"epoch {epoch:<4d}, train, " + print_metrics(metrics))
    metrics_list.append(metrics)

    # print(metrics_list)
    # release memory

    writer.add_scalar('epochLoss.Total/train', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/train', epoch_loss_contact / len(y_pred), epoch)
    writer.add_scalar('epochLoss.Contact_5A/train', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/train', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity_A/train', epoch_loss_affinity_A / len(affinity_pred_A), epoch)
    writer.add_scalar('epochLoss.Affinity_B/train', epoch_loss_affinity_B / len(affinity_pred_B), epoch)
    writer.add_scalar('epochLoss.RMSD/train', epoch_loss_rmsd / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.Pred_RMSD/train', epoch_loss_prmsd / len(PRMSD_pred), epoch)
    writer.add_scalar('epochNum.TrainedBatches/train', global_steps_train, epoch)
    writer.add_scalar('epochNum.TrainedSamples/train', global_samples_train, epoch)

    writer.add_scalar('epochLoss.rmsd_recycling_0/train', epoch_rmsd_recycling_0_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.rmsd_recycling_1/train', epoch_rmsd_recycling_1_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.rmsd_recycling_9/train', epoch_rmsd_recycling_9_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.rmsd_recycling_19/train', epoch_rmsd_recycling_19_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.rmsd_recycling_39/train', epoch_rmsd_recycling_39_loss / len(RMSD_pred), epoch)

    writer.add_scalar('epochLoss.tr/train', epoch_tr_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.rot/train', epoch_rot_loss / len(RMSD_pred), epoch)
    writer.add_scalar('epochLoss.tor/train', epoch_tor_loss / len(RMSD_pred), epoch)


    continue #TODO :debug
    #===================validation========================================



    y = None
    y_pred = None
    # torch.cuda.empty_cache()
    model.eval()
    use_y_mask = args.use_equivalent_native_y_mask or args.use_y_mask
    # saveFileName = f"{pre}/results/single_valid_epoch_{epoch}.pt"
    saveFileName = f"{pre}/results/valid_epoch_{epoch}.pt"
    info_va_only_compound = info_va.query("use_compound_com and group =='valid' and c_length < 100 and native_num_contact > 5")
    metrics, info_va_save = evaluate_with_affinity(valid_loader, model, contact_criterion, affinity_criterion, args.relative_k, device, pred_dis=pred_dis, info=info_va_only_compound, saveFileName=saveFileName)
    valid_result = pd.DataFrame(info_va_save, columns=['compound_name', 'candicate_conf_pos', 'affinity_pred_A', 'affinity_pred_B', 'rmsd_pred', 'prmsd_pred'])
    save_path = f"{pre}/results/valid_result_{epoch}.csv"
    valid_result.to_csv(save_path)
    # metrics = evaluate_with_affinity(all_pocket_valid_loader, model, contact_criterion, affinity_criterion, args.relative_k, device, pred_dis=pred_dis, info=info_va, saveFileName=saveFileName) #TODO
    #if metrics["auroc"] <= best_auroc and metrics['f1_1'] <= best_f1_1:
    #    # not improving. (both metrics say there is no improving)
    #    epoch_not_improving += 1
    #    ending_message = f" No improvement +{epoch_not_improving}"
    #else:
    #    epoch_not_improving = 0
    #    if metrics["auroc"] > best_auroc:
    #        best_auroc = metrics['auroc']
    #    if metrics['f1_1'] > best_f1_1:
    #        best_f1_1 = metrics['f1_1']
    #    ending_message = " "
    valid_metrics_list.append(metrics)
    #logging.info(f"epoch {epoch:<4d}, single_valid, " + print_metrics(metrics) + ending_message)

    writer.add_scalar('epochLoss.Total/validation', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/validation', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/validation', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/validation', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity_A/validation', metrics["loss_affinity_A"], epoch)
    writer.add_scalar('epochLoss.Affinity_B/validation', metrics["loss_affinity_B"], epoch)
    writer.add_scalar('epochLoss.RMSD/validation', metrics["loss_rmsd"], epoch)
    writer.add_scalar('epochLoss.PRMSD/validation', metrics["loss_prmsd"], epoch)
    writer.add_scalar('epochMetric.RMSE_A/validation', metrics["RMSE_A"], epoch)
    writer.add_scalar('epochMetric.Pearson_A/validation', metrics["Pearson_A"], epoch)
    writer.add_scalar('epochMetric.RMSE_B/validation', metrics["RMSE_B"], epoch)
    writer.add_scalar('epochMetric.Pearson_B/validation', metrics["Pearson_B"], epoch)
    # writer.add_scalar('epochMetric.NativeAUROC/validation', metrics["native_auroc"], epoch)
    # writer.add_scalar('epochMetric.SelectedAUROC/validation', metrics["selected_auroc"], epoch)
    #====================test============================================================


    # saveFileName = f"{pre}/results/test_epoch_{epoch}.pt"
    # metrics = evaluate_with_affinity(test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
    #                                  device, pred_dis=pred_dis, saveFileName=saveFileName, use_y_mask=use_y_mask)
    # test_metrics_list.append(metrics)
    # logging.info(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))


    # saveFileName = f"{pre}/results/single_epoch_{epoch}.pt"
    saveFileName = f"{pre}/results/test_epoch_{epoch}.pt"
    info_only_compound = info.query("use_compound_com and group =='test' and c_length < 100 and native_num_contact > 5")
    metrics,info_save  = evaluate_with_affinity(test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
                                     device, pred_dis=pred_dis, info=info_only_compound, saveFileName=saveFileName)
    test_metrics_list.append(metrics)
    test_result = pd.DataFrame(info_save, columns=['compound_name', 'candicate_conf_pos', 'affinity_pred_A', 'affinity_pred_B', 'rmsd_pred', 'prmsd_pred'])
    save_path = f"{pre}/results/test_result_{epoch}.csv"
    test_result.to_csv(save_path)
    # metrics = evaluate_with_affinity(all_pocket_test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
    #                                  device, pred_dis=pred_dis, info=info, saveFileName=saveFileName) #TODO
    logging.info(f"epoch {epoch:<4d}, test," + print_metrics(metrics))
    writer.add_scalar('epochLoss.Total/test', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/test', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/test', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/test', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity_A/test', metrics["loss_affinity_A"], epoch)
    writer.add_scalar('epochLoss.Affinity_B/test', metrics["loss_affinity_B"], epoch)
    writer.add_scalar('epochLoss.RMSD/test', metrics["loss_rmsd"], epoch)
    writer.add_scalar('epochLoss.PRMSD/test', metrics["loss_prmsd"], epoch)
    writer.add_scalar('epochMetric.RMSE_A/test', metrics["RMSE_A"], epoch)
    writer.add_scalar('epochMetric.Pearson_A/test', metrics["Pearson_A"], epoch)
    writer.add_scalar('epochMetric.RMSE_B/test', metrics["RMSE_B"], epoch)
    writer.add_scalar('epochMetric.Pearson_B/test', metrics["Pearson_B"], epoch)
    # writer.add_scalar('epochMetric.NativeAUROC/test', metrics["native_auroc"], epoch)
    # writer.add_scalar('epochMetric.SelectedAUROC/test', metrics["selected_auroc"], epoch)

    if epoch % 1 == 0:
        torch.save(model.state_dict(), f"{pre}/models/epoch_{epoch}.pt")
    # torch.save((y, y_pred), f"{pre}/results/epoch_{epoch}.pt")
    if epoch_not_improving > 100:
        # early stop.
        print("early stop")
        break
    
    # torch.cuda.empty_cache()
    os.system(f"cp {timestamp}.log {pre}/")

torch.save((metrics_list, valid_metrics_list, test_metrics_list), f"{pre}/metrics.pt")
