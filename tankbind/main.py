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
# %matplotlib inline
from data_nci import get_data_reproduced  # Modified!
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

from utils_nci import NCICriterion, eval_nci_classification, evaluate_with_affinity_and_nci



def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

parser.add_argument("--relative_dist_criterion", type=float, default=1,
                    help="adjust the strength of the dist losses head relative to the affinity loss.")

parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                    help="define how the relative_k changes over epochs")
parser.add_argument("--warm_up_epochs", type=int, default=15,
                    help="used in combination with relative_k_mode.")
parser.add_argument("--data_warm_up_epochs", type=int, default=0,
                    help="option to switch training data after certain epochs.")
# parser.add_argument("--resultFolder", type=str, default="../",
#                     help="information you want to keep a record.")
parser.add_argument("--resultFolder", type=str, default="../results/",
                    help="information you want to keep a record.")
parser.add_argument("--label", type=str, default="",
                    help="information you want to keep a record.")
parser.add_argument("--use_contact_loss", type=int, default=0,
                    help="whether to upgrade contact loss during training, 0 means use, other means not use")
parser.add_argument("--use_weighted_rmsd_loss", type=bool, default=False,
                    help="whether to change contact weight according to distance")
parser.add_argument("--unused_code", type=bool, default=False,
                    help="ok now PyCharm doesn't show `this codeblock is not accessible.")
parser.add_argument("--relative_nci_warm_up", type=bool, default=False,
                    help="nci loss factor will increase by epoch")
parser.add_argument("--nci_weight", type=int, default=1,
                    help="Weight of NCI loss")
parser.add_argument("--under_sampling_ratio", type=int, default=8, help="Sampling Ratio of Negative Samples")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

arg_hash =  str(abs(hash(str(args))))[0:10]

writer = SummaryWriter(f"./logs/{timestamp}_{arg_hash}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
handler = logging.FileHandler(f'./{timestamp}_{arg_hash}.log')
handler.setFormatter(logging.Formatter('%(message)s', ""))
logger.addHandler(handler)

logging.info(
    f'''
    {' '.join(sys.argv)}
    {timestamp}
    {arg_hash}
    {args.label}
    --------------------------------
    ''')

logging.info(
    f'''
    New Parameters:
        General:
            relative_dist_criterion: {args.relative_dist_criterion}
        NCI-related:
            nci_weight: {args.nci_weight}
            relative_nci_warm_up: {args.relative_nci_warm_up}
            under_sampling_ratio: {args.under_sampling_ratio}
    '''
)

pre = f"{args.resultFolder}/{timestamp}_{arg_hash}"
os.system(f"mkdir -p {pre}/models")
os.system(f"mkdir -p {pre}/results")
os.system(f"mkdir -p {pre}/src")
os.system(f"cp *.py {pre}/src/")
os.system(f"cp -r gvp {pre}/src/")

os.system('cd /tmp && ls ')
os.system('')

torch.set_num_threads(1)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

train, train_after_warm_up, valid, test, all_pocket_test, all_pocket_valid, info_test, info_va = get_data_reproduced(
    args.data, logging,
    addNoise=args.addNoise,
    pre="/home/jovyan/dataspace/NFT/main/tankbind_data")


logging.info(f"data point train: {len(train)}, train_after_warm_up: {len(train_after_warm_up)}, "
             f"valid: {len(valid)}, test: {len(test)}")
num_workers = 10

## Sampler and DataLoaders

if args.data_warm_up_epochs > 0:  ## if args.data_war_up_epochs == 0, the first sampler/train_loader are not used.
    sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
    train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler,
                              pin_memory=False, num_workers=num_workers)
else:
    sampler, train_loader = None, None

sampler2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=args.sample_n)
train_after_warm_up_loader = DataLoader(train_after_warm_up, batch_size=args.batch_size,
                                        follow_batch=['x', 'compound_pair'], sampler=sampler2, pin_memory=False,
                                        num_workers=num_workers)
valid_batch_size = test_batch_size = 4

if args.unused_code:  ##
    valid_loader = DataLoader(dataset=valid, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'],
                              shuffle=False, pin_memory=False, num_workers=num_workers)

test_loader = DataLoader(dataset=test, batch_size=test_batch_size, follow_batch=['x', 'compound_pair'],
                         shuffle=False, pin_memory=False, num_workers=num_workers)
all_pocket_test_loader = DataLoader(dataset=all_pocket_test, batch_size=2, follow_batch=['x', 'compound_pair'],
                                    shuffle=False, pin_memory=False, num_workers=4)
all_pocket_valid_loader = DataLoader(dataset=all_pocket_valid, batch_size=2, follow_batch=['x', 'compound_pair'],
                                     shuffle=False, pin_memory=False, num_workers=4)


## Import models
# import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
from model import *

device = 'cuda'
model = get_nft_model(args.mode, logging, device)

if args.restart:
    model.load_state_dict(torch.load(args.restart))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


## Criterion Settings

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
    raise ValueError("In current setting, argument `args.pred_dis` should be True")

affinity_criterion = nn.MSELoss()
nci_criterion = NCICriterion(class_weight=torch.tensor([1, 1]), under_sampling_ratio=args.under_sampling_ratio).to(model.device)


## Metrics Pre-definition

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
nci_weight = args.nci_weight

## ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ====
## ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ====
## ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ====

relative_dist_criterion = args.relative_dist_criterion  # Factor of distance loss

for epoch in range(200):
    
    print(f"!!====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ======!!")
    print(f"!!====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ====== EPOCH {epoch:<4d} ======!!")
    model.train()
    print("==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==")
    print("=== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ===")
    print("== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ==== 训练 ==== TRAIN ====")
    
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    nci_list = []
    nci_pred_list = []

    epochLoss_contact = 0.0
    epochLoss_contact_5A = 0.0
    epochNum_nan_contact_5A = 0
    epochLoss_contact_10A = 0.0
    epochNum_nan_contact_10A = 0
    epochLoss_affinity = 0.0
    epochLoss_sampled_nci = 0.0

    if epoch < data_warmup_epochs:
        if train_loader is None:
            raise ValueError(f"train_loader can't be None with args.data_warmup_epochs={args.data_warmup_epochs}.")
        data_it = tqdm(train_loader)
    else:
        data_it = tqdm(train_after_warm_up_loader)

    for data in data_it:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred, affinity_pred, nci_pred = model(data)
        ## y_pred: sequence, affinity_pred: sequence, nci_pred: None | sequence
        y = data.y
        affinity = data.affinity
        dis_map = data.dis_map
        nci_sequence = data.nci_sequence

        ## Application of mask to outputs
        if args.use_equivalent_native_y_mask:
            y_pred = y_pred[data.equivalent_native_y_mask]
            y = y[data.equivalent_native_y_mask]
            dis_map = dis_map[data.equivalent_native_y_mask]
            if nci_pred is not None:
                nci_sequence = nci_sequence[data.equivalent_native_nci_mask]
                nci_pred = nci_pred[data.equivalent_native_nci_mask]
        
        elif args.use_y_mask:  ## Not used
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
            dis_map = dis_map[data.real_y_mask]
            raise ValueError("In current setting, args.use_y_mask is not used for nci. AXVSAFW")
        else:
            raise ValueError("In current setting, args.use_equivalent_native_y_mask should be True. NXVCWX.")

        if args.use_affinity_mask:
            affinity_pred = affinity_pred[data.real_affinity_mask]
            affinity = affinity[data.real_affinity_mask]

        ## Computation of contact_loss
        if args.pred_dis:
            contact_loss = relative_dist_criterion * contact_criterion(y_pred, dis_map) \
                if len(dis_map) > 0 \
                else torch.tensor([0]).to(dis_map.device)
            with torch.no_grad():
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=5)):
                    epochNum_nan_contact_5A += len(y_pred)
                    contact_loss_cat_off_rmsd_5 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_5 = cut_off_rmsd(y_pred, dis_map, cut_off=5)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=10)):
                    epochNum_nan_contact_10A += len(y_pred)
                    contact_loss_cat_off_rmsd_10 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_10 = cut_off_rmsd(y_pred, dis_map, cut_off=10)
        else:
            contact_loss = relative_dist_criterion * contact_criterion(y_pred, y) \
                if len(y) > 0 \
                else torch.tensor([0]).to(y.device)
            y_pred = y_pred.sigmoid()


        ## Computation of weighted affinity_loss
        if args.restart is None:
            base_relative_k = args.relative_k
            if args.relative_k_mode == 0:
                # increase exponentially. reach base_relative_k at epoch = warm_up_epochs.
                relative_k = min(base_relative_k * (2 ** epoch) / (2 ** warm_up_epochs), base_relative_k)
            elif args.relative_k_mode == 1:
                # increase linearly
                relative_k = min(base_relative_k / warm_up_epochs * epoch, base_relative_k)
            else:
                raise ValueError(f"Invalid argument relative_k_mode: {args.relative_k_mode}.")
        else:
            relative_k = args.relative_k
        if args.affinity_loss_mode == 0:
            affinity_loss = relative_k * affinity_criterion(affinity_pred, affinity)
        elif args.affinity_loss_mode == 1:
            native_pocket_mask = data.is_equivalent_native_pocket
            ## my_affinity_criterion: marginal_affinity_criterion
            affinity_loss = relative_k * my_affinity_criterion(affinity_pred,
                                                               affinity,
                                                               native_pocket_mask, decoy_gap=args.decoy_gap)
        else:
            raise ValueError(f"Invalid argument affinity_loss_mode: {args.relative_k_mode}.")
        
        ## Computation of nci_loss when nci_pred is not None
        relative_nci_warm_up = args.relative_nci_warm_up
        if nci_pred is not None:
            nci_loss, batchNum_sampled_nci = nci_criterion(nci_pred, nci_sequence)
            if relative_nci_warm_up:
                nci_loss = relative_dist_criterion * nci_loss * nci_weight * relative_k * 100
            else:
                nci_loss = relative_dist_criterion * nci_loss * nci_weight
        else:
            nci_loss = torch.tensor([0]).to(y.device)
            batchNum_sampled_nci = 0
            
        ## Computation of total loss and backward
        if args.use_contact_loss == 0:
            loss = contact_loss + affinity_loss + nci_loss
        else:
            loss = affinity_loss
        loss.backward()
        optimizer.step()

        ## Record of batch-level loss -> to compute epoch-level loss
        epochLoss_contact += len(y_pred) * contact_loss.item()
        epochLoss_contact_5A += len(y_pred) * contact_loss_cat_off_rmsd_5.item()
        epochLoss_contact_10A += len(y_pred) * contact_loss_cat_off_rmsd_10.item()
        epochLoss_affinity += len(affinity_pred) * affinity_loss.item()
        if nci_pred is not None:
            if batchNum_sampled_nci != 0:
                epochLoss_sampled_nci += batchNum_sampled_nci * nci_loss.item()

        ## Record of batch-level output tensors -> to compute epoch-level tensors
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        if nci_pred is not None:
            nci_list.append(nci_sequence)
            nci_pred_list.append(nci_pred)

        writer.add_scalar(f'batchLoss.Total/train', loss.item(), global_steps_train)
        writer.add_scalar(f'sampleLoss.Total/train', loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact/train', contact_loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_5A/train', contact_loss_cat_off_rmsd_5.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_10A/train', contact_loss_cat_off_rmsd_10.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Affinity/train', affinity_loss.item(), global_samples_train)

        if nci_pred is not None:
            nci_accuracy, nci_recall, nci_precision, tp, fp, tn, fn = eval_nci_classification(nci_pred.detach(), nci_sequence)
            writer.add_scalar(f'sampleLoss.NCI/train', nci_loss.item(), global_samples_train)
            writer.add_scalar(f'sampleMetric.NCI.Accuracy/train', nci_accuracy, global_samples_train)
            writer.add_scalar(f'sampleMetric.NCI.Precision/train', nci_precision, global_samples_train)
            writer.add_scalar(f'sampleMetric.NCI.Recall/train', nci_recall, global_samples_train)

        global_steps_train += 1
        global_samples_train += len(data)

    ## ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ====
    ## ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ====

    ## Concatenation of batch-level output tensors to epoch-level tensor
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    if args.pred_dis:
        y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
        # we define 8A as the cutoff for contact, therefore, contact_threshold will be 1 - 8/10 = 0.2
        contact_threshold = 0.2
    else:
        contact_threshold = 0.5
    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    nci_true = torch.cat(nci_list)
    nci_pred = torch.cat(nci_pred_list)

    ## Epoch-level evaluation of NCI classification
    nci_accuracy, nci_recall, nci_precision, tp, tn, fp, fn = eval_nci_classification(nci_pred.detach(), nci_true)
    logging.info(f"epoch {epoch}: ")
    metrics = {
        "loss_total": epochLoss_contact / len(y_pred) + epochLoss_affinity / len(affinity_pred) + (
            epochLoss_sampled_nci / len(nci_pred)),
        "loss_affinity": epochLoss_affinity / len(affinity_pred),
        "loss_contact": epochLoss_contact / len(y_pred),
        "loss_contact_5A": epochLoss_contact_5A / (len(y_pred) - epochNum_nan_contact_5A),
        "loss_contact_10A": epochLoss_contact_10A / (len(y_pred) - epochNum_nan_contact_10A),
        "loss_nci": epochLoss_sampled_nci / len(nci_pred),
        "nci_accuracy": nci_accuracy,
        "nci_recall": nci_recall,
        "nci_precision": nci_precision,
        "nci_TP": tp,
        "nci_TN": tn,
        "nci_FP": fp,
        "nci_FN": fn
    }

    ## Rename ambiguous function names
    metrics.update(myMetric(y_pred, y, threshold=contact_threshold))
    metrics.update(affinity_metrics(affinity_pred, affinity))
    logging.info(f"epoch {epoch:<4d}| train_______, " + print_metrics(metrics))
    metrics_list.append(metrics)

    writer.add_scalar('epochLoss.Total/train', metrics["loss_total"], epoch)
    writer.add_scalar('epochLoss.Contact/train', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/train', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/train', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/train', metrics["loss_affinity"], epoch)
    writer.add_scalar('epochLoss.NCI/train', metrics["loss_nci"], epoch)
    writer.add_scalar('epochMetric.NCI.Accuracy/train', metrics["nci_accuracy"], epoch)
    writer.add_scalar('epochMetric.NCI.Recall/train', metrics["nci_recall"], epoch)
    writer.add_scalar('epochMetric.NCI.Precision/train', metrics["nci_precision"], epoch)
    writer.add_scalar('epochNum.TrainedBatches/train', global_steps_train, epoch)
    writer.add_scalar('epochNum.TrainedSamples/train', global_samples_train, epoch)
    
    
## ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION
## ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION
    print("== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION")
    print("= VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ")
    print(" VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION ==== 验证 ==== VALIDATION =")
    y = None
    y_pred = None
    model.eval()
    use_y_mask = args.use_equivalent_native_y_mask or args.use_y_mask
    
    
    saveFileName = f"{pre}/results/single_valid_epoch_{epoch}.pt"
    
    metrics = evaluate_with_affinity_and_nci(
        data_loader=all_pocket_valid_loader,
        model=model,
        contact_criterion=contact_criterion,
        affinity_criterion=affinity_criterion,
        nci_criterion=nci_criterion,
        relative_k=args.relative_k, device=device, pred_dis=pred_dis, 
        info=info_va, saveFileName=saveFileName)

    if metrics["auroc"] <= best_auroc and metrics['f1_1'] <= best_f1_1:
        # not improving. (both metrics say there is no improving)
        epoch_not_improving += 1
        ending_message = f" No improvement +{epoch_not_improving}"
    else:
        epoch_not_improving = 0
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics['auroc']
        if metrics['f1_1'] > best_f1_1:
            best_f1_1 = metrics['f1_1']
        ending_message = " "
    valid_metrics_list.append(metrics)
    logging.info(f"epoch {epoch:<4d}| single_valid, " + print_metrics(metrics) + ending_message)

    writer.add_scalar('epochLoss.Total/validation', metrics["loss_total"], epoch)
    writer.add_scalar('epochLoss.Contact/validation', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/validation', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/validation', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/validation', metrics["loss_affinity"], epoch)
    writer.add_scalar('epochLoss.NCI/validation', metrics["loss_nci"], epoch)
    writer.add_scalar('epochMetric.NCI.Accuracy/validation', metrics["nci_accuracy"], epoch)
    writer.add_scalar('epochMetric.NCI.Recall/validation', metrics["nci_recall"], epoch)
    writer.add_scalar('epochMetric.NCI.Precision/validation', metrics["nci_precision"], epoch)
    writer.add_scalar('epochMetric.NCI.AUROC/validation', metrics["metric_nci_auroc"], epoch)
    writer.add_scalar('epochMetric.RMSE/validation', metrics["metric_rmse"], epoch)
    writer.add_scalar('epochMetric.Pearson/validation', metrics["metric_pearson"], epoch)
    writer.add_scalar('epochMetric.NativeAUROC/validation', metrics["metric_native_auroc"], epoch)
    writer.add_scalar('epochMetric.SelectedAUROC/validation', metrics["metric_selected_metric"], epoch)
    
    
## ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 =
## ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 =
    print("==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试")
    print("=== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ")
    print("== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 ==== TEST ==== 测试 =")
    
    saveFileName = f"{pre}/results/epoch_{epoch}.pt"
    metrics = evaluate_with_affinity_and_nci(
        data_loader=test_loader,
        model=model,
        contact_criterion=contact_criterion,
        affinity_criterion=affinity_criterion,
        nci_criterion=nci_criterion,
        relative_k=args.relative_k, device=device, pred_dis=pred_dis, saveFileName=saveFileName, use_y_mask=use_y_mask)

    test_metrics_list.append(metrics)
    logging.info(f"epoch {epoch:<4d}| test________,  " + print_metrics(metrics))

    saveFileName = f"{pre}/results/single_epoch_{epoch}.pt"

    metrics = evaluate_with_affinity_and_nci(
        data_loader=all_pocket_test_loader,
        model=model,
        contact_criterion=contact_criterion,
        affinity_criterion=affinity_criterion,
        nci_criterion=nci_criterion,
        relative_k=args.relative_k, device=device, pred_dis=pred_dis, 
        info=info_test, saveFileName=saveFileName)

    logging.info(f"epoch {epoch:<4d}| single______," + print_metrics(metrics))

    writer.add_scalar('epochLoss.Total/test', metrics["loss_total"], epoch)
    writer.add_scalar('epochLoss.Contact/test', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/test', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/test', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/test', metrics["loss_affinity"], epoch)
    writer.add_scalar('epochLoss.NCI/test', metrics["loss_nci"], epoch)
    writer.add_scalar('epochMetric.NCI.Accuracy/test', metrics["nci_accuracy"], epoch)
    writer.add_scalar('epochMetric.NCI.Recall/test', metrics["nci_recall"], epoch)
    writer.add_scalar('epochMetric.NCI.Precision/test', metrics["nci_precision"], epoch)
    writer.add_scalar('epochMetric.NCI.AUROC/test', metrics["metric_nci_auroc"], epoch)
    writer.add_scalar('epochMetric.RMSE/test', metrics["metric_rmse"], epoch)
    writer.add_scalar('epochMetric.Pearson/test', metrics["metric_pearson"], epoch)
    writer.add_scalar('epochMetric.NativeAUROC/test', metrics["metric_native_auroc"], epoch)
    writer.add_scalar('epochMetric.SelectedAUROC/test', metrics["metric_selected_metric"], epoch)
    

    if epoch % 1 == 0:
        torch.save(model.state_dict(), f"{pre}/models/epoch_{epoch}.pt")
    # torch.save((y, y_pred), f"{pre}/results/epoch_{epoch}.pt")
    if epoch_not_improving > 100:
        # early stop.
        print("early stop")
        break

    # torch.cuda.empty_cache()
    os.system(f"cp {timestamp}_{arg_hash}.log {pre}/")

torch.save((metrics_list, valid_metrics_list, test_metrics_list), f"{pre}/metrics.pt")
