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
parser.add_argument("--num_epochs", type=int, default=200,
                    help="define the model will be trained for how many epochs")
args = parser.parse_args()


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

arg_hash = str(abs(hash(str(args))))[0:10]

writer = SummaryWriter(f"./logs/{timestamp}_{args.label}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
handler = logging.FileHandler(f'{timestamp}.log')
handler.setFormatter(logging.Formatter('%(message)s', ""))
logger.addHandler(handler)

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
sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler, pin_memory=False, num_workers=num_workers)
sampler2 = RandomSampler(train_after_warm_up, replacement=True, num_samples=args.sample_n)
train_after_warm_up_loader = DataLoader(train_after_warm_up, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], sampler=sampler2, pin_memory=False, num_workers=num_workers)
valid_batch_size = test_batch_size = 4
valid_loader = DataLoader(valid, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=test_batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
all_pocket_test_loader = DataLoader(all_pocket_test, batch_size=2, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=4)
all_pocket_valid_loader = DataLoader(all_pocket_valid, batch_size=2, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=4)
# import model is put here due to an error related to torch.utils.data.ConcatDataset after importing torchdrug.
from model import *
device = 'cuda'
model = get_model(args.mode, logging, device)

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
for epoch in range(args.num_epochs):
    model.train()
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    epoch_loss_contact = 0.0
    epoch_loss_contact_5A = 0.0
    epoch_num_nan_contact_5A = 0
    epoch_loss_contact_10A = 0.0
    epoch_num_nan_contact_10A = 0
    epoch_loss_affinity = 0.0
    if epoch < data_warmup_epochs:
        data_it = tqdm(train_loader)
    else:
        data_it = tqdm(train_after_warm_up_loader)

    for data in data_it:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred, affinity_pred = model(data)
        # print(data.y.sum(), y_pred.sum())
        y = data.y
        affinity = data.affinity
        dis_map = data.dis_map
        if args.use_equivalent_native_y_mask:
            y_pred = y_pred[data.equivalent_native_y_mask]
            y = y[data.equivalent_native_y_mask]
            dis_map = dis_map[data.equivalent_native_y_mask]
        elif args.use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
            dis_map = dis_map[data.real_y_mask]

        if args.use_affinity_mask:
            affinity_pred = affinity_pred[data.real_affinity_mask]
            affinity = affinity[data.real_affinity_mask]

        if args.pred_dis:
            if args.use_weighted_rmsd_loss:
                contact_loss = contact_criterion(y_pred, dis_map, args.contact_loss_mode) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)
            else:
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
        if args.affinity_loss_mode == 0:
            affinity_loss = relative_k * affinity_criterion(affinity_pred, affinity)
        elif args.affinity_loss_mode == 1:
            native_pocket_mask = data.is_equivalent_native_pocket
            affinity_loss = relative_k * my_affinity_criterion(affinity_pred,
                                                                affinity, 
                                                                native_pocket_mask, decoy_gap=args.decoy_gap)

        # print(contact_loss.item(), affinity_loss.item())
        if args.use_contact_loss == 0:
            loss = contact_loss + affinity_loss
        else:
            loss = contact_loss.float()
            loss = loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        epoch_loss_contact += len(y_pred) * contact_loss.item()
        epoch_loss_contact_5A += len(y_pred) * contact_loss_cat_off_rmsd_5.item()
        epoch_loss_contact_10A += len(y_pred) * contact_loss_cat_off_rmsd_10.item()
        epoch_loss_affinity += len(affinity_pred) * affinity_loss.item()
        # print(f"{loss.item():.3}")
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        # torch.cuda.empty_cache()

        writer.add_scalar(f'batchLoss.Total/train', loss.item(), global_steps_train)
        writer.add_scalar(f'sampleLoss.Total/train', loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact/train', contact_loss.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_5A/train', contact_loss_cat_off_rmsd_5.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Contact_10A/train', contact_loss_cat_off_rmsd_10.item(), global_samples_train)
        writer.add_scalar(f'sampleLoss.Affinity/train', affinity_loss.item(), global_samples_train)
        global_steps_train+=1
        global_samples_train+=len(data)


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
    affinity_pred = torch.cat(affinity_pred_list)
    metrics = {
        "loss": epoch_loss_contact / len(y_pred) + epoch_loss_affinity / len(affinity_pred),
        "loss_contact_5A": epoch_loss_contact_5A / (len(y_pred) - epoch_num_nan_contact_5A),
        "loss_contact_10A": epoch_loss_contact_10A / (len(y_pred) - epoch_num_nan_contact_10A),
    }

    # torch.cuda.empty_cache()
    metrics.update(myMetric(y_pred, y, threshold=contact_threshold))
    metrics.update(affinity_metrics(affinity_pred, affinity))
    logging.info(f"epoch {epoch:<4d}, train, " + print_metrics(metrics))
    metrics_list.append(metrics)
    # print(metrics_list)
    # release memory

    writer.add_scalar('epochLoss.Total/train', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/train', epoch_loss_contact / len(y_pred), epoch)
    writer.add_scalar('epochLoss.Contact_5A/train', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/train', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/train', epoch_loss_affinity / len(affinity_pred), epoch)
    writer.add_scalar('epochNum.TrainedBatches/train', global_steps_train, epoch)
    writer.add_scalar('epochNum.TrainedSamples/train', global_samples_train, epoch)


    #===================validation========================================



    y = None
    y_pred = None
    # torch.cuda.empty_cache()
    model.eval()
    use_y_mask = args.use_equivalent_native_y_mask or args.use_y_mask
    saveFileName = f"{pre}/results/single_valid_epoch_{epoch}.pt"
    metrics = evaluate_with_affinity(all_pocket_valid_loader, model, contact_criterion, affinity_criterion, args.relative_k, device, pred_dis=pred_dis, info=info_va, saveFileName=saveFileName)
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
    logging.info(f"epoch {epoch:<4d}, single_valid, " + print_metrics(metrics) + ending_message)

    writer.add_scalar('epochLoss.Total/validation', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/validation', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/validation', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/validation', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/validation', metrics["loss_affinity"], epoch)
    writer.add_scalar('epochMetric.RMSE/validation', metrics["RMSE"], epoch)
    writer.add_scalar('epochMetric.Pearson/validation', metrics["Pearson"], epoch)
    writer.add_scalar('epochMetric.NativeAUROC/validation', metrics["native_auroc"], epoch)
    writer.add_scalar('epochMetric.SelectedAUROC/validation', metrics["selected_auroc"], epoch)
    #====================test============================================================


    saveFileName = f"{pre}/results/test_epoch_{epoch}.pt"
    metrics = evaluate_with_affinity(test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
                                     device, pred_dis=pred_dis, saveFileName=saveFileName, use_y_mask=use_y_mask)
    test_metrics_list.append(metrics)
    logging.info(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))


    saveFileName = f"{pre}/results/single_epoch_{epoch}.pt"
    metrics = evaluate_with_affinity(all_pocket_test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
                                     device, pred_dis=pred_dis, info=info, saveFileName=saveFileName)
    logging.info(f"epoch {epoch:<4d}, single," + print_metrics(metrics))
    writer.add_scalar('epochLoss.Total/test', metrics["loss"], epoch)
    writer.add_scalar('epochLoss.Contact/test', metrics["loss_contact"], epoch)
    writer.add_scalar('epochLoss.Contact_5A/test', metrics["loss_contact_5A"], epoch)
    writer.add_scalar('epochLoss.Contact_10A/test', metrics["loss_contact_10A"], epoch)
    writer.add_scalar('epochLoss.Affinity/test', metrics["loss_affinity"], epoch)
    writer.add_scalar('epochMetric.RMSE/test', metrics["RMSE"], epoch)
    writer.add_scalar('epochMetric.Pearson/test', metrics["Pearson"], epoch)
    writer.add_scalar('epochMetric.NativeAUROC/test', metrics["native_auroc"], epoch)
    writer.add_scalar('epochMetric.SelectedAUROC/test', metrics["selected_auroc"], epoch)

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
