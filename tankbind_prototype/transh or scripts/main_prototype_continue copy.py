# training script for new model
# prototype 2023-1-17

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import glob
import torch
from torch.nn import MSELoss
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
from data_prototype import get_full_data_prototype
from sampler_prototype import SessionBatchSampler
from model import *


class PairwiseLoss(nn.Module):
    def __init__(self, keep_rate=1., sigmoid_lambda=0.5,
                 ingrp_thr=2, outgrp_thr=9999, eval=False):
        super(PairwiseLoss, self).__init__()
        self.eval = eval
        self.register_buffer('keep_rate', torch.tensor(keep_rate, dtype=torch.float64))
        self.register_buffer('sigmoid_lambda', torch.tensor(sigmoid_lambda, dtype=torch.float64))
        self.register_buffer('ingrp_thr', torch.tensor(ingrp_thr, dtype=torch.float64))
        self.register_buffer('outgrp_thr', torch.tensor(outgrp_thr, dtype=torch.float64))

    def forward(self, pred, true,): # groupid):

        """
        Customized pairwise ranking loss.

        """
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)
        if len(true.shape) == 1:
            true = true.unsqueeze(1)
        drop_rate = 1 - self.keep_rate
        true_tile_row = true.repeat((1, true.shape[0]))
        true_tile_col = torch.t(true_tile_row)
        assert (true_tile_row.shape == true_tile_col.shape)

        pred_tile_row = pred.repeat((1, pred.shape[0]))
        pred_tile_col = torch.t(pred_tile_row)
    
        diff = (true_tile_row - true_tile_col) / (torch.abs(true_tile_col) + 1e-4)

        pred_pair = torch.stack([pred_tile_row, pred_tile_col], dim=0)
        valid_ind = torch.logical_or(diff > self.ingrp_thr,
                                     diff > self.outgrp_thr)
        
        # print("len(valid_ind)",len(valid_ind))
        
        pred_pair_valid = pred_pair.masked_select(valid_ind).reshape(2, -1)
        pred_pair_diff = pred_pair_valid[0] - pred_pair_valid[1]
        reverse = torch.sum(pred_pair_diff > 0)
        ntotal = pred_pair_valid.shape[1] + 1e-8
        #print("Total valid pairs: {:.3f}, reversed: {:.3f}, reverse ratio: {:.3f}".format(ntotal, reverse, reverse/ntotal))
        if drop_rate > 1e-4:
            pred_pair_valid_dropout = torch.nn.functional.dropout(pred_pair_valid[0], drop_rate) * self.keep_rate
            pred_pair_valid_ind = torch.logical_or(pred_pair_valid_dropout == pred_pair_valid[0],
                                                   pred_pair_valid_dropout != 0.0)

            pred_pair_valid = pred_pair_valid.masked_select(pred_pair_valid_ind).reshape(2, -1)
        # print(pred_pair_valid[1] - pred_pair_valid[0])
        loss = torch.sum(torch.log(1. + torch.exp(self.sigmoid_lambda * (pred_pair_valid[1] - pred_pair_valid[0] + 1))))
        num = pred_pair_valid.shape[1] + 1e-8
        loss = torch.div(loss, num)
        return loss, (reverse/ntotal)


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

def data_split(split_mode):
    return None, None, None

def main(args):
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    w_label = "_" + args.label if args.label != "" else ""
    writer = SummaryWriter(f"./logs/{timestamp}{w_label}")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("")
    handler = logging.FileHandler(f'{timestamp}.log')
    handler.setFormatter(logging.Formatter('%(message)s', ""))
    logger.addHandler(handler)

    logging.info(
        f"{' '.join(sys.argv)}\n" \
        f"{args.label} at {timestamp}\n" \
        f"=================================================================\n" \
        f"{args}\n" \
        f"=================================================================\n\n"
    )
    pre = f"{args.resultFolder}/{timestamp}"
    os.system(f"mkdir -p {pre}/models")
    os.system(f"mkdir -p {pre}/results")
    os.system(f"mkdir -p {pre}/src")
    os.system(f"cp *.py {pre}/src/")
    os.system(f"cp -r gvp {pre}/src/")

    torch.set_num_threads(1)
    # # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
    torch.multiprocessing.set_sharing_strategy('file_system')

    input_path = "/home/jovyan/main_tankbind/dataset_prototype"
    print(f"Getting full dataset from {input_path}...", end="")
    train_dataset, iid_dataset, ood_dataset, test_dataset = get_full_data_prototype(input_path, "0")
    print("fin!")
    logging.info(
        f"train: {len(train_dataset.data)}\n"\
        f"iid_val: {(len(iid_dataset.data) if iid_dataset is not None else 0)}\n"\
        f"ood_val: {(len(ood_dataset.data) if ood_dataset is not None else 0)}\n"\
        f"test: {(len(test_dataset.data) if test_dataset is not None else 0)}\n"
    )
    
    num_workers = 10

    os.system(f"mkdir -p {pre}/train/batch_split_info")
    os.system(f"mkdir -p {pre}/train/epoch_result")
    print("Creating train_sampler and train_dataloader...", end="")
    train_sampler = SessionBatchSampler(train_dataset, n=args.sampler_batch_size, seed=0, name=timestamp, index_save_path=f"{pre}/train/batch_split_info")
    train_dataloader = DataLoader(train_dataset,
                        follow_batch=['x', 'compound_pair'], 
                        batch_sampler=train_sampler, 
                        pin_memory=False, 
                        num_workers=8)
    print("fin!")
    
    valid_batch_size = test_batch_size = 8

    print("Creating iid_dataloader, ood_dataloader, test_dataloader...", end="")
    iid_dataloader = DataLoader(iid_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
    ood_dataloader = DataLoader(ood_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
    print("fin!")
    



    
    device = 'cuda:0'
    
    
    print("Loading model, creating optimizer and loss function...", end="")
    model = get_model(0, logging, device, readout_mode=args.readout_mode, output_func=args.output_func)
    if args.restart:
        model.load_state_dict(torch.load(args.restart))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    pairwiseloss = PairwiseLoss(sigmoid_lambda=args.sigmoid_lambda, ingrp_thr=args.pair_threshold)
    mseloss = MSELoss()
    print("fin!")



    num_steps_train = 0
    num_steps_validation = 0
    num_steps_test = 0
    num_samples_train = 0
    num_samples_val = 0
    num_samples_test = 0

    for epoch in range(args.num_epochs):
        if (args.restart != False) and (epoch <= args.restart_epoch):
            logging.info(f"Skip epoch {epoch} | continue from epoch {args.restart_epoch}")
            continue
        # TRAINING
        print(f"=== training of epoch {epoch} ===")
        if True: # Alignment
            model.train()
            affinity_true_list = []
            affinity_pred_list = []
            loss_list = []
            recto_rate_list = []
            # 更新 batch 分割方式。
            train_dataloader.batch_sampler.prepare_batches_for_epoch(epoch=epoch) 
            for data in tqdm(train_dataloader):
                num_steps_train += 1
                num_samples_train += len(data)
                data = data.to(device)
                optimizer.zero_grad()
                _, affinity_pred = model(data) 
                del _ # Note: for now, y is not in need.
                affinity_true = data.value

                
                if not args.use_mse_loss:
                    loss, recto_rate = pairwiseloss(affinity_pred, affinity_true)
                else:
                    loss = mseloss(affinity_pred.float(), torch.log10(affinity_true).float())

                if not (abs(loss.item()) > 100000):
                    loss.backward()
                    optimizer.step()
                else:
                    logging.info(f"--loss error | epoch {epoch}, session {list(data.session_au)[0]}, sample_id {str(data.sample_id)}, pred {[str(_) for _ in affinity_pred]}, true {[str(_) for _ in affinity_true]}")
                
                
                affinity_true_list.append(affinity_true.detach().cpu())
                affinity_pred_list.append(affinity_pred.detach().cpu())
                loss_list.append(loss.detach().cpu())
                if not args.use_mse_loss:
                    recto_rate_list.append(recto_rate.detach().cpu().item())
                
                
                writer.add_scalar(f'rank_loss.by_step/train', loss.item(), num_steps_train)
                if not args.use_mse_loss:
                    writer.add_scalar(f'recto_rate.by_step/train', recto_rate.item(), num_steps_train)
                writer.add_scalar(f'rank_loss.by_sample/train', loss.item(), num_samples_train)
                if not args.use_mse_loss:
                    writer.add_scalar(f'recto_rate.by_sample/train', recto_rate.item(), num_samples_train)
            # end of sample scope

            affinity_true = torch.cat(affinity_true_list)
            affinity_pred = torch.cat(affinity_pred_list)
            loss = torch.stack(loss_list)
            
            torch.save({"affinity_true": affinity_true, "affinity_pred": affinity_pred, "loss": loss}, f"{pre}/train/epoch_result/epoch_{epoch}.pt")
            

        # IID VALIDATION
        print(f"=== iid validation of epoch {epoch} ===")
        model.eval()
        with torch.no_grad():
            validation_info = iid_dataset.data
            validation_sample_id = []
            affinity_pred_list = []
            validation_result = []
            validation_len = 0
            validation_loss = 0
            validation_recto_rate = 0
            # sample scope
            print("IID VALIDATION: Compute affinity for validation set.")
            for data in tqdm(iid_dataloader):
                validation_sample_id.extend(data.sample_id)
                data = data.to(device)
                _, affinity_pred = model(data)
                del _
                affinity_pred_list.append(affinity_pred.detach().cpu())
            # end of sample scope
            
            # write to DataFrame
            validation_info["dataset_sample_id"] = validation_sample_id
            validation_info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()
            
            print("IID VALIDATION: Compute loss and ratio for validation set.")
            for session in validation_info.session_au.unique():
                df = validation_info[validation_info.session_au==session]
                affinity_true = torch.tensor(df.value.values).to(device)
                affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
                loss, recto_rate = pairwiseloss(pred=affinity_pred, true=affinity_true)
                loss = loss.detach().cpu()
                recto_rate = recto_rate.detach().cpu()
                session_len= len(df)
                
                validation_len += session_len
                validation_loss += loss * session_len
                validation_recto_rate += recto_rate * session_len
                
                validation_result.append([session, session_len, loss.item(), recto_rate.item()])
                # end of sample scope: ratio and loss calculate
            
            
        # save result as .csv
        validation_result = pd.DataFrame(validation_result, columns=["session", "length", "loss", "recto_rate"])
        savepath = f"{pre}/results/iid_val_result_{epoch}.csv"
        validation_result.to_csv(savepath)
        validation_info.to_csv(f"{pre}/results/iid_val_info_{epoch}.csv")
        print(f"IID VALIDATION: save iid validation result of epoch {epoch} to {savepath}.")
            
            
        # save result to tensorboard
        validation_loss /= validation_len
        validation_recto_rate /= validation_len
        writer.add_scalar(f'rank_loss.by_epoch/iid_validation', validation_loss.item(), epoch)
        writer.add_scalar(f'recto_rate.by_epoch/iid_validation', validation_recto_rate.item(), epoch)
        logging.info(f"epoch {epoch} - iid_validation | rank_loss {validation_loss.item()}, averaged_recto_rate {validation_recto_rate.item()}")
        
        # OOD VALIDATION
        print(f"=== ood validation of epoch {epoch} ===")
        model.eval()
        with torch.no_grad():
            validation_info = ood_dataset.data
            validation_sample_id = []
            affinity_pred_list = []
            validation_result = []
            validation_len = 0
            validation_loss = 0
            validation_recto_rate = 0
            # sample scope
            print("OOD VALIDATION: Compute affinity for validation set.")
            for data in tqdm(ood_dataloader):
                validation_sample_id.extend(data.sample_id)
                data = data.to(device)
                _, affinity_pred = model(data)
                del _
                affinity_pred_list.append(affinity_pred.detach().cpu())
            # end of sample scope
            
            # write to DataFrame
            validation_info["dataset_sample_id"] = validation_sample_id
            validation_info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()
            
            print("OOD VALIDATION: Compute loss and ratio for validation set.")
            for session in validation_info.session_au.unique():
                df = validation_info[validation_info.session_au==session]
                affinity_true = torch.tensor(df.value.values).to(device)
                affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
                loss, recto_rate = pairwiseloss(pred=affinity_pred, true=affinity_true)
                loss = loss.detach().cpu()
                recto_rate = recto_rate.detach().cpu()
                session_len= len(df)
                
                validation_len += session_len
                validation_loss += loss * session_len
                validation_recto_rate += recto_rate * session_len
                
                validation_result.append([session, session_len, loss.item(), recto_rate.item()])
            
        # save result as .csv
        validation_result = pd.DataFrame(validation_result, columns=["session", "length", "loss", "recto_rate"])
        savepath = f"{pre}/results/ood_val_result_{epoch}.csv"
        validation_result.to_csv(savepath)
        validation_info.to_csv(f"{pre}/results/ood_val_info_{epoch}.csv")
        print(f"OOD VALIDATION: save ood validation result of epoch {epoch} to {savepath}.")
            
            
        # save result to tensorboard
        validation_loss /= validation_len
        validation_recto_rate /= validation_len
        writer.add_scalar(f'rank_loss.by_epoch/ood_validation', validation_loss.item(), epoch)
        writer.add_scalar(f'recto_rate.by_epoch/ood_validation', validation_recto_rate.item(), epoch)
        logging.info(f"epoch {epoch} - ood_validation | rank_loss {validation_loss.item()}, averaged_recto_rate {validation_recto_rate.item()}")
            
        # TEST
        print(f"=== test of epoch {epoch} ===")
        model.eval()
        with torch.no_grad():
            test_info = test_dataset.data
            affinity_pred_list = []
            test_result = []
            test_len = 0
            test_loss = 0
            test_recto_rate = 0
            test_sample_id = []
            
            # sample scope
            print("TEST: Compute affinity for test set.")
            for data in tqdm(test_dataloader):
                test_sample_id.extend(data.sample_id)
                data = data.to(device)
                _, affinity_pred = model(data)
                del _
                affinity_pred_list.append(affinity_pred.detach().cpu())
            # end of sample scope
            
            # write to DataFrame
            test_info["dataset_sample_id"] = test_sample_id
            test_info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()
            # sample scope: ratio and loss calculate
            print("TEST: Compute loss and ratio for test set.")
            for session in tqdm(test_info.session_au.unique(), total=len(test_info.session_au.unique())):
                df = test_info[test_info.session_au==session]
                affinity_true = torch.tensor(df.value.values).to(device)
                affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
                loss, recto_rate = pairwiseloss(pred=affinity_pred, true=affinity_true)
                loss = loss.detach().cpu()
                recto_rate = recto_rate.detach().cpu()
                session_len= len(df)
                
                test_len += session_len
                test_loss += loss * session_len
                test_recto_rate += recto_rate * session_len
                
                test_result.append([session, session_len, loss.item(), recto_rate.item()])
                # end of sample scope: ratio and loss calculate
                
            
        # save result as .csv
        test_result = pd.DataFrame(test_result, columns=["session", "length", "loss", "recto_rate"])
        savepath = f"{pre}/results/test_result_{epoch}.csv"
        test_result.to_csv(savepath)
        test_info.to_csv(f"{pre}/results/test_info_{epoch}.csv")
        print(f"TEST: save test result of epoch {epoch} to {savepath}.")

        # save result to tensorboard
        test_loss /= test_len
        test_recto_rate /= test_len
        writer.add_scalar(f'rank_loss.by_epoch/test', test_loss.item(), epoch)
        writer.add_scalar(f'recto_rate.by_epoch/test', test_recto_rate.item(), epoch)
        logging.info(f"epoch {epoch} - test | rank_loss {test_loss.item()}, averaged_recto_rate {test_recto_rate.item()}")
            
        # save model checkpoint
        savepath = f"{pre}/models/epoch_{epoch}.pt"
        torch.save(model.state_dict(), savepath)
        print(f"End of epoch: save model of epoch {epoch} to {savepath}.")
            
    
if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    Seed_everything()
    
    parser = argparse.ArgumentParser(description='Train your own TankBind model.')

    # Dataset configuration
    parser.add_argument("--sampler_batch_size", type=int, default=6,
                        help="batch size.")
    parser.add_argument("--validation_batch_size", type=int, default=8,
                        help="batch size.")
    
    # Model configuration
    parser.add_argument("--pair_threshold", type=float, default=2,
                        help="threshold for sample pairs.")
    parser.add_argument("--readout_mode", type=int, default=1,
                        help="readout_mode.")
    parser.add_argument("--use_mse_loss", type=bool, default=False,
                        help="use mse loss instead of pairwise loss during training")
    
    parser.add_argument("--output_func", type=str, default="no",
                        help="apply a function to model output")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="learning_rate of optimizer")
    
    parser.add_argument("--sigmoid_lambda", type=float, default=0.5,
                        help="factor in pairwise loss")    
    # Customized path and strings
    parser.add_argument("--resultFolder", type=str, default="./result/",
                        help="information you want to keep a record.")
    
    parser.add_argument("--label", type=str, default="",
                        help="information you want to keep a record.")
    

    parser.add_argument("--restart", type=str, default=False,
                        help="continue the training from the model we saved.")
    parser.add_argument("--restart_epoch", type=int, default=-1,
                        help="continue the training from the model we saved.")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="define the model will be trained for how many epochs")
       

    args = parser.parse_args()
    main(args)

 
# Trash
def demode_arguments(parser):
    # TODO: Session Mode ?
    # TODO: to be replaced with SessionDynamicSampler
    parser.add_argument("--sample_n", type=int, default=20000,
                        help="number of samples in one epoch.")
    parser.add_argument("--split_mode", type=int, default=0,
                        help="Mode of split applicated to dataset.")
    parser.add_argument("--use_affinity_mask", type=int, default=0,
                        help="mask affinity in loss evaluation based on data.real_affinity_mask") 
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
    parser.add_argument("-m", "--mode", type=int, default=0,
                        help="mode specify the model to use.")
    parser.add_argument("-d", "--data", type=str, default="0",
                        help="data specify the data to use.")   
    # demode

    parser.add_argument("--affinity_loss_mode", type=int, default=1,
                        help="define which affinity loss function to use.")
    parser.add_argument("--decoy_gap", type=int, default=1,
                        help="define deocy gap used in args.affinity_loss_mode=1")

    parser.add_argument("--pred_dis", type=int, default=1,
                        help="pred distance map or predict contact map.")
    # parser.add_argument("--posweight", type=int, default=8,
    #                     help="pos weight in pair contact loss, not useful if args.pred_dis=1")


    # demode
    parser.add_argument("--relative_k", type=float, default=0.01,
                        help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
    parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                        help="define how the relative_k changes over epochs")
    parser.add_argument("--warm_up_epochs", type=int, default=15,
                        help="used in combination with relative_k_mode.")
    parser.add_argument("--data_warm_up_epochs", type=int, default=0,
                        help="option to switch training data after certain epochs.")
    parser.add_argument("--resultFolder", type=str, default="../",
                        help="information you want to keep a record.")
    # demode
    parser.add_argument("--use_contact_loss", type=int, default=0,
                        help="whether to upgrade contact loss during training, 0 means both, other means only contact loss")
    parser.add_argument("--contact_loss_mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help="choose contact loss mode, 0 means dis^2, 1 means e^dis, 2 means 2^dis, 3,4,5 means dis^3,4,5")
    # demode
    parser.add_argument("--use_weighted_rmsd_loss", type=bool, default=False,
                        help="whether to change contact weight according to distance")

    
    
        
#     model.eval()
#     use_y_mask = args.use_equivalent_native_y_mask or args.use_y_mask



    
#     metrics = evaluate_with_affinity(all_pocket_valid_loader, model, contact_criterion, affinity_criterion, args.relative_k, device, pred_dis=pred_dis, info=info_va, saveFileName=saveFileName)
#     if metrics["auroc"] <= best_auroc and metrics['f1_1'] <= best_f1_1:
#         # not improving. (both metrics say there is no improving)
#         epoch_not_improving += 1
#         ending_message = f" No improvement +{epoch_not_improving}"
#     else:
#         epoch_not_improving = 0
#         if metrics["auroc"] > best_auroc:
#             best_auroc = metrics['auroc']
#         if metrics['f1_1'] > best_f1_1:
#             best_f1_1 = metrics['f1_1']
#         ending_message = " "
#     valid_metrics_list.append(metrics)
#     logging.info(f"epoch {epoch:<4d}, single_valid, " + print_metrics(metrics) + ending_message)

#     writer.add_scalar('epochLoss.Total/validation', metrics["loss"], epoch)
#     writer.add_scalar('epochLoss.Contact/validation', metrics["loss_contact"], epoch)
#     writer.add_scalar('epochLoss.Contact_5A/validation', metrics["loss_contact_5A"], epoch)
#     writer.add_scalar('epochLoss.Contact_10A/validation', metrics["loss_contact_10A"], epoch)
#     writer.add_scalar('epochLoss.Affinity/validation', metrics["loss_affinity"], epoch)
#     writer.add_scalar('epochMetric.RMSE/validation', metrics["RMSE"], epoch)
#     writer.add_scalar('epochMetric.Pearson/validation', metrics["Pearson"], epoch)
#     writer.add_scalar('epochMetric.NativeAUROC/validation', metrics["native_auroc"], epoch)
#     writer.add_scalar('epochMetric.SelectedAUROC/validation', metrics["selected_auroc"], epoch)
#     #====================test============================================================


#     saveFileName = f"{pre}/results/test_epoch_{epoch}.pt"
#     metrics = evaluate_with_affinity(test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
#                                      device, pred_dis=pred_dis, saveFileName=saveFileName, use_y_mask=use_y_mask)
#     test_metrics_list.append(metrics)
#     logging.info(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))


#     saveFileName = f"{pre}/results/single_epoch_{epoch}.pt"
#     metrics = evaluate_with_affinity(all_pocket_test_loader, model, contact_criterion, affinity_criterion, args.relative_k,
#                                      device, pred_dis=pred_dis, info=info, saveFileName=saveFileName)
#     logging.info(f"epoch {epoch:<4d}, single," + print_metrics(metrics))
#     writer.add_scalar('epochLoss.Total/test', metrics["loss"], epoch)
#     writer.add_scalar('epochLoss.Contact/test', metrics["loss_contact"], epoch)
#     writer.add_scalar('epochLoss.Contact_5A/test', metrics["loss_contact_5A"], epoch)
#     writer.add_scalar('epochLoss.Contact_10A/test', metrics["loss_contact_10A"], epoch)
#     writer.add_scalar('epochLoss.Affinity/test', metrics["loss_affinity"], epoch)
#     writer.add_scalar('epochMetric.RMSE/test', metrics["RMSE"], epoch)
#     writer.add_scalar('epochMetric.Pearson/test', metrics["Pearson"], epoch)
#     writer.add_scalar('epochMetric.NativeAUROC/test', metrics["native_auroc"], epoch)
#     writer.add_scalar('epochMetric.SelectedAUROC/test', metrics["selected_auroc"], epoch)

#     if epoch % 1 == 0:
#         torch.save(model.state_dict(), f"{pre}/models/epoch_{epoch}.pt")
#     # torch.save((y, y_pred), f"{pre}/results/epoch_{epoch}.pt")
#     if epoch_not_improving > 100:
#         # early stop.
#         print("early stop")
#         break
    
#     # torch.cuda.empty_cache()
#     os.system(f"cp {timestamp}.log {pre}/")

# torch.save((metrics_list, valid_metrics_list, test_metrics_list), f"{pre}/metrics.pt")
