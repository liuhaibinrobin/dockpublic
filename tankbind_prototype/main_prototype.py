# training script for new model
# prototype 2022-02-06

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
from torch import nn
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
from data_prototype import get_full_data_prototype, get_internal_dataset
from sampler_prototype import SessionBatchSampler
from model import *


class PairwiseLoss(nn.Module):
    def __init__(self, keep_rate=1., sigmoid_lambda=0.5,
                 ingrp_thr=2, outgrp_thr=9999, eval=False):
        """

        :param keep_rate:
        :param sigmoid_lambda:
        :param ingrp_thr: =2时是3倍组pair
        :param outgrp_thr:
        :param eval:
        """
        super(PairwiseLoss, self).__init__()
        self.eval = eval
        self.register_buffer('keep_rate', torch.tensor(keep_rate, dtype=torch.float64))
        self.register_buffer('sigmoid_lambda', torch.tensor(sigmoid_lambda, dtype=torch.float64))
        self.register_buffer('ingrp_thr', torch.tensor(ingrp_thr, dtype=torch.float64))
        self.register_buffer('outgrp_thr', torch.tensor(outgrp_thr, dtype=torch.float64))

    def forward(self, pred, true):
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
        # print("Total valid pairs: {:.3f}, reversed: {:.3f}, reverse ratio: {:.3f}".format(ntotal, reverse, reverse/ntotal))
        if drop_rate > 1e-4:
            pred_pair_valid_dropout = torch.nn.functional.dropout(pred_pair_valid[0], drop_rate) * self.keep_rate
            pred_pair_valid_ind = torch.logical_or(pred_pair_valid_dropout == pred_pair_valid[0],
                                                   pred_pair_valid_dropout != 0.0)

            pred_pair_valid = pred_pair_valid.masked_select(pred_pair_valid_ind).reshape(2, -1)
        # print(pred_pair_valid[1] - pred_pair_valid[0])
        loss = torch.sum(torch.log(1. + torch.exp(self.sigmoid_lambda * (pred_pair_valid[1] - pred_pair_valid[0] + 1))))
        num = pred_pair_valid.shape[1] + 1e-8
        loss = torch.div(loss, num)
        return loss, (reverse / ntotal), ntotal


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
    #writer = SummaryWriter(f"./tensorboard/{timestamp}{w_label}")
    writer = SummaryWriter(f"./tensorboard/tmp") #todo

    train_flag = True if "train" in args.run_mode else False
    iid_flag = True if "iid" in args.run_mode else False
    ood_flag = True if "ood" in args.run_mode else False
    test_flag = True if "test" in args.run_mode else False
    internal_flag = True if "internal" in args.run_mode else False

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("")
    handler = logging.FileHandler(f'log/{timestamp}.log')
    handler.setFormatter(logging.Formatter('%(message)s', ""))
    logger.addHandler(handler)

    logging.info(
        f"{' '.join(sys.argv)}\n" 
        f"-----------------------------------------------------------------\n" 
        f"{(args.label if args.label!='' else 'UNNAMED')}\n"
        f" at {timestamp}\n"
        f" data mode: {args.data_mode}\n" 
        f" run mode: {args.run_mode}\n" 
        f"-----------------------------------------------------------------\n" 
        f"{args}\n" 
        f"-----------------------------------------------------------------\n"
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
    print(f"Getting datasets from {input_path}...", end="")
    if train_flag or iid_flag or ood_flag or test_flag:
        train_dataset, iid_dataset, ood_dataset, test_dataset = get_full_data_prototype(input_path, random_state=args.trainset_seed, data_mode=args.data_mode,session_type=args.session_type)
    else:
        train_dataset = iid_dataset = ood_dataset = test_dataset = None
    if internal_flag:
        internal_dataset = get_internal_dataset(input_path,session_type=args.session_type)
    else:
        internal_dataset = None
    print("fin!")
    logging.info(
        # f"train: {(len(train_dataset.data) if train_dataset is not None else 0)}\n" 
        f"iid_val: {(len(iid_dataset.data) if iid_dataset is not None and iid_dataset.data is not None else 0)}\n" 
        f"ood_val: {(len(ood_dataset.data) if ood_dataset is not None and ood_dataset.data is not None  else 0)}\n" 
        f"test: {(len(test_dataset.data) if test_dataset is not None and test_dataset.data is not None else 0)}\n" 
        f"internal: {(len(internal_dataset.data) if internal_dataset is not None and internal_dataset.data is not None else 0)}\n"
        f"=================================================================\n"
    )

    num_workers = 10

    os.system(f"mkdir -p {pre}/train/batch_split_info")
    os.system(f"mkdir -p {pre}/train/epoch_result")
    print("Processing train_sampler and train_dataloader...", end="")
    if train_flag:
        train_sampler = SessionBatchSampler(train_dataset, n=args.sampler_batch_size, seed=0, name=timestamp,
                                            index_save_path=f"{pre}/train/batch_split_info")
        train_dataloader = DataLoader(train_dataset,
                                    follow_batch=['x', 'compound_pair'],
                                    batch_sampler=train_sampler,
                                    pin_memory=False,
                                    num_workers=8)
    print("fin!")

    valid_batch_size = 8

    print("Processing val/test dataloaders...", end="")
    if iid_flag:
        iid_dataloader = DataLoader(iid_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'],
                                    shuffle=False, pin_memory=False, num_workers=num_workers)
    if ood_flag:
        ood_dataloader = DataLoader(ood_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'],
                                    shuffle=False, pin_memory=False, num_workers=num_workers)
    if test_flag:
        test_dataloader = DataLoader(test_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'],
                                    shuffle=False, pin_memory=False, num_workers=num_workers)
    if internal_flag:
        internal_dataloader = DataLoader(internal_dataset, batch_size=valid_batch_size, follow_batch=['x', 'compound_pair'],
                                        shuffle=False, pin_memory=False, num_workers=num_workers)
    print("fin!")

    device = 'cuda:0'

    print("Loading model, creating optimizer and loss function...", end="")
    model = get_model(0, logging, device, readout_mode=args.readout_mode, output_func=args.output_func)
    if args.restart:
        model.load_state_dict(torch.load(args.restart))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    pairwiseloss = PairwiseLoss(sigmoid_lambda=args.sigmoid_lambda, ingrp_thr=args.pair_threshold-1)
    if args.use_mse_loss:
        mseloss = MSELoss()
    print("fin!")

    num_steps_train = 0
    num_samples_train = 0

    for epoch in range(args.num_epochs):
        if (args.restart is not False) and (epoch <= args.restart_epoch):
            logging.info(f"Epoch {epoch} skipped | continue from epoch {args.restart_epoch}")
            continue
        else:
            logging.info(f"Epoch {epoch} =================================================")
        # TRAIN
        if train_flag:
            num_steps_train, num_samples_train = run_train(args=args,pre=pre, dataloader=train_dataloader, epoch=epoch,
                                                        num_steps_train=num_steps_train, 
                                                        num_samples_train=num_samples_train,
                                                        model=model, optimizer=optimizer, pairwiseloss=pairwiseloss, 
                                                        device=device, writer=writer, logging=logging)
        
        # VAL/TEST
        if iid_flag and iid_dataset is not None and iid_dataset.data is not None:
            run_validation(pre=pre, label="iid", dataset=iid_dataset, dataloader=iid_dataloader, epoch=epoch,
                           model=model, pairwiseloss=(mseloss if args.use_mse_loss else pairwiseloss), 
                           device=device, writer=writer, logging=logging)
        if ood_flag and ood_dataset is not None and ood_dataset.data is not None:
            run_validation(pre=pre, label="ood", dataset=ood_dataset, dataloader=ood_dataloader, epoch=epoch,
                           model=model, pairwiseloss=(mseloss if args.use_mse_loss else pairwiseloss), 
                           device=device, writer=writer, logging=logging)
        if internal_flag and internal_dataset is not None and internal_dataset.data is not None:
            run_validation(pre=pre, label="internal", dataset=internal_dataset, dataloader=internal_dataloader, epoch=epoch,
                           model=model, pairwiseloss=(mseloss if args.use_mse_loss else pairwiseloss), 
                           device=device, writer=writer, logging=logging)
        if test_flag and test_dataset is not None and test_dataset.data is not None:
            run_validation(pre=pre, label="test", dataset=test_dataset, dataloader=test_dataloader, epoch=epoch,
                           model=model, pairwiseloss=(mseloss if args.use_mse_loss else pairwiseloss), 
                           device=device, writer=writer, logging=logging)

        # save model checkpoint
        if args.save_checkpoint:
            torch.save(model.state_dict(), f"{pre}/models/epoch_{epoch}.pt")
            print(f"End of epoch: save model of epoch {epoch} to {pre}/models/epoch_{epoch}.pt.")


def run_train(pre, args, dataloader,
              epoch, model, optimizer,
              pairwiseloss=None, mseloss=None,
              num_steps_train=0, num_samples_train=0,
              device=None, writer=None, logging=None,
              save_train_result=True):
    print(f"TRAIN | epoch {epoch}")
    model.train()
    affinity_true_list = []
    affinity_pred_list = []
    session_list = []
    sample_id_list = []
    loss_list = []
    recto_rate_list = []

    total_loss=0
    total_recto_rate=0
    length=0
    # 更新 batch 分割方式。
    dataloader.batch_sampler.prepare_batches_for_epoch(epoch=epoch)
    for data in tqdm(dataloader):
        num_steps_train += 1
        num_samples_train += len(data)
        session_list.append(data.session)
        sample_id_list.append(data.sample_id)
        data = data.to(device)
        optimizer.zero_grad()
        _, affinity_pred = model(data)
        del _  # Note: for now, y is not in need.
        affinity_true = data.value



        if not args.use_mse_loss:
            loss, recto_rate, num_pairs = pairwiseloss(pred=affinity_pred, true=affinity_true)
        else:
            loss = mseloss(affinity_pred.float(), torch.log10(affinity_true).float())

        if not (abs(loss.item()) > 100000):
            loss.backward()
            optimizer.step()
        else:
            logging.info(
                f"--loss error | epoch {epoch}, session {list(data.session)[0]}, sample_id {str(data.sample_id)}, pred {[str(_) for _ in affinity_pred]}, true {[str(_) for _ in affinity_true]}")

        affinity_true_list.append(affinity_true.detach().cpu())
        affinity_pred_list.append(affinity_pred.detach().cpu())

        loss_list.append(loss.detach().cpu())
        writer.add_scalar(f'rank_loss.by_step/train', loss.item(), num_steps_train)
        writer.add_scalar(f'rank_loss.by_sample/train', loss.item(), num_samples_train)

        if not args.use_mse_loss:
            recto_rate_list.append(recto_rate.detach().cpu())
            writer.add_scalar(f'recto_rate.by_step/train', recto_rate.item(), num_steps_train)
            writer.add_scalar(f'recto_rate.by_sample/train', recto_rate.item(), num_samples_train)
        # end of sample scope

        if not args.use_mse_loss:
            if num_pairs >= 1:
                length += np.sqrt(num_pairs)
                total_loss += loss * np.sqrt(num_pairs)
                total_recto_rate += recto_rate * np.sqrt(num_pairs)

            # save result to tensorboard
            total_loss /= length
            total_recto_rate /= length
            writer.add_scalar(f'rank_loss.by_epoch/train', total_loss, epoch)
            writer.add_scalar(f'recto_rate.by_epoch/train', total_recto_rate, epoch)
            logging.info(f"epoch {epoch} train | rank_loss {total_loss}, averaged_recto_rate {total_recto_rate}")



    if save_train_result:
        affinity_true = torch.cat(affinity_true_list)
        affinity_pred = torch.cat(affinity_pred_list)
        recto_rate = torch.stack(recto_rate_list)
        loss = torch.stack(loss_list)

        if not args.use_mse_loss:
            torch.save(
                {"affinity_true": affinity_true, "affinity_pred": affinity_pred, "loss": loss,
                 "recto_rate": recto_rate, "session_list": session_list, "sample_id_list": sample_id_list},
                f"{pre}/train/epoch_result/epoch_{epoch}.pt")
        else:
            torch.save(
                {"affinity_true": affinity_true, "affinity_pred": affinity_pred, "loss": loss,
                 "session_list": session_list, "sample_id_list": sample_id_list},
                f"{pre}/train/epoch_result/epoch_{epoch}.pt")
        print(f"-- Save result of epoch {epoch} to {pre}/train/epoch_result/epoch_{epoch}.pt.")
        
    return num_steps_train, num_samples_train

def run_validation(pre, dataset, dataloader,
                   model, epoch, label, device,
                   writer, pairwiseloss, logging):
    # VALIDATION
    print(f"VALTEST: {label} | Epoch {epoch}")
    model.eval()
    with torch.no_grad():
        info = dataset.data
        sample_id = []
        affinity_pred_list = []
        result = []
        length = 0
        total_loss = 0
        total_recto_rate = 0

        # sample scope
        print(f"---- Compute affinity ---------------------")
        for data in tqdm(dataloader):
            sample_id.extend(data.sample_id)
            data = data.to(device)
            _, affinity_pred = model(data)
            del _
            affinity_pred_list.append(affinity_pred.detach().cpu())
        # end of sample scope

        # write to DataFrame
        info["dataset_sample_id"] = sample_id
        info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()

        print(f"---- Compute loss and recto_ratio ---------")
        for session in info.session.unique():
            df = info[info.session == session]
            affinity_true = torch.tensor(df.value.values).to(device)
            affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
            
            
            torch.save(affinity_pred.cpu(), f"session_{session}_pred.pt")
            torch.save(affinity_pred.cpu(), f"session_{session}_true.pt")
            
            loss, recto_rate, num_pairs = pairwiseloss(pred=affinity_pred, true=affinity_true)
            loss = loss.detach().cpu().item()
            recto_rate = recto_rate.detach().cpu().item()
            session_len = len(df)

            #20230220日改（殷实秋）：num_pairs=1依然可以计算正序率，sample数=1才不可以计算
            #if num_pairs >= 2:
            if num_pairs >= 1:
                length += np.sqrt(num_pairs)
                total_loss += loss * np.sqrt(num_pairs)
                total_recto_rate += recto_rate * np.sqrt(num_pairs)
                result.append([session, session_len, num_pairs, loss, recto_rate, "used"])
            else:
                result.append([session, session_len, num_pairs, loss, recto_rate, "not used"])
            # end of sample scope: ratio and loss calculate

    # save result as .csv
    result = pd.DataFrame(result, columns=["session", "length", "num_pairs", "loss", "recto_rate", "use"])
    result.to_csv(f"{pre}/results/{label}_result_{epoch}.csv")
    info.to_csv(f"{pre}/results/{label}_info_{epoch}.csv")
    print(f"VALTEST: {label} | Save result of epoch {epoch} to {pre}/results/{label}_result_{epoch}.csv.")
    print(f"VALTEST: {label} | Save info of epoch {epoch} to {pre}/results/{label}_info_{epoch}.csv.")

    # save result to tensorboard
    total_loss /= length
    total_recto_rate /= length
    writer.add_scalar(f'rank_loss.by_epoch/{label}', total_loss, epoch)
    writer.add_scalar(f'recto_rate.by_epoch/{label}', total_recto_rate, epoch)
    logging.info(f"epoch {epoch} {label} | rank_loss {total_loss}, averaged_recto_rate {total_recto_rate}")


if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    Seed_everything()

    parser = argparse.ArgumentParser(description='Train your own TankBind model.')

    parser.add_argument("--run_mode", type=str, default="iid_ood_test_internal",
                        help="which actions will be taken")

    # Dataset configuration
    parser.add_argument("--sampler_batch_size", type=int, default=6,
                        help="batch size.")
    parser.add_argument("--validation_batch_size", type=int, default=8,
                        help="batch size.")
    parser.add_argument("--data_mode", type=str, default="full",
                        help="use 'full' or reduced 'small' dataset")
    parser.add_argument("--trainset_seed", type=int, default=0,
                        help="Random seed to generate training set")
    # Model configuration
    parser.add_argument("--pair_threshold", type=float, default=2,
                        help="threshold for the formation of sample pairs."
                             "For example with default argument value 2, sample A and sample B would be grouped "
                             "if `A >= 2B + d` (vice versa, and d is a very small variable).")
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
    parser.add_argument("--save_checkpoint", type=bool, default=True,
                        help="whether save checkpoint in the end of each epoch")
    parser.add_argument("--restart", type=str, default=False,
                        help="continue the training from the model we saved.")
    parser.add_argument("--restart_epoch", type=int, default=-1,
                        help="continue the training from the model we saved.")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="define the model will be trained for how many epochs")

    parser.add_argument("--session_type", type=str, default="session_au",
                        help="session_au/session_ap  assay_uniprot as session or assay_pdb as session ")
    args = parser.parse_args()
    main(args)


# # Trash
# def demode_arguments(parser):
#     # TODO: Session Mode ?
#     # TODO: to be replaced with SessionDynamicSampler
#     parser.add_argument("--sample_n", type=int, default=20000,
#                         help="number of samples in one epoch.")
#     parser.add_argument("--split_mode", type=int, default=0,
#                         help="Mode of split applicated to dataset.")
#     parser.add_argument("--use_affinity_mask", type=int, default=0,
#                         help="mask affinity in loss evaluation based on data.real_affinity_mask")
#     parser.add_argument("--addNoise", type=str, default=None,
#                         help="shift the location of the pocket center in each training sample \
#                         such that the protein pocket encloses a slightly different space.")
#     pair_interaction_mask = parser.add_mutually_exclusive_group()
#     # use_equivalent_native_y_mask is probably a better choice.
#     pair_interaction_mask.add_argument("--use_y_mask", action='store_true',
#                                        help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
#                         real_y_mask=True if it's the native pocket that ligand binds to.")
#     pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true',
#                                        help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
#                         real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")
#     parser.add_argument("-m", "--mode", type=int, default=0,
#                         help="mode specify the model to use.")
#     parser.add_argument("-d", "--data", type=str, default="0",
#                         help="data specify the data to use.")
#     # demode

#     parser.add_argument("--affinity_loss_mode", type=int, default=1,
#                         help="define which affinity loss function to use.")
#     parser.add_argument("--decoy_gap", type=int, default=1,
#                         help="define deocy gap used in args.affinity_loss_mode=1")

#     parser.add_argument("--pred_dis", type=int, default=1,
#                         help="pred distance map or predict contact map.")
#     # parser.add_argument("--posweight", type=int, default=8,
#     #                     help="pos weight in pair contact loss, not useful if args.pred_dis=1")

#     # demode
#     parser.add_argument("--relative_k", type=float, default=0.01,
#                         help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
#     parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
#                         help="define how the relative_k changes over epochs")
#     parser.add_argument("--warm_up_epochs", type=int, default=15,
#                         help="used in combination with relative_k_mode.")
#     parser.add_argument("--data_warm_up_epochs", type=int, default=0,
#                         help="option to switch training data after certain epochs.")
#     parser.add_argument("--resultFolder", type=str, default="../",
#                         help="information you want to keep a record.")
#     # demode
#     parser.add_argument("--use_contact_loss", type=int, default=0,
#                         help="whether to upgrade contact loss during training, 0 means both, other means only contact loss")
#     parser.add_argument("--contact_loss_mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
#                         help="choose contact loss mode, 0 means dis^2, 1 means e^dis, 2 means 2^dis, 3,4,5 means dis^3,4,5")
#     # demode
#     parser.add_argument("--use_weighted_rmsd_loss", type=bool, default=False,
#                         help="whether to change contact weight according to distance")
