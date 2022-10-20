import math

import torch
from torch import nn
from tqdm import tqdm

from metrics import myMetric, affinity_metrics
from utils import cut_off_rmsd, select_pocket_by_predicted_affinity, compute_numpy_rmse, \
    extract_list_from_prediction
import torchmetrics
import pdb

class NCICriterion(nn.Module):
    def __init__(self, class_weight, under_sampling_ratio):
        super().__init__()
        if class_weight.dtype != torch.float32:
            class_weight = class_weight.float()
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.ratio = under_sampling_ratio
    def forward(self, nci_pred, nci_true):
        true_indices = nci_true.nonzero()
        false_indices = (nci_true != True).nonzero()
        selected_false_indices = false_indices[
            torch.randperm(len(false_indices))[0:len(true_indices) * self.ratio]]
        selected_indices = torch.cat((true_indices, selected_false_indices)).squeeze(-1)
        # print(nci_pred[selected_indices].shape, nci_true[selected_indices].squeeze(-1).shape, 
        #      nci_pred[selected_indices].dtype, nci_true[selected_indices].squeeze(-1).dtype)
        return self.criterion(nci_pred[selected_indices], nci_true[selected_indices].squeeze(-1).long()), len(selected_indices)


def eval_nci_classification(nci_pred, nci_true):
    nci_pred = nci_pred.argmax(dim=1)
    tp = ((nci_pred==1)&(nci_pred==nci_true)).sum()
    tn = ((nci_pred==0)&(nci_pred==nci_true)).sum()
    fp = ((nci_pred==1)&(nci_pred!=nci_true)).sum()
    fn = ((nci_pred==0)&(nci_pred!=nci_true)).sum()
    accuracy = (tp+tn)/len(nci_pred)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return accuracy, recall, precision, tp, tn, fp, fn


def evaluate_with_affinity_and_nci(data_loader,
                                   model,
                                   contact_criterion,
                                   affinity_criterion,
                                   nci_criterion,
                                   relative_k,
                                   device,
                                   relative_dist_criterion=0.5,
                                   pred_dis=False, info=None, saveFileName=None, use_y_mask=False,
                                   skip_y_metrics_evaluation=False):
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    real_y_mask_list = []
    p_length_list = []
    c_length_list = []
    nci_list = []
    nci_pred_list = []

    epochLoss_contact = 0.0
    epochLoss_contact_5A = 0.0
    epochNum_nan_contact_5A = 0
    epochLoss_contact_10A = 0.0
    epochLoss_nan_contact_10A = 0
    epochLoss_affinity = 0.0
    epochLoss_sampled_nci = 0.0

    #for data in tqdm(data_loader):
    for i, data in enumerate(tqdm(data_loader)):
        protein_ptr = data['protein']['ptr']
        p_length_list += [int(protein_ptr[ptr] - protein_ptr[ptr - 1]) for ptr in range(1, len(protein_ptr))]
        compound_ptr = data['compound']['ptr']
        c_length_list += [int(compound_ptr[ptr] - compound_ptr[ptr - 1]) for ptr in range(1, len(compound_ptr))]
        with torch.no_grad():    
            data = data.to(device)
            y_pred, affinity_pred, nci_pred = model(data)

            y = data.y
            dis_map = data.dis_map
            nci_sequence = data.nci_sequence

            if use_y_mask:
                y_pred = y_pred[data.real_y_mask]
                y = y[data.real_y_mask]
                dis_map = dis_map[data.real_y_mask]
                if nci_pred is not None:
                    nci_sequence = nci_sequence[data.real_nci_mask]
                    nci_pred = nci_pred[data.real_nci_mask]

            ## Computation of contact_loss
            if pred_dis:
                contact_loss = relative_dist_criterion * contact_criterion(y_pred, dis_map) \
                    if len(dis_map) > 0 \
                    else torch.tensor([0]).to(dis_map.device)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=5)):
                    epochNum_nan_contact_5A += len(y_pred)
                    contact_loss_cat_off_rmsd_5 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_5 = cut_off_rmsd(y_pred, dis_map, cut_off=5)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=10)):
                    epochLoss_nan_contact_10A += len(y_pred)
                    contact_loss_cat_off_rmsd_10 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_10 = cut_off_rmsd(y_pred, dis_map, cut_off=10)
            else:
                contact_loss = relative_dist_criterion * contact_criterion(y_pred, y) if len(y) > 0 else torch.tensor(
                    [0]).to(y.device)
                y_pred = y_pred.sigmoid()
            ## Computation of nci_loss when nci_pred is not None
            if nci_pred is not None:
                nci_loss, batchNum_sampled_nci = nci_criterion(nci_pred, nci_sequence)
                nci_loss = relative_dist_criterion * nci_loss * 100
            else:
                nci_loss = torch.tensor([0]).to(y.device)
                batchNum_sampled_nci = 0
            ## Computation of affinity_loss
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)

        
        epochLoss_contact += len(y_pred) * contact_loss.item()
        epochLoss_contact_5A += len(y_pred) * contact_loss_cat_off_rmsd_5.item()
        epochLoss_contact_10A += len(y_pred) * contact_loss_cat_off_rmsd_10.item()
        epochLoss_affinity += len(affinity_pred) * affinity_loss.item()
        if nci_pred is not None:
            if batchNum_sampled_nci != 0:
                epochLoss_sampled_nci += batchNum_sampled_nci * nci_loss.item()
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        if nci_pred is not None:
            nci_list.append(nci_sequence)
            nci_pred_list.append(nci_pred)
        real_y_mask_list.append(data.real_y_mask)

    ## ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ==== TRAIN ==== Fin Iteration ====
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    if pred_dis:
        y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
        # we define 8A as the cutoff for contact, therefore, contact_threshold will be 1 - 8/10 = 0.2
        threshold = 0.2
    real_y_mask = torch.cat(real_y_mask_list)
    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    nci_true = torch.cat(nci_list)
    nci_pred = torch.cat(nci_pred_list)
    nci_accuracy, nci_recall, nci_precision, tp, tn, fp, fn = eval_nci_classification(nci_pred, nci_true)

    if saveFileName:
        torch.save((y, y_pred, affinity, affinity_pred, nci_true, nci_pred), saveFileName)

    metrics = {
        "loss_total": epochLoss_contact / len(y_pred) + epochLoss_affinity / len(affinity_pred)+ (
            epochLoss_sampled_nci / len(nci_pred)),
        "loss_contact": (epochLoss_contact / len(y_pred)),
        "loss_contact_5A": (epochLoss_contact_5A / (len(y_pred) - epochNum_nan_contact_5A)),
        "loss_contact_10A": (epochLoss_contact_10A / (len(y_pred) - epochLoss_nan_contact_10A)),
        "loss_affinity": (epochLoss_affinity / len(affinity_pred)),
        "loss_nci": (epochLoss_sampled_nci / len(nci_pred)),
        "nci_accuracy": nci_accuracy,
        "nci_recall": nci_recall,
        "nci_precision": nci_precision,
        "nci_TP": tp,
        "nci_TN": tn,
        "nci_FP": fp,
        "nci_FN": fn
    }
    if info is not None:
        # print(affinity, affinity_pred)
        info['affinity'] = affinity.cpu().numpy()
        info['affinity_pred'] = affinity_pred.cpu().numpy()
        selected = select_pocket_by_predicted_affinity(info)
        real_affinity = 'real_affinity' if 'real_affinity' in selected.columns else 'affinity'
        # result['Pearson'] = selected['affinity'].corr(selected['affinity_pred'])
        metrics['metric_pearson'] = selected[real_affinity].corr(selected['affinity_pred'])
        metrics['metric_rmse'] = compute_numpy_rmse(selected[real_affinity], selected['affinity_pred'])

        native_y = y[real_y_mask].bool()
        native_y_pred = y_pred[real_y_mask]
        native_auroc = torchmetrics.functional.auroc(native_y_pred, native_y)
        metrics['metric_native_auroc'] = native_auroc
        
        nci_pred_score = nci_pred.softmax(1)[:, 1]

        ## type(nci_true) == torch.float.
        nci_auroc = torchmetrics.functional.auroc(nci_pred_score, nci_true.long())
        metrics['metric_nci_auroc'] = nci_auroc

        info['p_length'] = p_length_list
        info['c_length'] = c_length_list
        y_list, y_pred_list = extract_list_from_prediction(info, y.cpu(), y_pred.cpu(), selected=selected,
                                                           smiles_to_mol_dict=None, coords_generated_from_smiles=False)
        selected_y = torch.cat([y.flatten() for y in y_list]).long()
        selected_y_pred = torch.cat([y_pred.flatten() for y_pred in y_pred_list])
        selected_auroc = torchmetrics.functional.auroc(selected_y_pred, selected_y)
        metrics['metric_selected_metric'] = selected_auroc
        for i in [90, 80, 50]:
            # cover ratio, CR.
            metrics[f'CR_{i}'] = (selected.cover_contact_ratio > i / 100).sum() / len(selected)
    if not skip_y_metrics_evaluation:
        metrics.update(myMetric(y_pred, y, threshold=threshold))
    metrics.update(affinity_metrics(affinity_pred, affinity))
    return metrics
