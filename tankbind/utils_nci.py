import torch
from torch import nn


class NCICriterion(nn.Module):
    def __init__(self, class_weight, under_sampling_ratio):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.ratio = under_sampling_ratio

    def forward(self, nci_pred, nci_true):
        ## nci_pred: length == nci_true
        true_indices = nci_true.nonzero()
        false_indices = (nci_true != True).nonzero()
        selected_false_indices = false_indices[
            torch.randperm(len(false_indices))[0:len(false_indices) * self.undersampling_ration]]
        selected_indices = torch.cat(true_indices, selected_false_indices)
        return self.criterion(nci_pred[selected_indices], nci_true[selected_indices])

def weighted_rmsd_loss(y_pred, y_true):
    return torch.mean(100 * (1 / (y_true ** 2)) * (y_pred - y_true) ** 2)