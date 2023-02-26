import warnings
from typing import Iterator, List, Optional

from torch_geometric.data.hetero_data import HeteroData

from typing import TypeVar, Optional, Iterator
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import numpy as np
T_co = TypeVar('T_co', covariant=True)


def all_random(a, n=6):
    if len(a) <= n and len(a) > 1:
        a = [ list(a) ]
        return a
    else:
        np.random.shuffle(a) #
        rest = len(a)%n
        if rest <= 1:
            b = [list(_) for _ in a[:(-1*rest)].reshape(-1, n)]
        else:
            b = [list(_) for _ in a[:(-1*rest)].reshape(-1, n)] + [list(a[(-1*rest):])]
        return b
    
    
class SessionBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, 
                 dataset: Dataset, 
                 n = 6,
                 seed: Optional[int] = None, 
                 name: Optional[str] = None,
                 index_save_path: Optional[str] = None):

        self.dataset = dataset
        self.seed_init = seed
        self.epoch = 0
        self.n = n
        self.group_info_saved = False
        self.name = name + "_" if name is not None else ""
        self.index_save_path = index_save_path
        self.batches = None
        self.prepare_batches_for_epoch(self.epoch)
        
        
    def prepare_batches_for_epoch(self, epoch):
        seed = self.seed_init + epoch
        print(f"SessionBatchSampler | Refreshing batches with seed {seed} for epoch {epoch}.")
        np.random.seed(seed)
        indices = self.dataset.data.groupby("session").indices
        _indices={}
        for tmp_i,key in enumerate(indices): #todo
            if tmp_i<100:
                _indices[key]=indices[key]
        indices=_indices
        if not self.group_info_saved:
            print(f"SessionBatchSampler | Saving all samples' group indices.")
            torch.save(indices, f"{self.index_save_path}/batch_of_all_sample.pt")
            self.group_info_saved = True
        all_indices = [all_random(a=indices[_], n=self.n) for _ in indices]
        indices = []
        for _ in all_indices:
            indices.extend(_)
        if self.index_save_path is not None:
            print(f"SessionBatchSampler | Saving group indices for epoch {epoch}.")
            torch.save(indices, f"{self.index_save_path}/batch_in_epoch_{epoch}_with_seed_{seed}.pt")
        self.batches = indices

    def __iter__(self) -> Iterator[List[int]]:
        for group in self.batches:
            if len(group) > 1:
                yield group
            else:
                continue
    def __len__(self) -> int:
        return len(self.batches)