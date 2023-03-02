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
        #todo small session num test
        # _indices={}
        # for tmp_i,key in enumerate(indices):
        #     if tmp_i<100:
        #         _indices[key]=indices[key]
        # indices=_indices
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
        group_cache=[]

        for group in self.batches:
            if len(group) > 1:

                if len(group_cache)+len(group)<self.n:
                    group_cache+=group
                else:
                    yield group_cache
                    group_cache=[]
            else:
                continue
        if group_cache!=[]:
            yield group_cache
    def __len__(self) -> int:
        return len(self.batches)


class DistributedSessionBatchSampler(Sampler[T_co]):
    r"""DistributedSessionBatchSampler.

    """

    def protein_edge_index__init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = None,
                 index_save_path: Optional[str] = None
                ) -> None:

        print("Initializing DDPS with parameters:")

        self.index_save_path=index_save_path
        self.group_info_saved = False

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0


        self.shuffle = shuffle
        if seed is None:
            raise ValueError("seed must be assigned")
        self.seed = seed


    def __iter__(self) -> Iterator[T_co]:
        # if self.shuffle:
        #     # deterministically shuffle based on epoch and seed
        #     g = torch.Generator()
        #     g.manual_seed(self.seed+self.rank+self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g, dtype=torch.long)  # type: ignore[arg-type]
        # else:
        #     indices = torch.arange(len(self.dataset), dtype=torch.long)  # type: ignore[arg-type]
        #
        self.total_size = len(self.batches) // self.num_replicas * self.num_replicas
        self.num_batch = self.total_size // self.num_replicas
        # print("Batched_indices before split", batched_indices)
        batched_indices = self.batches[self.rank:self.total_size:self.num_replicas]
        for group in batched_indices:
            if len(group) > 1:
                yield group
            else:
                continue

    def prepare_batches_for_epoch(self, epoch):
        seed = self.seed + epoch
        print(f"SessionBatchSampler | Refreshing batches with seed {seed} for epoch {epoch}.")
        np.random.seed(seed)
        indices = self.dataset.data.groupby("session").indices
        #todo small session num test
        # _indices={}
        # for tmp_i,key in enumerate(indices):
        #     if tmp_i<100:
        #         _indices[key]=indices[key]
        # indices=_indices
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

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

