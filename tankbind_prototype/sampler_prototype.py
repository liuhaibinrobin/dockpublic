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
"""
this script achieve 3 functions to speed the training process:
1. two or more different sessions can merge into one batch
2.batch size is dynamic, according to the sum of protein nodes and ligand nodes
3.can be run on ddp mode
"""

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
                 max_batch_size = 6,
                 seed: Optional[int] = None,
                 name: Optional[str] = None,
                 index_save_path: Optional[str] = None,
                 mode=None,
                 max_indication_num=None):#session_ap_p_node  session_au_p_node
        if mode is None:
            raise Exception
        if max_indication_num is None:
            raise Exception
        self.max_indication_num=max_indication_num
        self.mode=mode
        self.dataset = dataset
        self.seed_init = seed
        self.epoch = 0
        self.max_batch_size = max_batch_size
        self.group_info_saved = False
        self.name = name + "_" if name is not None else ""
        self.index_save_path = index_save_path
        self.batches = None
        self.prepare_batches_for_epoch(self.epoch)
        
        
    def prepare_batches_for_epoch(self, epoch):
        seed = self.seed_init + epoch
        print(f"SessionBatchSampler | Refreshing batches with seed {seed} for epoch {epoch}.")
        np.random.seed(seed)

        # indices case
        # {'101010_5LP1': array([49, 50]),
        #  '208_3RDE': array([29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        #                     46, 47, 48]),
        #  '52_1I5R': array([0, 2, 4, 7, 8]),
        #  '52_3HB4': array([1, 3, 5, 6, 9]),
        #  '80_1W6J': array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        #                    27, 28])}
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
        all_indices = [all_random(a=indices[session_name], n=self.max_batch_size) for session_name in indices]
        indices = []
        for _ in all_indices:
            indices.extend(_)
        if self.index_save_path is not None:
            print(f"SessionBatchSampler | Saving group indices for epoch {epoch}.")
            torch.save(indices, f"{self.index_save_path}/batch_in_epoch_{epoch}_with_seed_{seed}.pt")
        self.batches = indices

    def __iter__(self) -> Iterator[List[int]]:
        group_cache=[]
        indication_cache=0
        for group in self.batches:
            if len(group) > 1:
                #print("group_"+str(group))
                if self.mode == "session_ap_p_node":
                    indication=self.dataset.data.iloc[group[0]]["p_length"]
                else:
                    raise Exception

                if len(group_cache)+len(group)<=self.max_batch_size and indication_cache+indication<=self.max_indication_num:
                    group_cache+=group
                    indication_cache+=indication

                else:
                    #print("group_cache_a"+str(group_cache))
                    if group_cache == []:
                        raise Exception
                    yield group_cache
                    group_cache=[]
                    group_cache+=group
                    indication_cache=indication
            else:
                continue
        if group_cache!=[]:
            #print("group_cache_b" + str(group_cache))
            if group_cache == []:
                raise Exception
            yield group_cache

    #dynamic sampler is without definitive length
    # def __len__(self) -> int:
    #     return len(self.batches)


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





class InstanceDynamicBatchSampler(torch.utils.data.sampler.Sampler):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges). When data samples have a
    wide range in sizes, specifying a mini-batch size in terms of number of
    samples is not ideal and can cause CUDA OOM errors.

    Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
    ambiguous, depending on the order of the samples. By default the
    :meth:`__len__` will be undefined. This is fine for most cases but
    progress bars will be infinite. Alternatively, :obj:`num_steps` can be
    supplied to cap the number of mini-batches produced by the sampler.

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num (int): Size of mini-batch to aim for in number of nodes or
            edges.
        mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
            batch size. (default: :obj:`"node"`)
        shuffle (bool, optional): If set to :obj:`True`, will have the data
            reshuffled at every epoch. (default: :obj:`False`)
        skip_too_big (bool, optional): If set to :obj:`True`, skip samples
            which cannot fit in a batch by itself. (default: :obj:`False`)
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying examples, but :meth:`__len__` will be :obj:`None` since
            it is be ambiguous. (default: :obj:`None`)
        error_path (str, optional): 如果需要填一下，会保存碰到的错误数据
    """

    def __init__(self, dataset: Dataset, max_num: int, mode: str = 'node',
                 shuffle: bool = False, skip_too_big: bool = False,
                 num_steps: Optional[int] = None,
                 # New parameters
                 seed: Optional[int] = None,
                 error_path=None):
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`max_num` should be a positive integer value "
                             "(got {max_num}).")
        if mode not in ['node', 'edge']:
            raise ValueError("`mode` choice should be either "
                             f"'node' or 'edge' (got '{mode}').")

        if num_steps is None:
            num_steps = len(dataset)

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.error_path = error_path
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        batch_n = 0
        num_steps = 0
        num_processed = 0

        if self.shuffle:
            if self.seed is None:
                indices = torch.randperm(len(self.dataset), dtype=torch.long)  # type: ignore[arg-type]
            else:
                g = torch.Generator()
                g.manual_seed(self.seed)
                indices = torch.randperm(len(self.dataset), generator=g, dtype=torch.long)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)

        while (num_processed < len(self.dataset) and num_steps < self.num_steps):
            # Fill batch
            for idx in indices[num_processed:]:
                # Size of sample
                data = self.dataset[idx]

                if not isinstance(data, HeteroData):
                    if self.error_path is not None:
                        with open(self.error_path, "a") as f:
                            f.write(str(data) + "\n")
                    continue

                n = data.num_nodes if self.mode == 'node' else data.num_edges

                if batch_n + n > self.max_num:
                    if batch_n == 0:
                        if self.skip_too_big:
                            continue
                        else:
                            warnings.warn("Size of data sample at index "
                                          f"{idx} is larger than "
                                          f"{self.max_num} {self.mode}s "
                                          f"(got {n} {self.mode}s.")
                    else:
                        # Mini-batch filled
                        break

                # Add sample to current batch
                batch.append(idx.item())
                num_processed += 1
                batch_n += n

            if batch != []:
                yield batch
                batch = []
                batch_n = 0
                num_steps += 1
            else:
                break

    def __len__(self) -> int:
        return self.num_steps


class DistributedDynamicBatchSampler(Sampler[T_co]):
    r"""DistributedDynamicSampler.

    """

    def __init__(self,
                 dataset: Dataset,
                 dyn_max_num: int,
                 dyn_mode: Optional[str] = None, dyn_sample_info: Optional[list] = None,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0,
                 dyn_skip_too_big: bool = False,
                 dyn_num_steps: Optional[int] = None) -> None:

        print("Initializing DDPS with parameters:")
        print(f" dataset: {dataset}\n"
              f" dyn_max_num: {dyn_max_num}  dyn_mode: {dyn_mode}  dyn_sample_info: {dyn_sample_info}\n"
              f" num_replicas: {num_replicas}  rank: {rank}  shuffle: {shuffle}  seed: {seed}\n"
              f"dyn_skip_too_big: {dyn_skip_too_big}  dyn_num_steps: {dyn_num_steps}")

        if dyn_mode is None and dyn_sample_info is None:
            raise ValueError("You have to specify `dyn_mode` or `dyn_sample_info`.")
        if dyn_mode is not None and dyn_sample_info is not None:
            raise ValueError("You cannot specify both `dyn_mode` and `dyn_sample_info` at the same time.")
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
        if dyn_num_steps is None:
            dyn_num_steps = len(dataset)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.dyn_max_num = dyn_max_num
        self.dyn_mode = dyn_mode
        self.dyn_num_steps = dyn_num_steps
        self.dyn_skip_too_big = dyn_skip_too_big

        self.num_samples_floor = int(len(self.dataset) / self.num_replicas)
        self.total_size = len(self.dataset)
        self.shuffle = shuffle
        self.seed = seed
        print("Processing dyn_sample_info!")
        if dyn_sample_info:
            self.dyn_sample_info = dyn_sample_info

        else:
            dyn_sample_info = []
            for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
                try:
                    if self.dyn_mode == "node":
                        dyn_sample_info.append(data.num_nodes)
                    else:
                        dyn_sample_info.append(data.num_edges)
                except:
                    dyn_sample_info.append(-1)
            with open(f"dyn_sample_info/dyn_sample_info_{self.rank}.pkl", "wb") as f:
                pickle.dump(dyn_sample_info, f)

        self.dyn_sample_info = dyn_sample_info
        """
        with open(f"dyn_sample_info_{self.rank}.pkl", "wb") as f:
            pickle.dump(dyn_sample_info, f)

        #with open(f"dyn_sample_info_{self.rank}", "rb") as f:
        #    self.dyn_sample_info = pickle.load(f)      
        """
        print("Processed dyn_sample_info!")

    def __iter__(self) -> Iterator[T_co]:
        # print("Epoch", self.epoch, "Seed", self.seed)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g, dtype=torch.long)  # type: ignore[arg-type]
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)  # type: ignore[arg-type]

        # print("Shuffled index", indices)

        batched_indices = []
        batch = []
        batch_n = 0
        num_steps = 0
        for idx in tqdm(indices):
            n = self.dyn_sample_info[idx]
            # print(f"{self.rank}:{i}: -idx",idx," -n", n, " -batch_n", batch_n)
            if n == -1:  # False Sample
                continue
            if batch_n + n > self.dyn_max_num:  # Exceed
                if batch:  # Already has some index in the current batch list
                    num_steps += 1  # Step += 1
                    batched_indices.append(batch)  # Save current batch to batched_indices list
                    if n > self.dyn_max_num:  # If current sample is too large
                        if self.dyn_skip_too_big:  # Skip too large sample
                            batch = []
                            batch_n = 0
                            continue
                        else:  # Or raise warning
                            warnings.warn(f"Size of data sample at index {idx} is larger "
                                          f"than {self.dyn_max_num} {self.dyn_mode}s: got {n} {self.dyn_mode}s.")
                    else:  # If current sample is small enough
                        batch = [idx]
                        batch_n = n
            else:  # Not exceed
                batch.append(idx)
                batch_n += n

        if batch:  # Save last batch if exists.
            batched_indices.append(batch)

        '''
        while num_processed < len(indices) and (num_steps < self.dyn_num_steps):
            #print(self.rank, ":", num_processed, len(indices), num_steps, self.dyn_num_steps)
            for idx in indices[num_processed:]:
                #print(self.rank,"::", idx)
                num_processed += 1
                n = self.dyn_sample_info[idx]
                #print(self.rank, ":: n =", n)
                if n == -1:
                    continue
                if batch_n + n > self.dyn_max_num:
                    if batch_n == 0:
                        if self.dyn_skip_too_big:
                            continue
                        else:
                            warnings.warn(f"Size of data sample at index {idx} is larger "
                                          f"than {self.dyn_max_num} {self.dyn_mode}s: got {n} {self.dyn_mode}s.")
                    else:
                        break

                batch.append(idx.item())
                batch_n += n
            # print(self.rank, ":: after for", batch)
            if batch:
                #print(batch, type(batch))
                num_steps += 1
                batched_indices.append(batch)
                batch = []
                batch_n = 0
        '''

        self.total_size = len(batched_indices) // self.num_replicas * self.num_replicas
        self.num_batch = self.total_size // self.num_replicas
        # print("Batched_indices before split", batched_indices)
        batched_indices = batched_indices[self.rank:self.total_size:self.num_replicas]
        # print("My batched indices", batched_indices)
        # print("Batched_indices", batched_indices)
        # print(f"Rank {self.rank}, Indices {batched_indices}")
        for _batch in batched_indices:
            yield _batch

    def __len__(self) -> int:
        return self.num_batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class OldDistributedDynamicBatchSampler(Sampler[T_co]):
    r"""DynamicBatchSampler for DDP training.

    Args:
        dataset (Dataset): Dataset to sample from.
        dyn_max_num (int): Size of mini-batch to aim for in number of nodes or
            edges.
        dyn_mode (str): :obj:`"node"` or :obj:`"edge"` to measure
            batch size. (default: :obj:`"node"`)
        num_replicas (int, optional): Number of processes participating in distributed
            training. By default, :attr:`world_size` is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        dyn_skip_too_big (bool, optional): If set to :obj:`True`, skip samples
            which cannot fit in a batch by itself. (default: :obj:`False`)
        dyn_num_steps (int, optional): The number of mini-batches to draw for a single
            epoch. If set to :obj:`None`, will iterate through all the underlying examples,
            but :meth:`__len__` will be :obj:`None` since it is be ambiguous. (default: :obj:`None`)
    """

    def __init__(self,
                 dataset: Dataset,
                 dyn_max_num: int, dyn_mode: str = "node",
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0,  # drop_last: bool = False,
                 dyn_skip_too_big: bool = False, dyn_num_steps: Optional[int] = None
                 ) -> None:
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

        if dyn_num_steps is None:
            dyn_num_steps = len(dataset)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.dyn_max_num = dyn_max_num
        self.dyn_mode = dyn_mode
        self.dyn_num_steps = dyn_num_steps
        self.dyn_skip_too_big = dyn_skip_too_big

        self.num_samples_floor = int(len(self.dataset) / self.num_replicas)
        self.total_size = len(self.dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        print("ITER!", self.seed)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g, dtype=torch.long)  # type: ignore[arg-type]
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)  # type: ignore[arg-type]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        print(indices)
        print(self.rank, self.total_size, self.num_replicas, self.num_samples_floor)
        assert len(indices) == self.num_samples_floor + (self.rank < self.total_size % self.num_replicas)

        batch_indices = []
        batch_n = 0
        num_steps = 0
        num_processed = 0

        while (num_processed < len(indices)) and (num_steps < self.dyn_num_steps):
            # Fill batch
            for idx in indices[num_processed:]:
                # Size of sample
                data = self.dataset[idx]

                if not isinstance(data, HeteroData):
                    continue
                n = data.num_nodes if self.dyn_mode == "node" else data.num_deges

                if batch_n + n > self.dyn_max_num:
                    if batch_n == 0:
                        if self.dyn_skip_too_big:
                            continue
                        else:
                            warnings.warn(f"Size of data sample at index {idx} is larger "
                                          f"than {self.dyn_max_num} {self.dyn_mode}s: got {n} {self.dyn_mode}s.")
                    else:
                        break

                batch_indices.append(idx.item())
                num_processed += 1
                batch_n += n

            if batch_indices:
                print("BBB", batch_indices, type(batch_indices))
                yield batch_indices
                batch_indices = []
                batch_n = 0
                num_steps += 1
            else:
                break

    def __len__(self) -> int:
        return self.dyn_num_steps

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
