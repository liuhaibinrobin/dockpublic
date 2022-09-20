import warnings
from typing import Iterator, List, Optional

from torch_geometric.data.hetero_data import HeteroData

from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


class DistributedDynamicBatchSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        dyn_max_num (int):
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will always be used.
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

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.dyn_max_num = dyn_max_num
        self.dyn_mode = dyn_mode
        self.dyn_num_steps = dyn_num_steps
        self.dyn_skip_too_big = dyn_skip_too_big

        self.num_samples_floor = len(self.dataset) / self.num_replicas
        self.total_size = len(self.dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
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
        assert len(indices) == self.num_samples_floor + (self.rank < self.total_size % self.num_replicas)

        batch = []
        batch_n = 0
        num_steps = 0
        num_processed = 0

        while (num_processed < len(indices)) and num_steps < self.dyn_num_steps:
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

                batch.append(idx.item())
                num_processed += 1
                batch_n += n

            if batch:
                yield batch
                batch = []
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