import torch
from torch.utils.data.distributed import DistributedSampler
from .dataloader import LipReadingDataset

def dataloader(dataset_cfg, batch_size, num_gpus):

    dataset = LipReadingDataset(split='train', **dataset_cfg)

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    return trainloader