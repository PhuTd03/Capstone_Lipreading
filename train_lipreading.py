import os
import sys
import numpy as np
import time
import torch.multiprocessing as mp

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from utils import local_directory

from models import dataloader

from distributed_utils import init_distributed, reduce_tensor, apply_gradient_allreduce

import hydra
from omegaconf import DictConfig, OmegaConf

def distributed_train(rank, num_gpus, group_name, cfg):
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    else:
        train(
            rank=rank, num_gpus=num_gpus,
            dataset_cfg=cfg.dataset,
            model_cfg=cfg.model,
            optimizer_cfg=cfg.optimizer,
            scheduler_cfg=cfg.scheduler,
            **cfg.train,
        )

def train(
        rank, num_gpus, save_dir,
        model_cfg, dataset_cfg, optimizer_cfg, scheduler_cfg,
        ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
        learning_rate,
        name=None,
): # config train
    if rank == 0:
        writer = SummaryWriter(save_dir)
    
    model_name, checkpoint_dir = local_directory(name, model_cfg, save_dir, "checkpoint")

    # load training data
    train_loader = dataloader(dataset_cfg, num_gpus)
    print('Data Loaded!')



@hydra.main(config_path='configs/', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False) # set false to allow adding new keys in config
    
    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)
    
    num_gpus = torch.cuda.device_count()
    print("There are {num_gpus} GPUs available.")

    train_fn = partial(
        train = distributed_train,
        num_gpus = num_gpus,
        group_name = time.strftime("%Y%m%d-%H%M%S"),
        cfg = cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        process = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            process.append(p)
        for p in process:
            p.join()

if __name__ == "__main__":
    main()
