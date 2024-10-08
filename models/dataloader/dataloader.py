import os
import sys
import torch
import glob
from pathlib import Path
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from .utils import *
from PIL import Image, ImageEnhance
import cv2
import numpy as np

def files_to_list(data_path, suffix): # suffix = 'mp4': find all mp4 files in data_path
    """
    Load all .mp4 files in data_path
    """
    files = glob(os.path.join(data_path, f'**/*.{suffix}'), recursive=True)
    return files

class LipReadingDataset(Dataset):
    def __init__(self, split, videos_dir, mouthrois_dir, videos_dir_val, mouthrois_dir_val, ratio=0.1, ):
        self.mouthrois_dir = mouthrois_dir
        self.videos_dir = videos_dir
        self.split = split
        self.videos_list = sorted(files_to_list(self.videos_dir, 'mp4'))
        self.mouthrois_list = sorted(files_to_list(self.mouthrois_dir, '.mp4'))
        self.ratio = ratio

        if self.split == 'train':
            self.videos_list_train = self.videos_list[:int(self.ratio*len(self.videos_list))]
            self.mouthrois_list_train = self.mouthrois_list[:int(self.ratio*len(self.mouthrois_list))]
            self.videos_list_val = self.videos_list[int(self.ratio*len(self.videos_list)):]
            self.mouthrois_list_val = self.mouthrois_list[int(self.ratio*len(self.mouthrois_list)):]
        else:
            self.videos_list_train = self.videos_list
            self.mouthrois_list_train = self.mouthrois_list
            self.videos_list_val = sorted(files_to_list(videos_dir_val, 'mp4'))
            self.mouthrois_list_val = sorted(files_to_list(mouthrois_dir_val, '.mp4'))

        



    def __len__(self):
        return len(self.videos_list)
    

    pass


