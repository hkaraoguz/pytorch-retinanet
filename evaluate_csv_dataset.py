import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval
import sys


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args()
    
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    
    retinanet = torch.load(parser.model)
    
    use_gpu = True
    
    if use_gpu:
        retinanet = retinanet.cuda()
        
    retinanet.eval()
    
    mAP = csv_eval.evaluate(dataset_val, retinanet, iou_threshold = 0.25, max_detections=2000)

    print(mAP)