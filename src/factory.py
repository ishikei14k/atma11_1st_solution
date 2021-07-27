#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset.custom_dataset import *
from models.custom_model import *
from loss.custom_loss import *


def get_imgs(input_npy_dir):
    imgs = np.load(input_npy_dir / 'train_imgs_numpy_array_200840x128x128_just_resize_wo_maximize.npy')
    imgs = np.reshape(imgs, (-1, 128, 128)).astype(np.uint8)
    return imgs


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


def get_dataset(cfg, folds, transforms):
    return eval(cfg.dataset_name)(cfg, folds, transforms)


def get_dataset_loader(cfg, folds):
    transforms = get_transforms(cfg)
    dataset = get_dataset(cfg, folds, transforms)
    loader = DataLoader(dataset, **cfg.loader)
    return dataset, loader


def get_model(cfg):
    return eval(cfg.name)(cfg.architecture, **cfg.params)


def get_device(device_no=0):
    return torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")


def get_optimizer(cfg):
    if hasattr(optim, cfg.name):
        optimizer = getattr(optim, cfg.name)
    else:
        optimizer = eval(cfg.name)
    return optimizer


def get_loss(cfg):
    if hasattr(nn, cfg.name):
        loss = getattr(nn, cfg.name)(**cfg.params)
    else:
        loss = eval(cfg.name)(**cfg.params)
    return loss


def get_scheduler(cfg, optimizer):
    if hasattr(optim.lr_scheduler, cfg.name):
        scheduler = getattr(optim.lr_scheduler, cfg.name)(optimizer, **cfg.params)
    else:
        scheduler = eval(cfg.name)(optimizer, **cfg.params)
    return scheduler

