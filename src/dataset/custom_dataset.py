#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2



class CustomDataset(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target.reshape(-1)}


class CustomDatasetFill(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def resize_fill(self, _img):
        w,h,c = _img.shape
        if w>h:
            left = int((w-h)/2)
            right = int((w-h)/2)
            top = 0
            bottom = 0
        elif h>w:
            top = int((h-w)/2)
            bottom = int((h-w)/2)
            left = 0
            right = 0
        else:
            left = 0
            right = 0
            top = 0
            bottom = 0

        _img = cv2.copyMakeBorder(
            _img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REPLICATE,
            value=[0,0,0]
            )

        return cv2.resize(_img, self.target_size)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = self.resize_fill(img)
        #img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target.reshape(-1)}


class CustomDatasetSoftLabelFill(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        if 'test' in str(self.cfg.data_path):
            self.df['sorting_date'] = 1600.0
        self.soft_label = (self.df.sorting_date.values / 100.0) - 15.51
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def resize_fill(self, _img):
        w,h,c = _img.shape
        if w>h:
            left = int((w-h)/2)
            right = int((w-h)/2)
            top = 0
            bottom = 0
        elif h>w:
            top = int((h-w)/2)
            bottom = int((h-w)/2)
            left = 0
            right = 0
        else:
            left = 0
            right = 0
            top = 0
            bottom = 0

        _img = cv2.copyMakeBorder(
            _img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REPLICATE,
            value=[0,0,0]
            )

        return cv2.resize(_img, self.target_size)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        soft_label = self.soft_label[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = self.resize_fill(img)
        #img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target.reshape(-1),
                'soft_label': soft_label.reshape(-1)}


class CustomDatasetClassification(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}


class CustomDatasetSoftLabel(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        if 'test' in str(self.cfg.data_path):
            self.df['sorting_date'] = 1600.0
        self.soft_label = (self.df.sorting_date.values / 100.0) - 15.51
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        soft_label = self.soft_label[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target.reshape(-1),
                'soft_label': soft_label.reshape(-1)}


class CustomDatasetClassificationFill(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.object_id.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def resize_fill(self, _img):
        w,h,c = _img.shape
        if w>h:
            left = int((w-h)/2)
            right = int((w-h)/2)
            top = 0
            bottom = 0
        elif h>w:
            top = int((h-w)/2)
            bottom = int((h-w)/2)
            left = 0
            right = 0
        else:
            left = 0
            right = 0
            top = 0
            bottom = 0

        _img = cv2.copyMakeBorder(
            _img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_REPLICATE,
            value=[0,0,0]
            )

        return cv2.resize(_img, self.target_size)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = self.target[idx]
        path = str(self.path / self.name[idx])+'.jpg'
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        #img = cv2.resize(img, self.target_size)
        img = self.resize_fill(img)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}