#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    #for target in targets:
    #    shuffled_targets.append(targets[indices])

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    results = {
        'targets': targets,
        'shuffled_targets': shuffled_targets,
        'lam': lam
    }
    return data, results

def mixup(data, targets, alpha):
    """
    original mixup.
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)

    results = {
        'targets': targets,
        'shuffled_targets': shuffled_targets,
        'lam': lam
    }
    return data, results


def mixup2(data, targets, alpha):
    """
    mixup: use different lambda each samples.
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lams = [np.random.beta(alpha, alpha) for i in range(len(data))]
    #print(lams)

    for i in range(len(data)):
        data[i] = data[i] * lams[i] + shuffled_data[i] * (1 - lams[i])

    results = {
        'targets': targets,
        'shuffled_targets': shuffled_targets,
        'lam': lams
    }
    return data, results

def mixup_3sample(data, targets, alpha):
    """
    mixup: mixup 3 sample.
    """
    indices1 = torch.randperm(data.size(0))
    indices2 = torch.randperm(data.size(0))
    
    shuffled_data1 = data[indices1]
    shuffled_targets1 = targets[indices1]
    
    shuffled_data2 = data[indices2]
    shuffled_targets2 = targets[indices2]

    lam0, lam1, lam2 = np.random.dirichlet([alpha, alpha, alpha])
    data = data * lam0 + shuffled_data1 * lam1 + shuffled_data2 * lam2

    results = {
        'targets': targets,
        'shuffled_targets1': shuffled_targets1,
        'shuffled_targets2': shuffled_targets2,
        'lam': [lam0, lam1, lam2]
    }
    return data, results


def cutmix_soft_label(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = []
    for target in targets:
        shuffled_targets.append(target[indices])

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    results = {
        'targets': targets,
        'shuffled_targets': shuffled_targets,
        'lam': lam
    }
    return data, results

def mixup_soft_label(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = []
    for target in targets:
        shuffled_targets.append(target[indices])

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)

    results = {
        'targets': targets,
        'shuffled_targets': shuffled_targets,
        'lam': lam
    }
    return data, results