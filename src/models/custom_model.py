#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

# model.
import timm

# custom modules.
from . import vision_transformer as vits
from .utils import load_pretrained_weights, load_pretrained_weights_resnet


class AtmaCustomModel(nn.Module):
    def __init__(self, architecture):
        super(AtmaCustomModel, self).__init__()

        self.model = timm.create_model(architecture, pretrained=False, in_chans=3)
        #print(self.model)

        if 'vit' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, 1)
        elif 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, 1)
        elif 'efficient' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 1)
        elif 'ensenet' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 1)
        elif 'nfnet' in architecture:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        
        return x


class AtmaCustomModelSoftLabel(nn.Module):
    def __init__(self, architecture):
        super(AtmaCustomModelSoftLabel, self).__init__()

        self.model = timm.create_model(architecture, pretrained=False, in_chans=3)
        #print(self.model)

        if 'vit' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        elif 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif 'efficient' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'ensenet' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'nfnet' in architecture:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()

        self.fc1 = nn.Linear(self.n_features, 1)
        self.fc2 = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        
        return x1, x2


class AtmaCustomModelViTDINO(nn.Module):
    def __init__(self, architecture, pretrained_path):
        super(AtmaCustomModelViTDINO, self).__init__()
        self.model = vits.__dict__[architecture](patch_size=16)
        load_pretrained_weights(self.model, pretrained_path, 'teacher', architecture, 16)

        self.n_features = self.model.embed_dim

        self.head = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        
        return x


class AtmaCustomModelResNetDINO(nn.Module):
    def __init__(self, architecture, pretrained_path):
        super(AtmaCustomModelResNetDINO, self).__init__()
        
        self.model = timm.create_model(architecture, pretrained=False, in_chans=3)
        load_pretrained_weights_resnet(self.model, pretrained_path, 'teacher', architecture, 16)

        if 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, 1)
        elif 'efficient' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        x = self.model(x)
        
        return x

class AtmaCustomModelResNetDINOClass(nn.Module):
    def __init__(self, architecture, pretrained_path):
        super(AtmaCustomModelResNetDINOClass, self).__init__()
        
        self.model = timm.create_model(architecture, pretrained=False, in_chans=3)
        load_pretrained_weights_resnet(self.model, pretrained_path, 'teacher', architecture, 16)

        if 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, 4)
        elif 'efficient' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 4)

    def forward(self, x):
        x = self.model(x)
        
        return x


class AtmaCustomModelViTDINOSoftLabel(nn.Module):
    def __init__(self, architecture, pretrained_path):
        super(AtmaCustomModelViTDINOSoftLabel, self).__init__()
        self.model = vits.__dict__[architecture](patch_size=16)
        load_pretrained_weights(self.model, pretrained_path, 'teacher', architecture, 16)

        self.n_features = self.model.embed_dim

        self.head1 = nn.Linear(self.n_features, 1)
        self.head2 = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        
        return x1, x2


class AtmaCustomModelViTDINOClass(nn.Module):
    def __init__(self, architecture, pretrained_path):
        super(AtmaCustomModelViTDINOClass, self).__init__()
        self.model = vits.__dict__[architecture](patch_size=16)
        load_pretrained_weights(self.model, pretrained_path, 'teacher', architecture, 16)

        self.n_features = self.model.embed_dim

        self.head = nn.Linear(self.n_features, 4)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        
        return x