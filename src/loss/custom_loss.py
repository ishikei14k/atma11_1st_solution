#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score


class CustomLoss:
    def __init__(self):
        self.loss = nn.MSELoss()

    def calc_loss(self, preds, targets):
        return self.loss(preds, targets)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']

        loss = lam*self.loss(preds, targets) + (1-lam)*self.loss(preds, shuffled_targets)
        return loss

    def calc_metrics(self, preds, targets):
        loss = np.sqrt(mean_squared_error(targets, preds))
        return loss


class CustomLossClassification:
    def __init__(self):
        self.loss_reg = nn.MSELoss()
        self.loss_cls = nn.CrossEntropyLoss()

    def calc_loss(self, preds, targets):
        return self.loss_cls(preds, targets)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']

        loss = lam*self.loss_cls(preds, targets) + (1-lam)*self.loss_cls(preds, shuffled_targets)
        return loss

    def calc_metrics(self, preds, targets):
        loss = np.sqrt(mean_squared_error(targets, preds))
        return loss

    def calc_metrics_classification(self, preds, targets):
        loss = accuracy_score(targets, preds)
        return loss


class CustomLossClassification2:
    def __init__(self):
        self.loss_reg = nn.MSELoss()
        self.loss_cls = nn.CrossEntropyLoss()

    def calc_loss(self, preds, targets):
        return self.loss_cls(preds, targets)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']

        loss = lam*self.loss_cls(preds, targets) + (1-lam)*self.loss_cls(preds, shuffled_targets)
        return loss

    def calc_metrics(self, preds, targets):
        loss = np.sqrt(mean_squared_error(targets, preds))
        return loss

    def calc_metrics_classification(self, preds, targets):
        loss = np.sqrt(mean_squared_error(targets, preds))
        return loss


class CustomLossSoftLabel:
    def __init__(self):
        self.loss = nn.MSELoss()
        self.weights = [1,1]

    def calc_loss(self, preds, targets):
        loss = 0.0
        for i, _ in enumerate(preds):
            current_loss = self.loss(preds[i], targets[i]) * self.weights[i]
            loss += current_loss
        return loss/sum(self.weights)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']
        loss = 0.0

        for i, _ in enumerate(preds):
            current_loss = (lam*self.loss(preds[i], targets[i]) + (1-lam)*self.loss(preds[i], shuffled_targets[i])) * self.weights[i]
            loss += current_loss
        return loss/sum(self.weights)

    def calc_metrics(self, preds, targets):
        loss = np.sqrt(mean_squared_error(targets, preds))
        return loss
