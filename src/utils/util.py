#!/usr/bin/env python
# coding: utf-8


import csv
import logging
import time
from contextlib import contextmanager
import os
import numpy as np
import random
import torch
import argparse


import requests


def make_output_dir_if_needed(output_dir):
    os.makedirs(output_dir / 'weight', exist_ok=True)


def line_notify(message):
    line_token = 'xxxxxxxx'  # 終わったら無効化する
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time.time() - t0:.0f} s')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomLogger:
    def __init__(self, name, logging_info, print=True):
        self.name = name
        self.print = print
        self.logging_info = logging_info
        self.init_log_file()

    def init_log_file(self):
        self.header = ['epoch', 'wrap_time']
        self.header.extend(sum([[f'trn_{i}', f'val_{i}'] for i in self.logging_info], []))
        with open(self.name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def write_log(self, item):
        with open(self.name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(item)
        if self.print:
            print(item)


class Snapshot:
    def __init__(self, save_best_only=True, mode='min', initial_metric=None, output_dir='', name='', monitor='metric'):
        self.save_best_only = save_best_only
        if mode=='min':
            self.mode = 1
        elif mode=='max':
            self.mode = -1
        self.init_metric(initial_metric)
        self.output_dir = output_dir
        self.name = name
        self.monitor = monitor
            
    def init_metric(self, initial_metric):
        if initial_metric:
            self.best_metric = initial_metric
        else:
            self.best_metric = self.mode*1000

    def update_best_metric(self, metric):
        if self.mode*metric <= self.mode*self.best_metric:
            self.best_metric = metric
            return 1
        else:
            return 0

    def save_weight_optimizer(self, model, optimizer, prefix):
        torch.save(model.state_dict(), self.output_dir / str(self.name+f'_{prefix}.pt'))
        #torch.save(optimizer.state_dict(), self.output_dir / str(self.name+f'_optimizer_{prefix}.pt'))

    def snapshot(self, metric, model, optimizer, epoch):
        is_updated = self.update_best_metric(metric)
        if is_updated:
            self.save_weight_optimizer(model, optimizer, 'best')
            print('--> [best score was updated] save shapshot.')
        if not self.save_best_only:
            self.save_weight_optimizer(model, optimizer, f'epoch{epoch}')
            print(f'--> [epoch:{epoch}] save shapshot.')