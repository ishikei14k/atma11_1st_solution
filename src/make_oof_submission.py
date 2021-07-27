import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

args = get_args()

version = args.version
folds = [0,1,2,3,4]

sub = pd.read_csv('../input/test_with_fold.csv')
pred = np.zeros(len(sub))

train = pd.read_csv('../input/train_with_fold.csv')
oof = pd.DataFrame()

for fold in folds:
    df = pd.read_csv(f'../output/{version}/{fold}/sub_{version}.csv')
    if df.target.isnull().sum():
        print('null detected!!')
        df.target = df.target.fillna(0)
    pred += df.target.values
    oof_name = [x for x in os.listdir(f'../output/{version}/{fold}/') if 'oof' in x]
    print(oof_name)
    _oof = pd.read_csv(f'../output/{version}/{fold}/{oof_name[0]}')
    oof = pd.concat([oof, _oof], axis=0)

assert len(oof)==len(train)
score = np.sqrt(mean_squared_error(oof.target.values, oof.pred.values))
print(score)

pred /= len(folds)

sub.target = pred
sub[['target']].to_csv(f'../output/sub_{version}.csv', index=False)
oof.to_csv(f'../output/oof_{version}_{score:.4f}.csv', index=False)
