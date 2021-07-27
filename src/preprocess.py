import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from  sklearn.model_selection  import StratifiedGroupKFold


input_dir = Path('../input')
img_dir = input_dir / 'photos'
img_dir_dino = input_dir / 'photos_ssl' / 'train_test'


#---------------------------------------#
# make input dir. for DINO pretraineing.
os.makedirs(img_dir_dino, exist_ok=True)


#---------------------------------------#
# copy photos/*.jpg files to phtos_ssl/train_test/. 
img_list = os.listdir(img_dir)
for from_name in img_list:
    to_name = from_name
    shutil.copy(img_dir / from_name, img_dir_dino / to_name)


#---------------------------------------#
# make fold data.
train = pd.read_csv(input_dir / 'train.csv')
test = pd.read_csv(input_dir / 'test.csv')


skf = StratifiedGroupKFold(n_splits=5, random_state=1111, shuffle=True)
splits = skf.split(np.arange(len(train)), y=train.sorting_date.values, groups=train.art_series_id.values)
train["fold"] = -1

for fold, (train_set, val_set) in enumerate(splits):
    train.loc[train.index[val_set], "fold"] = fold

for i in range(5):
    df = train[train.fold==i]


test['fold'] = 0
test['target'] = 0
test['sorting_date'] = 0


train.to_csv(input_dir / 'train_with_fold.csv', index=False)
test.to_csv(input_dir / 'test_with_fold.csv', index=False)


