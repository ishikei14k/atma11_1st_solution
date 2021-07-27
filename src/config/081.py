#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import os


version = f'081'
seed = 1111

n_fold = 5
num_workers = 2
target_size = (224,224) # None=Original size.
use_amp = True
use_mixup_cutmix = True
num_gpu = 1
batch_size = 32*num_gpu
mixup_alpha = 0.4
cutmix_alpha = 0.4
mixup_freq = 0.8
normalize = False

with_soft_label = False

wo_mixup_epochs = 501
n_epochs = 100


project = 'atma11_1st_solution'
input_dir = Path(f'/home/ishikei/work/atma/{project}/input')
output_dir = Path(f'/home/ishikei/work/atma/{project}/output')


# dataset
dataset_name = 'CustomDatasetClassification'

# model config
# model
model = dict(
    name = 'AtmaCustomModelViTDINOClass',
    architecture = 'vit_base',
    pretrained_weight=None,
    params = dict(
        pretrained_path='../dino/output/007/checkpoint.pth',
    )
)

# optimizer
optim = dict(
    name = 'AdamW',
    lr = 0.0001*num_gpu,
    weight_decay = 0.01
)

# loss
loss = dict(
    name = 'CustomLossClassification2',
    params = dict(
    ),
)

# scheduler
scheduler = dict(
    name = 'CosineAnnealingLR',
    params = dict(
        T_max=n_epochs,
        eta_min=0,
        last_epoch=-1,
    )
)


# snapshot
snapshot = dict(
    save_best_only=True,
    mode='min',
    initial_metric=None,
    name=version,
    monitor='metric'
)

# logger
logger = dict(
    params=dict(
        logging_info=['loss', 'metric'],
        print=False
    ),
)

# augmentations.
horizontalflip = dict(
    name = 'HorizontalFlip',
    params = dict()
)

verticalflip = dict(
    name = 'VerticalFlip',
    params = dict()
)

shiftscalerotate = dict(
    name = 'ShiftScaleRotate',
    params = dict(
        shift_limit = 0.1,
        scale_limit = 0.1,
        rotate_limit = 15,
    ),
)

gaussnoise = dict(
    name = 'GaussNoise',
    params = dict(
        var_limit = 5./255.
        ),
)

blur = dict(
    name = 'Blur',
    params = dict(
        blur_limit = 3
    ),
)

randommorph = dict(
    name = 'RandomMorph',
    params = dict(
        size = target_size,
        num_channels = 1,
    ),
)

randombrightnesscontrast = dict(
    name = 'RandomBrightnessContrast',
    params = dict(),
)

griddistortion = dict(
    name = 'GridDistortion',
    params = dict(),
)

elastictransform = dict(
    name = 'ElasticTransform',
    params = dict(
        sigma = 50,
        alpha = 1,
        alpha_affine = 10
    ),
)

cutout = dict(
    name = 'Cutout',
    params = dict(
        num_holes=1,
        max_h_size=int(256*0.3),
        max_w_size=int(256*0.3),
        fill_value=0,
        p=0.7
    ),
)

totensor = dict(
    name = 'ToTensorV2',
    params = dict(),
)

oneof = dict(
    name='OneOf',
    params = dict(),
)

normalize = dict(
    name = 'Normalize',
    params = dict(
        max_pixel_value=1.0
    ),
)


# train.
train = dict(
    is_valid = False,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [
        horizontalflip,
        shiftscalerotate,
        blur,
        randombrightnesscontrast,
        normalize,
        totensor
        ],
)


# valid.
valid = dict(
    is_valid = True,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [normalize,totensor],
)

# test.
test = dict(
    is_valid = True,
    data_path = input_dir / 'test_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    weight_name = f'{version}_best.pt',
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [normalize,totensor],
)