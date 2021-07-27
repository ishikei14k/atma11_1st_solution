# atmaCup 1st place solution
- atmaCup#11の最終submissionコードです。
- 解法自体は[こちらのdiscussion](https://www.guruguru.science/competitions/17/discussions/d4a737a1-2f39-4c1e-9ac9-1b52c1b419e2/)に記載しています。

# Derectory Layout
```
.
├── dino      : Code for DINO pretraining.
├── input     : Input files.
├── output    : Output files. (weight, history.csv, submission file...)
└── src       : Code for train models & make submission file.

```

# Rquirements
下記の環境で動作確認済みです。
- Python 3.8.10
- CUDA 11.1
- torch==1.8.0

# How to run
## Data download & Preprocess
1. https://www.guruguru.science/competitions/17/data-sources からコンペのデータをダウンロードし、`.input` に解凍してください。
2. 下記コマンドを実行してください。DINO pretrain用のデータセットとfoldデータが作成されます。
```
cd src
python preprocess.py
```

## DINO pretraining
1. 下記コマンドを実行してください。コンペティションデータでDINOの学習を行います。
```
cd dino

# DINO pretraining for vit_small.
sh 005.sh

# DINO pretraining for vit_base.
sh 007.sh

# DINO pretraining for resnet18d.
sh 008.sh
```

## Train models & Make submission file.
1. 下記コマンドを実行してください。各モデルの学習とsubmissionファイルの作成を行います。
```
cd src
# Train all models.
sh bin/train_all_models.sh

# If you want to train specific model.
sh bin/train_*.sh
```

