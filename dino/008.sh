python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --num_workers 2 --arch resnet18d --epochs 300 --data_path ../input/photos_ssl --output_dir ./output/008

cd ../src
bash train_1.sh
