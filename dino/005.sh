python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --epochs 300 --data_path ../input/photos_ssl --output_dir ./output/005