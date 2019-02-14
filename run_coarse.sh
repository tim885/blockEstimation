#!/bin/bash
# train
# python coarse_estimation.py --csv_path 'coarse_set/' --dataset_name 'data' --epochs 150 --batch-size 128 --lr 0.01 --gpu 0
# python coarse_estimation.py --csv_path 'coarse_set/' --dataset_name 'data' --resume 'coarse_estimation/checkpoint.pth.tar' --epochs 300 --batch-size 128 --lr 0.01 --gpu 0

# test
python coarse_estimation.py --csv_path 'coarse_set/' --dataset_name 'data' --resume 'coarse_estimation/model_best.path.tar' --evaluate --gpu 0

