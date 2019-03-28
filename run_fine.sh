#!/bin/bash
# train
python fine_estimation.py --dataset_path 'PATH_TO_DATASET/fine_estimation_data/' --dataset_name 'data' --epochs 191 --batch-size 128 --lr 0.01 #--gpu 1
python fine_estimation.py --dataset_path 'PATH_TO_DATASET/fine_estimation_data/' --dataset_name 'data' --resume 'fine_estimation/checkpoint.pth.tar' --epochs 193 --batch-size 128 --lr 0.005


