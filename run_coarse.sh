#!/bin/bash
# train
python coarse_estimation.py --dataset_path 'PATH_TO_DATASET/rough_estimation_data/' --epochs 150 --batch-size 128 --lr 0.01 --gpu 0

# test on synthetic data
#python coarse_estimation.py --csv_path 'PATH_TO_DATASET/rough_estimation_data/' --dataset_name 'data' --resume 'coarse_estimation/model_best.path.tar' --evaluate --gpu 0




