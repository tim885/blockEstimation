#!/bin/bash 

#CUDA_VISIBLE_DEVICES=0 python coarse_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_L/' --dataset_name '2017_09_20-13_32-data-5-5-5' --epochs 130 --batch-size 128 --lr 0.01 --gpu 0 

CUDA_VISIBLE_DEVICES=0 python coarse_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_LV3.1/' --dataset_name '2018_01_30-10_21-data-5-5-5' --epochs 140 --batch-size 128 --lr 0.01 --gpu 0 



