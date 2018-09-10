#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python coarse_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_L/' --dataset_name '2017_09_20-13_32-data-5-5-5' --epochs 130 --batch-size 128 --lr 0.01 --gpu 0 
