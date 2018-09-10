#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python tool_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_toolDetectionV3.1/' --dataset_name '2018_02_22-20_17-data-0.02-0' --epochs 40 --batch-size 128 --lr 0.01 --gpu 1

CUDA_VISIBLE_DEVICES=1 python tool_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_toolDetectionV3.1/' --dataset_name '2018_02_22-20_17-data-0.02-0' --resume 'tool_estimation/checkpoint.pth.tar' --epochs 40 --batch-size 128 --lr 0.001 --gpu 1






