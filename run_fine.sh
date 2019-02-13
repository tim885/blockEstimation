#!/bin/bash
# Train
# CUDA_VISIBLE_DEVICES=1 python fine_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_PV3.1/' --dataset_name '2018_01_15-13_59-data-2-2-2' --epochs 201 --batch-size 128 --lr 0.01 --gpu 1

# CUDA_VISIBLE_DEVICES=1 python fine_estimation.py --csv_path '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_PV3.1/' --dataset_name '2018_01_15-13_59-data-2-2-2' --resume 'fine_estimation/checkpoint.pth.tar' --epochs 201 --batch-size 128 --lr 0.005 --gpu 1

# python fine_estimation.py --csv_path 'fine_set/' --dataset_name '2018_01_15-13_59-data-2-2-2' --resume 'fine_estimation/checkpoint.pth.tar' --epochs 201 --batch-size 128 --lr 0.001 --gpu 1

# train
python fine_estimation.py --csv_path 'fine_set/' --dataset_name '2018_01_15-13_59-data-2-2-2' --epochs 201 --batch-size 128 --lr 0.001 --gpu 1

# Validate
#python fine_estimation.py --csv_path 'fine_set/' --dataset_name '2018_01_15-13_59-data-2-2-2' --resume 'fine_estimation/model_best.path.tar' --evaluate --gpu 3
