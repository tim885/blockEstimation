#!/bin/bash
# train
# python fine_estimation.py --csv_path 'fine_set/' --dataset_name 'data' --epochs 201 --batch-size 128 --lr 0.001 --gpu 1
# python fine_estimation.py --csv_path 'fine_set/' --dataset_name 'data' --resume 'fine_estimation/checkpoint.pth.tar' --epochs 201 --batch-size 128 --lr 0.001 --gpu 1

# Validate
python fine_estimation.py --csv_path 'fine_set/' --dataset_name 'data' --resume 'fine_estimation/model_best.path.tar' --evaluate --gpu 1
