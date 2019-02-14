#!/bin/bash
# train
# python tool_estimation.py --csv_path 'tool_set/' --dataset_name 'data' --epochs 40 --batch-size 128 --lr 0.001 --gpu 1
# python tool_estimation.py --csv_path 'tool_set/' --dataset_name 'data' --resume 'tool_estimation/checkpoint.pth.tar' --epochs 40 --batch-size 128 --lr 0.001 --gpu 1

# validation
python tool_estimation.py --csv_path 'tool_set/' --dataset_name 'data' --resume 'tool_estimation/model_best.path.tar' --evaluate --gpu 1



