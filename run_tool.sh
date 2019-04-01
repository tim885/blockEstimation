#!/bin/bash
# train
python tool_estimation.py --dataset_path '/home/vianney/DATA/UnrealData/tool_detection_data/' --dataset_name 'data' --epochs 40 --batch-size 128 --lr 0.001 #--gpu 1 
#python tool_estimation.py --dataset_path 'PATH_TO_DATASET/tool_detection_data/' --dataset_name 'data' --epochs 40 --batch-size 128 --lr 0.001 #--gpu 1 
#python tool_estimation.py --dataset_path 'PATH_TO_DATASET/tool_detection_data/' --dataset_name 'data' --resume 'tool_estimation/checkpoint.pth.tar' --epochs 40 --batch-size 128 --lr 0.001 #--gpu 1

# test on synthetic dataset
#python tool_estimation.py --csv_path 'PATH_TO_DATASET/tool_detection_data/' --dataset_name 'data' --resume 'tool_estimation/model_best.path.tar' --evaluate  "--gpu 1



