# blockEstimation
This repository contains the test codes implemented with PyTorch for the paper [Virtual Training for a Real Application: Accurate Object-Robot Relative Localization without Calibration](http://imagine.enpc.fr/~loingvi/unloc/).The running environment is Python3.6 with PyTorch 0.4 framework. 

## required packages
numpy  
torch  
torchvision  
pillow  
pandas  
matplotlib

## Data and Trained models
The synthetic dataset used for training as well as the three UnLoc datasets ('lab', 'field', 'adv') used for validation can be downloaded [here](https://zenodo.org/record/2563622#.XJ0CYB2nGbk).

CAD models of the ABB IRB120 robot are available on this [page](https://new.abb.com/products/robotics/industrial-robots/irb-120/irb-120-cad).

Clamp 3D models are available here: [[clamp.stl]](http://imagine.enpc.fr/~loingvi/unloc/clamp.stl)[[clamp.3ds]](http://imagine.enpc.fr/~loingvi/unloc/clamp.3ds)[[clamp.obj]](http://imagine.enpc.fr/~loingvi/unloc/clamp.obj)

Trained models on synthetic dataset and relevant log files are stored in coarse_estimation/fine_estimation/tool_estimation folders. 

## Train
OLD (no more necessary to convert):
Convert .t7 dataset config file to .txt for PyTorch reading(change hard coded path):   
th inspect_t7_coarse.lua   
th inspect_t7_fine.lua   
th inspect_t7_tool.lua  

To train model, just run:    
bash run_coarse.sh   
bash run_fine.sh   
bash run_tool.sh   
 
## Validation on the UnLoc datasets
To test model on real data (with view aggregation), change PATH_TO_DATASETS and DIRECTORY_PATH in the following commands and run:
python coarse_validation.py --dataset '/PATH_TO_DATASETS/UnLoc_Lab_Dataset/' --model '/DIRECTORY_PATH/coarse_estimation/model_best.path.tar'
python fine_validation.py --dataset '/PATH_TO_DATASETS/UnLoc_Lab_Dataset/' --model '/DIRECTORY_PATH/fine_estimation/checkpoint_192.pth.tar' 

## Acknowledgement
The original test code is implemented by Vianney Loing with Torch and this is a PyTorch version of it. Some parts of the code come from  [PyTorch official example code](https://github.com/pytorch/examples/tree/master/imagenet).
