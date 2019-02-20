# blockEstimation
This repository contains the train/test code implemented with PyTorch for the paper [Virtual Training for a Real Application: Accurate Object-Robot Relative Localization without Calibration](http://imagine.enpc.fr/~loingvi/unloc/).The running environment is Python3.6 with PyTorch 0.4 framework. 

## required packages
numpy  
torch  
torchvision  
pillow  
pandas  
matplotlib

## Data and Trained models
Real dataset called UnLoc is composed of three sub-datasets ('lab', 'field', 'adv'). Synthetic/real dataset can be downloaded [here](https://zenodo.org/record/2563622#.XGQSQlVKg-U). 

CAD models of the ABB IRB120 robot are available on this [page](https://new.abb.com/products/robotics/industrial-robots/irb-120/irb-120-cad).

Clamp 3D models are available here: [[clamp.stl]](http://imagine.enpc.fr/~loingvi/unloc/clamp.stl)[[clamp.3ds]](http://imagine.enpc.fr/~loingvi/unloc/clamp.3ds)[[clamp.obj]](http://imagine.enpc.fr/~loingvi/unloc/clamp.obj)

Trained models on synthetic dataset and relevant log files are stored in coarse_estimation/fine_estimation/tool_estimation folders. 

## Train
Convert .t7 dataset config file to .txt for PyTorch reading(change hard coded path):   
th inspect_t7_coarse.lua   
th inspect_t7_fine.lua   
th inspect_t7_tool.lua  

To train model, just run:    
bash run_coarse.sh   
bash run_fine.sh   
bash run_tool.sh   
 
## Test
(Test on real image dataset hasn't been done yet, create .lua script to convert .t7 dataset to .txt  for PyTorch reading and use evaluation mode with trained model to test performance) 

## Author
### Xuchong Qiu
### Vianney Loing 

## Acknowledgement
The original train/test code is implemented by Vianney Loing with Torch and this is a PyTorch version of it. Some parts of the code come from  [PyTorch official example code](https://github.com/pytorch/examples/tree/master/imagenet).
