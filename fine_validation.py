# test fine estimation on real data with model trained on synthetic data

import argparse  # module for user-friendly command-line interfaces
import pandas as pd
import math
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# command-line interface arguments
parser = argparse.ArgumentParser(
   description='Pytorch transfer learning for block pose fine estmation')
parser.add_argument('--dataset',
                    metavar='DIR',
                    default='/PATH_TO_DATASETS/UnLoc_Lab_Dataset/',
                    help='path to dataset : ../UnLoc_Lab_Dataset/ or ../UnLoc_Field_Dataset/ or ../UnLoc_Adv_Dataset/')  # dataset dir argument
parser.add_argument('--model',
                    default='/DIRECTORY_PATH/fine_estimation/model_best.path.tar',
                    type=str, metavar='PATH', help='path to model')


def main():
    '''
    Here we fill the dict realPictData. Each realPict object corresponds to one block
    pose. realPict object is filled with the path of each picture from differents
    cameras, and with the value (X, Y, Rot) of the block w.r.t. the robot base.
    '''

    global args
    args = parser.parse_args()

    data_file_path = args.dataset + 'pictures_data.txt'
    realPictData = {}
    csv = pd.read_csv(data_file_path)

    for index, row in csv.iterrows():
        pictPath = row['#pict_path']
        X_abs = row[' X_abs']
        Y_abs = row[' Y_abs']
        Rot_abs = row[' Rot_abs']
        X_tool_abs = row[' X_tool_abs']
        Y_tool_abs = row[' Y_tool_abs']
        Rot_tool_abs = row[' Rot_tool_abs']
        camera_ID = row[' camera_ID']
        pict_number = row[' pict_number']
        poseID = row[' position_ID']

        # the pictures for this fine estimation step are the cropped one
        # obtained with tool_detection
        local_folder = pictPath.split('/')[0]
        pict_name = pictPath.split('/')[1]
        new_local_folder = local_folder + '_fine'
        new_pict_name = pict_name.split('.')[0] + '_fine.' + pict_name.split('.')[1]
        pictPath_P = new_local_folder + '/' + new_pict_name

        realPict = realPictData[poseID] if (poseID in realPictData) else RealPict(poseID)
        if not (poseID in realPictData):
            realPictData[poseID] = realPict
        realPict.setPath(pict_number, camera_ID, pictPath)
        realPict.setL_abs(X_abs, Y_abs, Rot_abs)

        if (X_tool_abs != '-'):
            realPict.setPath_P(pict_number, camera_ID, pictPath_P)
            realPict.set_tool_abs(pict_number, camera_ID, X_tool_abs, Y_tool_abs, Rot_tool_abs)
            realPict.setP_abs(pict_number, camera_ID)

    #labels coarse
    labels_X = symTable(30, 2)
    labels_Y = symTable(30, 2)
    labels_theta = signTable(90, 2)

    #load model
    model_path = args.model
    model = models.__dict__["resnet18"]()
    model = nn.Sequential(*list(model.children())[:-1])
    concat_fc = ConcatTable(*[60,60,90])
    model.add_module('concat_fc', concat_fc)
    model = model.cuda(0)
    model_best = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(model_best['state_dict'])
    model.eval()

    #image transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_val = transforms.Compose([transforms.CenterCrop((378,378)),transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    error_x = []
    error_y = []
    error_theta = []

    for k, v in realPictData.items():
        tot_output_X = None
        tot_output_Y = None
        tot_output_Rot = None
        clampPos1 = False
        clampPos2 = False
        for _,pictNumber in {0:3, 1:5}.items(): #for 2 clamp positions  (pict_number = 3 and pict_number = 5 in pictures_data.txt)
            images = None

            for _,cameraId in {0:0, 1:1, 2:2}.items(): #which cameras to use
                img_path = args.dataset + realPictData[k].getPath_P(pictNumber,cameraId)
                image = Image.open(img_path)
                image = transform_val(image)
                image = image.unsqueeze(0)
                image = image.cuda(0, non_blocking=True)
                images = torch.cat((images, image), 0) if (images is not None) else image

            output_x, output_y, output_theta = model(images)

            if (pictNumber == 3):
                tot_output_X = output_x.sum(0).type(torch.FloatTensor)
                tot_output_Y = output_y.sum(0).type(torch.FloatTensor)
                tot_output_Rot = output_theta.sum(0).type(torch.FloatTensor)
                clampPos1 = True
            elif (pictNumber == 5 and clampPos1):
                tot_output_X = tot_output_X + changeScoreRef(output_y.sum(0).type(torch.FloatTensor), 'X', 90) # X=-Y
                tot_output_Y = tot_output_Y + changeScoreRef(output_x.sum(0).type(torch.FloatTensor), 'Y', 90) # Y=X
                tot_output_Rot = tot_output_Rot + changeScoreRef(output_theta.sum(0).type(torch.FloatTensor), 'Rot', 90)
                clampPos2 = True

            if (clampPos1 and clampPos2):
                with torch.no_grad():
                    _, pred_x = torch.max(tot_output_X, 0)
                    _, pred_y = torch.max(tot_output_Y, 0)
                    _, pred_theta = torch.max(tot_output_Rot, 0)


                    pred_x = labels_X[pred_x]
                    pred_y = labels_Y[pred_y]
                    pred_theta = labels_theta[pred_theta]

                    target_x, target_y, target_theta = realPictData[k].getP_abs(3,0)#in the reference system of clamPos1 (i.e. pict_number = 3)
                    error_x.append(math.fabs(pred_x - target_x))
                    error_y.append(math.fabs(pred_y - target_y))
                    error_theta.append(min((pred_theta - target_theta) % 180, (target_theta - pred_theta) % 180))

                    print("{:.3f}".format(math.fabs(pred_x - target_x)), "{:.3f}".format(math.fabs(pred_y - target_y)))

    error_x = torch.FloatTensor(error_x)
    error_y = torch.FloatTensor(error_y)
    error_theta = torch.FloatTensor(error_theta)

    x_ok = torch.le(error_x, 5).type(torch.FloatTensor)
    y_ok = torch.le(error_y, 5).type(torch.FloatTensor)
    theta_ok = torch.le(error_theta, 2).type(torch.FloatTensor)
    x_y_ok = x_ok * y_ok
    x_y_theta_ok = x_y_ok * theta_ok

    print( '% (error_x < 5 mm) = ' + "{:.3f}".format(torch.mean(x_ok,0)*100))
    print( '% (error_y < 5 mm) = ' + "{:.3f}".format(torch.mean(y_ok,0)*100))
    print( '% (error_theta < 2°) = ' + "{:.3f}".format(torch.mean(theta_ok,0)*100))
    print( '% (error_x_y < 5mm) = ' + "{:.3f}".format(torch.mean(x_y_ok,0)*100))
    print( '% (error_x_y_theta < 5mm,2°) = ' + "{:.3f}".format(torch.mean(x_y_theta_ok,0)*100))
    print('error X (mm) = ' + "{:.3f}".format(torch.mean(error_x,0)) + ' +- ' + "{:.3f}".format(torch.std(error_x,0)))
    print('error Y (mm) = ' + "{:.3f}".format(torch.mean(error_y,0)) + ' +- ' + "{:.3f}".format(torch.std(error_y,0)))
    print('error Theta (°) = ' + "{:.3f}".format(torch.mean(error_theta,0)) + ' +- ' + "{:.3f}".format(torch.std(error_theta,0)))


class ConcatTable(nn.Module):
    """Define ConcatTable module for parallel FC layers"""
    def __init__(self, out_x, out_y, out_theta):
        super(ConcatTable, self).__init__()
        self.FC_x = nn.Linear(512, out_x)
        self.FC_y = nn.Linear(512, out_y)
        self.FC_theta = nn.Linear(512, out_theta)

    def forward(self, x):
        x = x.view(-1, 512*1*1)
        out = [self.FC_x(x), self.FC_y(x), self.FC_theta(x)]
        return out


class RealPict:
    def __init__(self, poseID):
        self.poseID = poseID
        self.L_abs = {}
        self.L_classes = {}
        self.L_modelPath = ''
        self.line = ''
        self.tool_abs = {}
        self.tool_abs['pict_number'] = {0:{'camera_ID': {0:{},1:{},2:{}}},1:{'camera_ID': {0:{},1:{},2:{}}},2:{'camera_ID': {0:{},1:{},2:{}}},3:{'camera_ID': {0:{},1:{},2:{}}},4:{'camera_ID': {0:{},1:{},2:{}}},5:{'camera_ID': {0:{},1:{},2:{}}},6:{'camera_ID': {0:{},1:{},2:{}}}}
        self.P_abs = {}
        self.P_abs['pict_number'] = {0:{'camera_ID': {0:{},1:{},2:{}}},1:{'camera_ID': {0:{},1:{},2:{}}},2:{'camera_ID': {0:{},1:{},2:{}}},3:{'camera_ID': {0:{},1:{},2:{}}},4:{'camera_ID': {0:{},1:{},2:{}}},5:{'camera_ID': {0:{},1:{},2:{}}},6:{'camera_ID': {0:{},1:{},2:{}}}}
        self.path = {}
        self.path['pict_number'] = {0:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},1:{'camera_ID': {0:"pathCam0",1:"pathCam1", 2:"pathCam2"}},2:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},3:{'camera_ID': {0:"pathCam0",1:"pathCam1", 2:"pathCam2"}},4:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},5:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},6:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}}}
        self.path_P = {}
        self.path_P['pict_number'] = {0:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},1:{'camera_ID': {0:"pathCam0",1:"pathCam1", 2:"pathCam2"}},2:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},3:{'camera_ID': {0:"pathCam0",1:"pathCam1", 2:"pathCam2"}},4:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},5:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}},6:{'camera_ID': {0:"pathCam0",1:"pathCam1",2:"pathCam2"}}}
        self.results_L ={}
        self.results_L['pict_number'] = {0:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},1:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},2:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},3:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},4:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},5:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}},6:{'camera_ID': {0:"resL0",1:"resL1", 2:"resL2"}}}
        self.pose_completed = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}

    """pictNumber between 0 and 6, cameraID between 0 and 2"""
    def setPath(self,pictNumber, cameraID, path):
        self.path['pict_number'][pictNumber]['camera_ID'][cameraID] = path

    """pictNumber between 0 and 6, cameraID between 0 and 2 """
    def getPath(self,pictNumber, cameraID):
        return self.path['pict_number'][pictNumber]['camera_ID'][cameraID]

    """pictNumber between 0 and 6, cameraID between 0 and 2 """
    def setPath_P(self,pictNumber, cameraID, path):
        self.path_P['pict_number'][pictNumber]['camera_ID'][cameraID] = path

    """pictNumber between 0 and 6, cameraID between 0 and 2 """
    def getPath_P(self,pictNumber, cameraID):
        return self.path_P['pict_number'][pictNumber]['camera_ID'][cameraID]

    """ """
    def setResults_L(self,pictNumber, cameraID, results):
        self.results_L['pict_number'][pictNumber]['camera_ID'][cameraID] = results

    """  """
    def getResults_L(self,pictNumber, cameraID, results):
        return self.results_L['pict_number'][pictNumber]['camera_ID'][cameraID]

    def setL_abs(self,X, Y, Rot):
        self.L_abs['X'] = X
        self.L_abs['Y'] = Y
        self.L_abs['Rot'] = Rot

    def set_tool_abs(self,pictNumber, cameraID, X_tool, Y_tool, Rot_tool):
        self.tool_abs['pict_number'][pictNumber]['camera_ID']['X'] = X_tool
        self.tool_abs['pict_number'][pictNumber]['camera_ID']['Y'] = Y_tool
        self.tool_abs['pict_number'][pictNumber]['camera_ID']['Rot'] = Rot_tool

    #X_tool, Y_tool and Rot_tool w.r.t. the robot base
    def get_tool_abs(self, pictNumber, cameraID):
        x = self.tool_abs['pict_number'][pictNumber]['camera_ID']['X']
        y = self.tool_abs['pict_number'][pictNumber]['camera_ID']['Y']
        rot = self.tool_abs['pict_number'][pictNumber]['camera_ID']['Rot']
        return x, y, rot

    def getP_abs(self, pictNumber, cameraID):
        x = self.P_abs['pict_number'][pictNumber]['camera_ID']['X']
        y = self.P_abs['pict_number'][pictNumber]['camera_ID']['Y']
        rot = self.P_abs['pict_number'][pictNumber]['camera_ID']['Rot']
        return x, y, rot

    '''first, setL_abs and set_tool_abs'''
    def setP_abs(self, pictNumber, cameraID):
        X_tool = float(self.tool_abs['pict_number'][pictNumber]['camera_ID']['X']) #X_tool w.r.t. the robot base
        Y_tool = float(self.tool_abs['pict_number'][pictNumber]['camera_ID']['Y']) #Y_tool w.r.t. the robot base
        Rot_tool = float(self.tool_abs['pict_number'][pictNumber]['camera_ID']['Rot']) #Rot_tool w.r.t. the robot base
        Rot_tool_rad = math.radians(Rot_tool) #Rot_tool w.r.t. the robot base in radians
        X_block = float(self.L_abs['X']) #X_block w.r.t. the robot base
        Y_block = float(self.L_abs['Y']) #Y_block w.r.t. the robot base
        Rot_block = float(self.L_abs['Rot']) #block rotation w.r.t. the robot base in degree

        X_block_tool = (X_block -X_tool)*math.cos(Rot_tool_rad) + (Y_block -Y_tool)*math.sin(Rot_tool_rad) #X_block w.r.t. the tool
        Y_block_tool = -(X_block -X_tool)*math.sin(Rot_tool_rad) + (Y_block -Y_tool)*math.cos(Rot_tool_rad) #Y_block w.r.t. the tool
        Rot_block_tool = (Rot_block - Rot_tool) % 180

        self.P_abs['pict_number'][pictNumber]['camera_ID']['X'] = X_block_tool
        self.P_abs['pict_number'][pictNumber]['camera_ID']['Y'] = Y_block_tool
        self.P_abs['pict_number'][pictNumber]['camera_ID']['Rot'] = Rot_block_tool


def changeScoreRef(Tensor1D, XorYorRot, angle):
    '''This function permits to change the score in one clamp reference system to a score in the clamp reference system where the clamp is rotated of 90°. Only works if same number of Xclasses and Yclasses and same number of negative classes and positive classes. '''
    assert(Tensor1D.dim() == 1)
    newScore = Tensor1D.clone()
    size = Tensor1D.size(0)

    if (XorYorRot == 'X' and angle == 90):
        for i in range(0,size):
            newScore[i] = Tensor1D[size - 1 - i]
    elif (XorYorRot == 'Rot' and angle == 90): #for size being an even number !!
        for i in range(0,size):
            newScore[i] = Tensor1D[int((i - size/2)%size)]

    return newScore


def symTable(j, pas):
    t=[]
    for i in range(-j, j+1):
        if i<0:
            t.append((i+0.5)*pas)
        elif i>0:
            t.append((i-0.5)*pas)
    return t


def signTable(j, pas):
    t=[]
    for i in range(1,j+1):
        t.append((i-0.5)*pas)
    return t


if __name__ == '__main__':
    main()
