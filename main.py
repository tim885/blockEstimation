# transfer learning script for block pose(x,y,theta) coarse estimation
# originally implemented with Torch by Vianney Loing
# derived from pytorch/examples/imagenet
#
# created by QIU Xuchong
# 2018/07

import argparse  # module for user-friendly command-line interfaces
import os
from skimage import io
from PIL import Image
import random
import shutil  # high-level file operations
import pandas as pd  # easy csv parsing
import matplotlib.pyplot as plt  # for debug
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# return sorted list from the items in iterable
model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))

# command-line interface arguments
parser = argparse.ArgumentParser(description='Pytorch transfer learning for blockEstmation')
# parser.add_argument('data', metavar='DIR', help='path to dataset')  # dataset dir argument
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names, help='model_architecture: ' +
                    ' | '.join(model_names) +
                    ' (default:resnet18)')  # {--arch | -a} argument 'arch' is added
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')  # default is 1 for powerless machine
parser.add_argument('--epochs', default=140, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')  # for runtime surveillance
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # resume mode
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')  # dest; action
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training') # for multi-gpu training
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')  # seed for random init
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--numClass', default=[202, 202, 36], type=int,
                    help='number of class for x, y and theta')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:  # for reproducibility
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1  # whether validate distributed processes
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)  # load pretrained model in torchvision
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]() # load relevant architecture in torchvision

    # replace FC layers of resnet with paradllel FC layers
    model = nn.Sequential(*list(model.children())[:-1])  # remove FC layer and store in nn.Sequential container

    # add concatenated FC module for classification of x,y and theta
    concat_fc = ConcatTable(*args.numClass)
    model.add_module('concat_fc', concat_fc)

    # gpu settings, default is on one GPU
    '''
    if args.gpu is not None:
        model = model.cuda(args.gpu)  # convert model to relevant single GPU
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # module replicated in multi-device and each handles a portion of input
            model = torch.nn.DataParallel(model).cuda()
    '''

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # .cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)  # load model at checkpoint
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])  # copy params and buffers to model
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True  # optimize cudnn if input_sz is fix, if not, worse speed

    # load block estimation dataset

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # imagenet statistics

    transform = transforms.Compose([transforms.RandomResizedCrop(224),  # fix input size
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),  # convert image to tensor
                                    normalize])

    csv_dir = '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_LV3.1/'
    csv_train = csv_dir + '2018_01_30-10_21-data-5-5-5_train.txt'
    csv_val = csv_dir + '2018_01_30-10_21-data-5-5-5_val.txt'

    train_dataset = BlockDataset(csv_file=csv_train, transform=transform)
    val_dataset = BlockDataset(csv_file=csv_val, transform=transform)

    # define sampler for data fetching distributed training
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # define training data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # define validation dataset and loader together
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:  # evaluate on val set and end function
        validate(val_loader, model, criterion)
        return

    # runtime for training
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # update learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set after training of one epoch
        prec1 = validate(val_loader, model, criterion)

        # remember best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dic': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),  # save optimizer state
        }, is_best, 'checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()  # class instance
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()  # switch to training mode

    end = time.time()
    # training with mini-batch
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        target = target.view(-1, 3)
        target_x = target[:, 0]
        target_y = target[:, 1]
        target_theta = target[:, 2]

        # forward pass
        output_x, output_y, output_theta = model(input)

        # sum up losses for different classification task
        loss_x = criterion(output_x, target_x)
        loss_y = criterion(output_y, target_y)
        loss_theta = criterion(output_theta, target_theta)
        global_loss = loss_x + loss_y + loss_theta

        # measure average accuracy and record loss
        prec1_x, prec1_y, prec1_theta, prec5_x, prec5_y, prec5_theta = \
            accuracy([output_x, output_y, output_theta], [target_x, target_y, target_theta], topk=(1, 5))
        prec1 = (prec1_x + prec1_y + prec1_theta)/3
        prec5 = (prec5_x + prec5_y + prec5_theta)/3

        losses.update(global_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD for this mini-batch
        optimizer.zero_grad()
        global_loss.backward()
        optimizer.step()

        # measure ellapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # runtime display settings per mini-batch
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # runtime for only  forward pass
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):  # for each mini-batch
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1_x, prec1_y, prec1_theta, prec5_x, prec5_y, prec5_theta = \
                accuracy(output, target, topk=(1, 5))
            prec1 = (prec1_x + prec1_y + prec1_theta) / 3
            prec5 = (prec5_x + prec5_y + prec5_theta) / 3

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure ellapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:  # store model with best perf
        shutil.copyfile(filename, 'model_best.path.tar')


def adjust_learning_rate(optimizer, epoch):
    """set learning rate dynamic to epoch"""
    # lr = agrs.lr * (0.1**(epoch//30))  # decay by 10 per 30 epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr  # fixed lr


def accuracy(output, target, topk=(1,)):
    """Compute accuracy for x,y,theta seperately"""
    with torch.no_grad():  # no grad computation to reduce memory
        maxk = max(topk)  # topk = (1,5)
        batch_size = target.size(0)

        # split groundtruth to groundtruth for x, y and theta(to modify)
        target_x = target[:, 0]
        target_y = target[:, 1]
        target_theta = target[:, 2]

        # split output to outputs for x, y and theta(to modify)
        output_x = output[0]
        output_y = output[1]
        output_theta = output[2]

        _, pred_x = output_x.topk(maxk, 1, True, True)  # output value and indices
        _, pred_y = output_y.topk(maxk, 1, True, True)
        _, pred_theta = output_theta.topk(maxk, 1, True, True)

        pred_x = pred_x.t()  # transpose dim size
        pred_y = pred_y.t()
        pred_theta = pred_theta.t()

        correct_x = pred_x.eq(target_x.view(1, -1).expand_as(pred_x))
        correct_y = pred_y.eq(target_y.view(1, -1).expand_as(pred_y))
        correct_theta = pred_theta.eq(target_theta.view(1, -1).expand_as(pred_theta))

        res = []
        for k in topk:  # store in res and output
            correct_k_x = correct_x[:k].view(-1).float().sum(0, keepdim=True)
            correct_k_y = correct_y[:k].view(-1).float().sum(0, keepdim=True)
            correct_k_theta = correct_theta[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k_x.mul_(100 / batch_size))
            res.append(correct_k_y.mul_(100 / batch_size))
            res.append(correct_k_theta.mul_(100 / batch_size))

        return res


class BlockDataset(Dataset):
    """block pose estimation dataset"""
    def __init__(self, csv_file, transform=None):
        self.samples = pd.read_csv(csv_file)  # (input, label)
        self.transform = transform  # pre-processing transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples.iloc[idx, 0]
        # image = io.imread(img_path)
        image = Image.open(img_path)
        label = self.samples.iloc[idx, 1:].values
        label = label.astype('float').reshape(-1, 3)  # 1*3 narray
        label = torch.from_numpy(label)
        label = label.long() - 1  # lua index begins at 1

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'label': label}
        sample = (image, label)

        return sample


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # n is num of samples
        self.count += n
        self.avg = self.sum / self.count


class ConcatTable(nn.Module):
    """Define ConcatTable module in pytorch """
    def __init__(self, out_x, out_y, out_theta):
        super(ConcatTable, self).__init__()
        self.FC_x = nn.Linear(512, out_x)
        self.FC_y = nn.Linear(512, out_y)
        self.FC_theta = nn.Linear(512, out_theta)

    def forward(self, x):
        x = x.view(-1, 512*1*1)
        out = [self.FC_x(x), self.FC_y(x), self.FC_theta(x)]
        return out


if __name__ == '__main__':
    main()
