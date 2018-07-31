# generate dataset configuration file for training
# created by QIU Xuchong
# 2018/07

import argparse  # module for user-friendly command-line interfaces

# command-line interface arguments
parser = argparse.ArgumentParser(description='Dataset generation arguments')
parser.add_argument('--dir', default='/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData'
                                     '/scenario_LV3.1/', type=str, help='directory of data')
parser.add_argument('--stepXY', default=5, type=int, help='step along X/Y')
parser.add_argument('--stepRot', default=5, type=int, help='step along theta')
parser.add_argument('--r_min', default=210, type=int, help='min radius')
parser.add_argument('--r_max', default=510, type=int, help='max radius')
parser.add_argument('--trainSize', default=15, type=int, help='num of samples for train per bin')
parser.add_argument('--valSize', default=2, type=int, help='num of samples for val per bin')


def main():
    global args
    args = parser.parse_args()

    # load data dir and generate train_dataset and val_dataset
    csvfile_root = args.dir + 'CSV_files/'

    # generate train_dataset and val_dataset
    train_dataset, val_dataset = gen_dataset(data_params)


if __name__ == '__main__':
    main()
