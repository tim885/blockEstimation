# storage of codes which are not used temporally



# functions for creating block pose dataset


data_opts = {
    'stepXY': 5,
    'stepRot': 5,
    'r_min': 210,
    'r_max': 510,
    'trainSize': 4,
    'valSize': 2,
}

# params for histTable
        histSize = 2*math.ceil(opts.r_max/opts.stepXY) # square 2D sample num hist size
        trainHist = torch.zeros(histSize, histSize)
        valHist = torch.zeros(histSize, histSize)

        # select samples for training based on class balance strategy
        if train:
            sample_idx = 0
            train_sample_idx = 0
            self.train_paths = {}
            self.train_X = {}
            self.train_Y = {}
            self.train_theta = {}

            while True:
                print(sample_idx)
                # get block pose and robot pose in world frame
                BlockX = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{BlockX}")]
                BlockY = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{BlockY}")]
                BlockTheta = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{BlockTheta}")]
                RobotX = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{RobotX}")]
                RobotY = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{RobotY}")]
                RobotTheta = self.samples_attr.iloc[sample_idx, self.samples_attr.columns.get_loc("{RobotTheta}")]

                # get block pose with regard to robot base frame
                X, Y, Theta = get_block_pose(BlockX, BlockY, BlockZ, RobotX, RobotY, RobotTheta)

                # convert pose to class indices
                Xc, Yc,Theta_c = get_pose_label(X, Y, Theta, opts)

                if Xc and Yc and Theta_c:
                    X_hist = ((Xc > 0) and 0 or 1) + Xc + histSize / 2
                    Y_hist = ((Yc > 0) and 0 or 1) + Yc + histSize / 2
                    if trainHist[X_hist, Y_hist] < opts.trainSize:
                        trainHist[X_hist, Y_hist] += 1
                        self.train_paths[train_sample_idx] = self.samples_attr.iloc[sample_idx, 0]

#######################################################################################################################

# create weighted data sampler in training
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# define weighted sampler
dataset_train = datasets.ImageFolder(traindir)

# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                           sampler=sampler, num_workers=args.workers, pin_memory=True)