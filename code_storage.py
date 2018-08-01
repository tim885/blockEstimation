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