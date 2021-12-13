import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv2d # CoordConv2d layer

######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains
#    affine_grid and grid_sample modules.
#


class Net(nn.Module):
    def __init__(self, conv_layer):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        """self.hidden1 = nn.Conv2d(1, 32, kernel_size=3)
        self.batchNorm2D1 = nn.BatchNorm2d(32)
        self.hidden2 = nn.Conv2d(32, 32, kernel_size=3)
        self.hidden3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=(2,2))
        self.dropout2D = nn.Dropout2d(p=0.4)
        self.hidden4 = nn.Conv2d(32, 64, kernel_size=3)
        self.hidden4 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchNorm2D2 = nn.BatchNorm2d(64)
        self.hidden5 = nn.Conv2d(64, 64, kernel_size=3)
        self.hidden6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=(2,2))
        self.hidden7 = nn.Conv2d(64, 128, kernel_size=4)
        self.batchNorm2D3 = nn.BatchNorm2d(128)
        self.droput1D = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(128, 10)"""



        print("Using {} layer...".format(conv_layer))
        # Spatial transformer localization-network
        if conv_layer == "conv2d":
            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        elif conv_layer == "coordconv2d":
            self.localization = nn.Sequential(
                CoordConv2d(1, 8, kernel_size=7, with_r=True),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                CoordConv2d(8, 10, kernel_size=5, with_r=True),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode="border")

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.batchnorm1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.batchnorm2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

        # Modified forward pass (See: https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist/notebook)
        """x = self.batchNorm2D1(F.relu(self.hidden1(x)))
        x = self.batchNorm2D1(F.relu(self.hidden2(x)))
        x = self.dropout2D(self.batchNorm2D1(F.relu(self.hidden3(x))))

        x = self.batchNorm2D2(F.relu(self.hidden4(x)))
        x = self.batchNorm2D2(F.relu(self.hidden5(x)))
        x = self.dropout2D(self.batchNorm2D2(F.relu(self.hidden6(x))))

        x = self.batchNorm2D3(F.relu(self.hidden7(x)))
        x = self.droput1D(x.view(-1, 128))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)"""