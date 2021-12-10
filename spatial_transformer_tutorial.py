# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_
.. figure:: /_static/img/stn/FSeq.png
In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__
Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.
One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from net import Net
from train_test import train, test
from utils import download_mnist_dataset, visualize_stn, log_to_tensorboard, calculate_result_metrics

from tensorboardX import SummaryWriter # to visualize training data
from sys import argv
import argparse

def parse_args(in_argv=argv):
    """Parse incoming arguments.
    :param in_argv: incoming argv list or tuple
    :returns: Namespace with arguments
    """
    description = "MNIST - Spatial Transformer Network training pipeline"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ep", "--epochs",
                        dest="epochs",
                        default=30,
                        type=int,
                        help="Number of epochs of the training loop")
    parser.add_argument("-cv", "--conv-layer",
                        dest="conv_layer",
                        default="coordconv2d",
                        choices=("coordconv2d", "conv2d"),
                        help="Type of convolutional layer used inside the localization network")

    args = parser.parse_args(in_argv[1:])
    return args


def main(in_argv):
    args = parse_args(in_argv)

    plt.ion()  # interactive mode

    # Download MNIST dataset (first run only)
    download_mnist_dataset()

    # Declare hardware device where to perform the training/testing computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = Net(args.conv_layer).to(device)

    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)

    # SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Instantiate tensorboardX writer
    writer = SummaryWriter()

    best_acc = 0; best_acc_preds = []; best_acc_targets = []
    # Train loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, acc, acc_preds, acc_targets = test(model, device, test_loader)
        scheduler.step()

        # Live-log training metrics to TensorboardX
        log_to_tensorboard(writer, train_loss, test_loss, acc, epoch)

        # Store preds and targets if the test accuracy is better
        if acc >= best_acc:
            best_acc = acc; best_acc_preds = acc_preds; best_acc_targets = acc_targets

    calculate_result_metrics(best_acc, best_acc_targets, best_acc_preds)

    writer.close()

    # Visualize the STN transformation on some input batch
    visualize_stn(model, device, test_loader)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(argv)
