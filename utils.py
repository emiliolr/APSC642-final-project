import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.utils import make_grid

def mse_loss(x, x_recon, sigmoid = False, train = True):
    batch_size = x.size(0)
    assert batch_size != 0

    if sigmoid:
        x_recon = torch.sigmoid(x_recon)

    if train:
        recon_loss = F.mse_loss(x_recon, x, reduction = 'sum').div(batch_size)
    else:
        recon_loss = F.mse_loss(x_recon, x, reduction = 'none')
        recon_loss = torch.sum(recon_loss, dim = (1, 2, 3))

    return recon_loss

def get_inlier_outlier_dataset(inlier_class = 0, num_inlier_samples = 6000, num_outlier_samples = 700):
    transform_list = Compose([ToTensor()])

    dataset_train = CIFAR10(root = 'data/cifar10',
                            download = True,
                            train = True,
                            transform = transform_list)
    dataset_test = CIFAR10(root = 'data/cifar10',
                           download = True,
                           train = False,
                           transform = transform_list)
    full_dataset = ConcatDataset([dataset_train, dataset_test])

    #  extract all labels in order
    class_idx = np.array([int(full_dataset[i][1]) for i in range(len(full_dataset))])

    #  subsetting the dataset for inlier class + sanity check
    inlier_idx = np.nonzero(class_idx == inlier_class)[0] # get indices where statement is true
    inlier_random_sample = np.random.choice(inlier_idx, size = num_inlier_samples, replace = False)
    inlier_dataset = Subset(full_dataset, inlier_random_sample)
    print(f'{len(inlier_dataset)} inlier samples')

    #  randomly sampling outlier class samples
    outlier_idx = np.nonzero(class_idx != inlier_class)[0]
    outlier_random_sample = np.random.choice(outlier_idx, size = num_outlier_samples, replace = False)
    outlier_dataset = Subset(full_dataset, outlier_random_sample)
    print(f'{len(outlier_dataset)} outlier samples')

    #  combining inlier and outlier subsets into a single dataset
    final_dataset = ConcatDataset([inlier_dataset, outlier_dataset])

    return final_dataset

def save_images(images_as_tensor, filename):
    grid = make_grid(images_as_tensor, nrow = 10)

    plt.imshow(grid.permute(1, 2, 0))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filename)
