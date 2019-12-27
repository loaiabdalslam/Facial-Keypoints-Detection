import matplotlib.pylab as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net

# instantiate the model
net = Net()
print(net)


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

# define the data transform using transfroms.Compose([..])
# Note the order is matter
data_tranform = transforms.Compose([Rescale(250),
                                    RandomCrop(224),
                                    Normalize(),
                                    ToTensor()])

# Create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_tranform)

print('Number of images: ', len(transformed_dataset))


# Loadiing the data in batches
# for windows users, change the num_workers to 0 or you will face some issues with your DataLoader failing
batch_size = 16
train_loader = DataLoader(transformed_dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 4)


