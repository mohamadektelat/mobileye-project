try:
    import torch
    from skimage import io
    import torch.nn as nn
    import numpy as np
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import torchvision.transforms as transforms
    import torchvision
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.utils.data import DataLoader, default_collate
    import torchvision.transforms.functional as functional
    import torch.nn.functional as f
    from TrafficLightDataset import TrafficLightDataset
    from ConvNet_Model import ConvNet
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    from numpy import dtype
    import pandas as pd
    from skimage.feature import peak_local_max
    import os
    import json
    import glob
    import argparse
    import cv2
    from scipy import signal as sg, ndimage
    from scipy.ndimage import convolve
    import scipy.ndimage as filters
    from scipy import misc
    from PIL import Image
except ImportError:
    print("Need to fix the installation")
    raise

# -----------------------------------------For The training loop--------------------------------------------------------
# check if Gpu is available if not use cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 100

# Prepare My DataSet
dataset = TrafficLightDataset(cropped_image_file="cropped_table.h5", root_dir='', transform=transforms.ToTensor())

# Prepare My Model
model = ConvNet().to(device)

# Prepare the training
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Prepare my Train and Test sets Images
train_set, test_set = torch.utils.data.random_split(dataset, [4500, 855])

# Create a Train and Test Loader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)

# ----------------------------------------------------------------------------------------------------------------------
