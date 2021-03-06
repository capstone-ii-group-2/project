import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models

training_path = 'training_datasets/datasets/asl_alphabet_train'
testing_path = 'training_datasets/datasets/asl_alphabet_test'

test_size = 0.2
batch_size = 32
num_epoch = 10
learning_rate = 0.001
num_classes = 29

# define transformations for datasets
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor()
])

# load datasets
train_dataset = datasets.ImageFolder(training_path, transform=train_transforms)
num_train = len(train_dataset)
print(train_dataset)

# define train dataset loader
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
classes = train_dataloader.dataset.classes
print(classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
