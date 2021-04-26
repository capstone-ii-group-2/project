import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#from PIL import Variable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models
#from model2 import CNN
import project_globals
from torch.autograd import Variable

# most of this code is from https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

model: any
device: any
train_dataloader: any


def run():
    batch_size = 32

    # define transformations for datasets
    train_transforms = project_globals.TRANSFORMS
    test_transforms = project_globals.TRANSFORMS

    # load datasets
    train_dataset = datasets.ImageFolder(project_globals.TRAINING_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(project_globals.TESTING_PATH, transform=test_transforms)

    # define train dataset loader
    global train_dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_model()
    return

def train_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 29), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 5

    # this found here https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    print_every = 20
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0
        for inputs, labels in train_dataloader:
            steps = steps + 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == print_every-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, steps + 1, running_loss / print_every))
                running_loss = 0.0
    torch.save(model, 'combo_model.pth')
    print('Im done')


def predict_image(image):
    # TODO: delete line below if unnecesary
    #image = transforms.ToPILImage(image) # converting to image we can use
    global model
    img_tensor = project_globals.TRANSFORMS(image).cuda() # TODO: might replace .cuda with .float if there are issues with CPU only machines
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    index = output.data.cpu().numpy().argmax()

    return index


if __name__ == "__main__":
    run()
