from torch import nn
import torch

# tutorial https://nestedsoftware.com/2019/09/09/pytorch-image-recognition-with-convolutional-networks-4k17.159805.html
# my attempt at making a custom network. It's from the link above
# the old version of the model is commented out, it followed the tutorial from the comment in neural_network.py
# both versions of this file are unused

INPUT_SIZE = 200*200*3
OUTPUT_SIZE = 29
NUM_EPOCHS = 30
LEARNING_RATE = 3.0

class ConvNNTwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.conv_layer_2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(32*40*53*53, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

        #self.cnn_layers = nn.Sequential(
        #    nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(4),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(4),
        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
#
        #)

        #self.linear_layers = nn.Sequential(
        #    nn.Linear(4 * 7 * 7,10)
        #)


    def forward(self, x):
        #x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        #x = self.linear_layers(x)
        #return x
        x = self.conv_layer_1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv_layer_2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        #print(x.shape)
        x = x.view(-1, 32*40*53*53)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x
