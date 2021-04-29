import torch
from torch import nn
from torch import optim
from torchvision import datasets, models
import project_globals

# tutorial for some of this https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

model: any
device: any
train_dataloader: any


def run():
    batch_size = 32

    # define transformations for datasets
    train_transforms = project_globals.TRANSFORMS

    # load datasets
    train_dataset = datasets.ImageFolder(project_globals.TRAINING_PATH, transform=train_transforms)

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

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 27), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 5

    # tutorial for this section https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
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
    torch.save(model, 'combo_model.mdl')
    print('Im done')


if __name__ == "__main__":
    run()
