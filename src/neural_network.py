import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models


def run():
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
    test_dataset = datasets.ImageFolder(testing_path, transform=test_transforms)
    num_train = len(train_dataset)
    print(train_dataset)

    # define train dataset loader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    classes = train_dataloader.dataset.classes
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
    print(classes)
    print(len(classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #device = torch.device('cpu')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512,29), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [],[]

    for epoch in range(num_epoch):
        for inputs, labels in train_dataloader:
            steps +=1
            print('step: ' + str(steps))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss =0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        train_losses.append(running_loss/len(train_dataloader))
                        test_losses.append(test_loss/len(test_dataloader))
                        print(f"Epoch {epoch + 1}/{epochs}.. "
                              f"Train loss: {running_loss / print_every:.3f}.. "
                              f"Test loss: {test_loss / len(test_dataloader):.3f}.. "
                              f"Test accuracy: {accuracy / len(test_dataloader):.3f}")
                        running_loss = 0
                        model.train()
    torch.save(model, 'testmodel.pth')

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    run()
