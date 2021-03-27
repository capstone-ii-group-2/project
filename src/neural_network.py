import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from PIL import Variable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models
from torch.autograd import Variable


# most of this code is from https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

def run():
    global training_path
    training_path = 'training_datasets/datasets/asl_alphabet_train'
    global testing_path
    testing_path = 'training_datasets/datasets/asl_alphabet_test'

    test_size = 0.2
    batch_size = 32
    global num_epoch
    num_epoch = 10
    learning_rate = 0.001
    num_classes = 29

    # define transformations for datasets
    global train_transforms
    global test_transforms
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
    global train_dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    global classes
    classes = train_dataloader.dataset.classes
    global test_dataloader
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
    print(classes)
    print(len(classes))

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    global model

    try:
        model = torch.load('testmodel.pth')
        model.eval()
        test_model()
        print('tested!')
        return
    except Exception:
        print('no model found, training machine to build one')
        return
        train_model()



def train_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 29), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(num_epoch):
        for inputs, labels in train_dataloader:
            steps += 1
            print('step: ' + str(steps))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        train_losses.append(running_loss / len(train_dataloader))
                        test_losses.append(test_loss / len(test_dataloader))
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


def test_model():
    #model = torch.load('testmodel.pth')
    #model.eval()

    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image)
        sub = fig.add_subplot(1, len(images), ii + 1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(training_path, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

if __name__ == "__main__":
    run()
