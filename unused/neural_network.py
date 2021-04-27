import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#from PIL import Variable
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
#from model2 import CNN
import project_globals

# TODO: Decide whether to get project_globals working in this unused file or not

# most of this code is from https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

def run():

    batch_size = 32

    # define transformations for datasets
    global train_transforms
    global test_transforms
    train_transforms = project_globals.TRANSFORMS
    test_transforms = project_globals.TRANSFORMS

    # load datasets
    train_dataset = datasets.ImageFolder(project_globals.TRAINING_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(project_globals.TESTING_PATH, transform=test_transforms)

    # define train dataset loader
    global train_dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    #global classes
    #classes = project_globals.CLASSES
    global test_dataloader
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)
    #print(classes)
    #print(len(classes))
    return
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    global model

    try:
        # to use with cpu change to model = torch.load('testmodel.pth', map_location=torch.device('cpu'))
        model = torch.load('combo_model.pth')
        model.eval()
        run_webcam()
        #test_model()
        print('tested!')
        return
    except Exception as e:
        print(e)
        print('no model found, training machine to build one')
        train_model_2()
        return

def train_model_2():
    print('RUNNING TRAIN MODEL 2')
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 29), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 5
    steps = 0

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
                      (epoch + 1, steps + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(model, 'combo_model.mdl')
    print('Im done')


def train_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 27), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    print(model)

    epochs = 2
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        #if epoch < 4:
        #    for g in optimizer.param_groups:
        #        if epoch == 1:
        #            g['lr'] = 0.1
        #        elif epoch == 2:
        #            g['lr'] = 0.01
        #        elif epoch == 3:
        #            g['lr'] = 0.001


        for inputs, labels in train_dataloader:
            steps += 1
            # print('step: ' + str(steps))
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
        sub.set_title(str(project_globals.CLASSES[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()



def run_webcam():
    cv2.namedWindow('preview')
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # attempt to get the first frame
        # rval determines whether to keep running
        # frame is the actual image from the webcam
        rval, frame = vc.read()
    else:
        rval = False
    MAX_HEIGHT = frame.shape[0]
    MAX_WIDTH = frame.shape[1]
    predictions = []
    prediction: str
    while rval:
        # code from https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
        #converted_image = frame[:, :, [0, 1, 2]]
        #converted_image = frame[:, :, [2, 1, 0]]
        converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height = 200
        width = 200
        x=int(MAX_WIDTH*.15)
        y=0
        #y=int(MAX_HEIGHT*.1)
        #print(frame.size)
        #print(frame.shape[0])
        #print(frame.shape[1])
        #subsection = converted_image[y:y+height, x:x+width].copy()
        subsection = frame[y:y+height, x:x+width].copy()

        #subsection_resized = cv2.resize(subsection, (0,0), fx=0.5, fy=0.5)
        #subsection_canny = cv2.Canny(subsection, 150, 250)
        converted_image_values = Image.fromarray(subsection)


        #frame_prediction = predict_image(converted_image_values)
        #predictions.append(frame_prediction)
        #if len(predictions) == 15:
        #    prediction = average_prediction(predictions)
        #    print('predicted as ' + classes[prediction])
        #    predictions = []

        #print('predicted as ' + classes[frame_prediction])
        upper_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        frame_with_rectangle = cv2.rectangle(frame, upper_left_corner, bottom_right_corner, color, thickness)
        text_background_br_corner = (MAX_WIDTH, MAX_HEIGHT)
        text_background_ul_corner = (MAX_WIDTH - 150, MAX_HEIGHT - 150)
        text_background_color = (0, 0, 0)
        frame_formatted = cv2.rectangle(frame_with_rectangle, text_background_ul_corner, text_background_br_corner, text_background_color, -1)
        frame_prediction = predict_image(converted_image_values)
        sign = project_globals.CLASSES[frame_prediction]
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_bl_corner = (MAX_WIDTH - 125, MAX_HEIGHT - 70)
        size = 1
        text_color = (255, 255, 255)
        linetype = 2
        cv2.putText(img=frame_formatted, text=sign, org=text_bl_corner, fontFace=text_font, fontScale=size, color=text_color, lineType=linetype)
        cv2.imshow('preview', frame_formatted)
        #cv2.imshow('preview2', converted_image)
        cv2.imshow('subsection', subsection)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on escape key press
            break
    #cv2.destroyWindow("preview")
    cv2.destroyAllWindows()

def average_prediction(predictions):
    pred_sum = sum(predictions)

    pred_sum = pred_sum / len(predictions)
    if (pred_sum - math.floor(pred_sum)) < 0.5:
        return math.ceil(pred_sum)
    else:
        return math.floor(pred_sum)



def predict_image(image):
    # TODO: delete line below if unnecesary
    #image = transforms.ToPILImage(image) # converting to image we can use

    img_tensor = test_transforms(image).cuda() # TODO: might replace .cuda with .float if there are issues with CPU only machines
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    index = output.data.cpu().numpy().argmax()

    return index

def get_random_images(num):
    data = datasets.ImageFolder(project_globals.TESTING_PATH, transform=test_transforms)
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
