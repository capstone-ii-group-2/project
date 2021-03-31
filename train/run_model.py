import numpy as np
import torch
import cv2
import seaborn as sns
from torch.autograd import Variable
#from generate_model import reshape_to_2d
from network import Network

def reshape_to_2d(data, dim):
    reshaped = []
    for i in data:
        reshaped.append(i.reshape(1, dim, dim))

    return np.array(reshaped)

def run():
    model = torch.load("C:/Users/Samuel/Documents/repos/school/senior/capstone/project/models/mnist_model")
    model.eval()

    cv2.namedWindow('preview')
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # attempt to get the first frame
        # rval determines whether to keep running
        # frame is the actual image from the webcam
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # code from https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
        test_frame = cv2.resize(frame, (360,  360))
        print(cv2.getWindowImageRect('preview'))
        formattedframe = reshape_to_2d(test_frame, 360) # TODO: this doesn't work
        #formattedframe = torch.FloatTensor(formattedframe)
#
        #formattedframe = Variable(formattedframe)
        #result = model(formattedframe)

        cv2.imshow('preview', test_frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on escape key press
            break
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    run()
