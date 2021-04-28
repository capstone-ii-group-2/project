import numpy as np
import torch
import cv2
import seaborn as sns
from torch.autograd import Variable
#from generate_model import reshape_to_2d
from network import Network

def reshape_to_2d(data, dim):
    reshaped = []
    count = 0
    for i in data:
        count += 1
        #print(i)
        reshaped.append(i.reshape(1, dim, dim))

   # print(count)
    #return
    return np.array(reshaped)

def run():
    model = torch.load("C:/Users/Samuel/Documents/repos/school/senior/capstone/project/models/mnist_model")
    model.eval()

    #cv2.namedWindow('preview')
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # attempt to get the first frame
        # rval determines whether to keep running
        # frame is the actual image from the webcam
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # code from https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
        #test_frame = cv2.resize(frame, (360,  360))
        #test_frame2 = frame[:, :, [2,0,1]]


        resized_frame = cv2.resize(frame, (400, 400))
        #print(resized_frame.shape)
        #print(resized_frame.size)
        #print(resized_frame)
        mono_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        mono_frame_formatted = torch.FloatTensor(mono_frame)
        test_var_sample = Variable(mono_frame_formatted)
        prediction = model(test_var_sample)
        #print(mono_frame.shape)
        #print(mono_frame.size)
        #print(mono_frame)
        #example_array = [0, 1, 2, 3, 4, 5]
        #example_array = np.array(example_array)
        #print(example_array.shape)
        #print(example_array.size)
        #print(example_array)
        #print(cv2.getWindowImageRect('preview'))
        #formattedframe = reshape_to_2d(test_frame, 360)


        #reshape_to_2d(mono_frame,20)


        #print(test_frame)

        #formattedframe = torch.FloatTensor(formattedframe)
#
        #formattedframe = Variable(formattedframe)
        #result = model(formattedframe)

        cv2.imshow('frame', frame)
        cv2.imshow('mono_frame', mono_frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on escape key press
            break
    #cv2.destroyWindow("preview")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
