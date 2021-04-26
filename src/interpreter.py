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

def run():
    model = torch.load("combo_model.pth")
    model.eval()
    run_webcam()



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
        converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height = 200
        width = 200
        x = int(MAX_WIDTH * .15)
        y = 0
        subsection = frame[y:y + height, x:x + width].copy()
        converted_image_values = Image.fromarray(subsection)
        upper_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        frame_with_rectangle = cv2.rectangle(frame, upper_left_corner, bottom_right_corner, color, thickness)
        text_background_br_corner = (MAX_WIDTH, MAX_HEIGHT)
        text_background_ul_corner = (MAX_WIDTH - 150, MAX_HEIGHT - 150)
        text_background_color = (0, 0, 0)
        frame_formatted = cv2.rectangle(frame_with_rectangle, text_background_ul_corner, text_background_br_corner,
                                        text_background_color, -1)
        ##frame_prediction = predict_image(converted_image_values)
        ##sign = classes[frame_prediction]
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_bl_corner = (MAX_WIDTH - 125, MAX_HEIGHT - 70)
        size = 1
        text_color = (255, 255, 255)
        linetype = 2
        ##cv2.putText(img=frame_formatted, text=sign, org=text_bl_corner, fontFace=text_font, fontScale=size,
        ##            color=text_color, lineType=linetype)
        cv2.imshow('preview', frame_formatted)
        ##cv2.imshow('preview2', converted_image)
        cv2.imshow('subsection', subsection)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on escape key press
            break
    # cv2.destroyWindow("preview")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()