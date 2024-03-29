import cv2
import torch
import math
import numpy as np
import time

# uncomment this to test your webcam

#cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)

if vc.isOpened(): # attempt to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#img = cv2.imread('training_datasets/asl_alphabet_train/B/B713.jpg', 0)
img_number = 1

while rval and img_number < 3001:
    # tutorial https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
    # tutorial for Canny https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/
    #cv2.imwrite('testing.jpg', cv2.Canny(img, 200, 300))
    #cv2.imshow('canny', cv2.imread('testing.jpg'))
    time.sleep(0.01)
    img = cv2.imread('training_datasets/asl_alphabet_train/B/B'+ str(img_number)+'.jpg', 0)
    #hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower = np.array([0, 48, 80], dtype = "uint8")
    #upper = np.array([20, 255, 255], dtype = "uint8")
    #skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    #blurred = cv2.blur(skinRegionHSV, (2,2))
    #ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    cv2.imshow('B', cv2.Canny(img, 200, 300))
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #try: 
    #contours = max(contours, key=lambda x: cv2.contourArea(x),default=0)
    #except:
    #    print('oops')
    #contours = max(contours, key=lambda x: cv2.contourArea(x),default=0)
    #cv2.drawContours(frame, contours, -1, (255,255,0), 2)
    #cv2.imshow("contours", frame)
#
    ##-------------------------------------------------------
    ##   HULL CHANGES
    ##-------------------------------------------------------
    #for i in range(len(contours)):
    #    hull = cv2.convexHull(contours[i])
    #    cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
    #
    #cv2.imshow("hull", frame)   
    ##-------------------------------------------------------
    print(img_number)
    img_number = img_number + 1
    rval, frame = vc.read()
    #cv2.destroyWindow(str(img_number))
    key = cv2.waitKey(20)
    if key == 27: # exit on escape key press
        break
cv2.destroyAllWindows()


# uncomment this to test pytorch
'''
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
'''
