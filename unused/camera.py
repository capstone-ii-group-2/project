import cv2

# uncomment this to test your webcam
cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)

if vc.isOpened():  # attempt to get the first frame
    # rval determines whether to keep running
    # frame is the actual image from the webcam
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # tutorial for this https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

    cv2.imshow('preview', frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on escape key press
        break
cv2.destroyWindow("preview")