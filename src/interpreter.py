import cv2
from PIL import Image
import torch
import project_globals

device: any
model: any


def run():
    global model
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = torch.load("combo_model.mdl")
        print("Using CUDA")
    else:
        model = torch.load(f="combo_model.mdl", map_location=torch.device('cpu'))
        print("Using CPU")

    model.eval()
    run_webcam()
    return


def run_webcam():
    global device
    pred_img: any
    if torch.cuda.is_available():
        pred_img = predict_image_cuda
    else:
        pred_img = predict_image_cpu

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

    while rval:
        # code from https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

        # getting the subsection of the webcam image to be interpreted by the model
        height = 200
        width = 200
        x = int(MAX_WIDTH * .15)
        y = 0
        subsection = frame[y:y + height, x:x + width].copy()

        # converting the image into something interpretable by the model, then interpreting it
        converted_image_values = Image.fromarray(subsection)
        frame_prediction = pred_img(converted_image_values)

        # drawing the input box rectangle
        upper_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        frame_with_rectangle = cv2.rectangle(frame, upper_left_corner, bottom_right_corner, color, thickness)

        # drawing the black background for prediction output box
        text_background_br_corner = (MAX_WIDTH, MAX_HEIGHT)
        text_background_ul_corner = (MAX_WIDTH - 150, MAX_HEIGHT - 150)
        text_background_color = (0, 0, 0)
        frame_formatted = cv2.rectangle(frame_with_rectangle, text_background_ul_corner, text_background_br_corner,
                                        text_background_color, -1)

        # drawing the prediction inside the prediction box
        sign = project_globals.CLASSES[frame_prediction]
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_bl_corner = (MAX_WIDTH - 125, MAX_HEIGHT - 70)
        size = 1
        text_color = (255, 255, 255)
        linetype = 2
        cv2.putText(img=frame_formatted, text=sign, org=text_bl_corner, fontFace=text_font, fontScale=size,
                    color=text_color, lineType=linetype)

        # showing the frame with input box and output box
        cv2.imshow('preview', frame_formatted)
        cv2.imshow('subsection', subsection)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on escape key press
            break
    cv2.destroyAllWindows()


def predict_image_cuda(image):
    global model
    img_tensor = project_globals.TRANSFORMS(image).cuda()
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    index = output.data.cpu().numpy().argmax()

    return index

def predict_image_cpu(image):
    global model
    img_tensor = project_globals.TRANSFORMS(image).float()
    img_tensor = img_tensor.unsqueeze(0)
    output = model(img_tensor)
    index = output.data.cpu().numpy().argmax()

    return index


if __name__ == "__main__":
    run()