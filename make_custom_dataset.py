import cv2, sys, os
import re
from PIL import Image
import numpy as np

# lmao gonna make my own database
# dictionary for mapping keypresses to letters
KEY_TO_LETTER_DICT = {
    -1: "no input",
    97: "A",
    65: "A",
    98: "B",
    66: "B",
    99: "C",
    67: "C",
    100: "D",
    68: "D",
    101: "E",
    69: "E",
    102: "F",
    70: "F",
    103: "G",
    71: "G",
    104: "H",
    72: "H",
    105: "I",
    73: "I",
    106: "J",
    74: "J",
    107: "K",
    75: "K",
    108: "L",
    76: "L",
    109: "M",
    77: "M",
    110: "N",
    78: "N",
    111: "O",
    79: "O",
    112: "P",
    80: "P",
    113: "Q",
    81: "Q",
    114: "R",
    82: "R",
    115: "S",
    83: "S",
    116: "T",
    84: "T",
    117: "U",
    85: "U",
    118: "V",
    86: "V",
    119: "W",
    87: "W",
    120: "X",
    88: "X",
    121: "Y",
    89: "Y",
    122: "Z",
    90: "Z",
}

CHAR_LIST = ['A', 'B', 'C', 'D', 'E',
             'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y',
             'Z', 'del', 'nothing', 'space']
# 61 is equals key by backspace

vc: any
rval: any
frame: any
MAX_HEIGHT: int
MAX_WIDTH: int

def run():

    print(len(CHAR_LIST))
    dataset_train_dir: str = 'training_datasets/datasets/custom_dataset_train'
    make_dir(dataset_train_dir)
    dataset_test_dir: str = 'training_datasets/datasets/custom_dataset_test'
    make_dir(dataset_test_dir)
    make_dataset_directories(dataset_train_dir)
    make_dataset_directories(dataset_test_dir)
    current_dir: str


    cv2.namedWindow('preview')
    global vc
    vc = cv2.VideoCapture(0)
    global rval
    global frame
    if vc.isOpened():  # attempt to get the first frame
        # rval determines whether to keep running
        # frame is the actual image from the webcam
        rval, frame = vc.read()
    else:
        rval = False

    print("Type 1 to make training data, type 2 to make testing data")
    select_database = cv2.waitKey(1000000)

    if select_database == 49:
        current_dir = dataset_train_dir
    elif select_database == 50:
        current_dir = dataset_test_dir
    else:
        print('invalid input for database selection, exiting')
        return

    print('Database directory is ' + current_dir)
    print("Press the key you want to record pictures for. It's best to be signing before you start.")
    print("Press the 'Escape' key during recording when you want to stop recording.")
    print("Press 'Escape' to leave the program when you aren't recording.")

    global MAX_HEIGHT
    global MAX_WIDTH
    MAX_HEIGHT = frame.shape[0]
    MAX_WIDTH = frame.shape[1]
    while rval:
        print('Waiting for input...')
        rval, frame = vc.read()
        key = cv2.waitKey(200)

        height = 200
        width = 200
        x = int(MAX_WIDTH * .15)
        y = 0
        # y=int(MAX_HEIGHT*.1)
        # print(frame.size)
        # print(frame.shape[0])
        # print(frame.shape[1])
        subsection = frame[y:y + height, x:x + width].copy()

        upper_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        frame_with_rectangle = cv2.rectangle(frame, upper_left_corner, bottom_right_corner, color, thickness)
        cv2.imshow('preview', frame_with_rectangle)
        # cv2.imshow('preview2', converted_image)
        cv2.imshow('subsection', subsection)

        if (key <= 90 and key >= 65) or (key <= 122 and key >= 97):
            print(KEY_TO_LETTER_DICT[key])
            print('========RECORDING STARTING FOR ' + KEY_TO_LETTER_DICT[key] + '========')
            record_and_write(KEY_TO_LETTER_DICT[key], current_dir)
        elif key == -1:
            continue
        else:
            print(key)
            print('bad input')

        if key == 27:  # exit on escape key press
            break
    cv2.destroyAllWindows()
    return



def record_and_write(letter, current_dir):
    base_write_path = current_dir + '/' + letter + '/'
    #print(base_write_path)
    pic_list = os.scandir(base_write_path)
    pic_nums: list = []
    pic_num = 1
    # get pic numbers from pic list
    for pic in pic_list:
        find_numbers = re.findall(r'\d+', pic.name)
        pic_number = list(map(int, find_numbers))
        pic_nums.append(pic_number[0])

    # gets the largest number  in pic list
    if len(pic_nums) > 0:
        pic_nums.sort()
        pic_num = pic_nums[len(pic_nums) - 1]
        pic_num += 1
        print("There are already " + str(len(pic_nums)) + ' pictures for this entry, starting at picture number ' + str(pic_num))

    rval, frame = vc.read()
    while rval:
        # converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height = 200
        width = 200
        x = int(MAX_WIDTH * .15)
        y = 0
        # y=int(MAX_HEIGHT*.1)
        # print(frame.size)
        # print(frame.shape[0])
        # print(frame.shape[1])
        subsection = frame[y:y + height, x:x + width].copy()

        upper_left_corner = (x, y)
        bottom_right_corner = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        frame_with_rectangle = cv2.rectangle(frame, upper_left_corner, bottom_right_corner, color, thickness)
        cv2.imshow('preview', frame_with_rectangle)
        cv2.imshow('subsection', subsection)

        # ADJUST SPEED HERE
        # waitkey() takes in milliseconds as its parameter and waits that long for a keypress
        # this determines how fast you can write images
        key = cv2.waitKey(50)

        pic_write_path = base_write_path + letter + str(pic_num) + '.jpg'
        write_image(pic_write_path, subsection)
        pic_num += 1
        print("Writing to " + pic_write_path)
        rval, frame = vc.read()
        if key == 27:  # exit on escape key press
            print('========RECORDING STOPPED========')
            break
    return


def make_dataset_directories(dir):
    for letter in CHAR_LIST:
        formatted_path: str = replace_path_backslash(dir)
        formatted_path = formatted_path + '/' + letter
        make_dir(formatted_path)


def write_image(image_path, image):
    try:
        cv2.imwrite(image_path, image)
    except Exception as error:
        print(error)


def make_dir(local_path):
    try:
        os.mkdir(local_path)
    except OSError as error:
        print(error)


def replace_path_backslash(path):
    return path.replace("\\", "/")


if __name__ == "__main__":
    run()
