import cv2, sys, os
import numpy as np
# lmao gonna copy the whole database but make it canny god help me


def run():

    exec_path: str = os.path.dirname(sys.executable)

    old_train_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train'
    new_train_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train_canny'
    make_dataset(old_train_dataset_dir, new_train_dataset_dir)

    old_test_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_test'
    new_test_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_test_canny'
    make_dataset(old_test_dataset_dir, new_test_dataset_dir)

    #dir_list = os.scandir(old_train_dataset_dir)
    #for entry in dir_list:
    #    #formatted_path: str = entry.path
    #    formatted_path: str = replace_path_backslash(entry.path)
    #    formatted_path_split = formatted_path.split('/')
    #    #print(formatted_path)
    #    #print(formatted_path_split)
    #    new_letter_dir = new_train_dataset_dir + '/' + formatted_path_split[len(formatted_path_split) - 1]
    #    #print(new_letter_dir)
    #    make_dir(new_letter_dir)
    #    #print(formatted_path)
    #    pic_list = os.scandir(formatted_path)
    #    for pic in pic_list:
    #        formatted_pic_path = replace_path_backslash(pic.path)
    #        split_path = formatted_pic_path.split('/')
    #        #print(split_path)
    #        #print(len(split_path))
    #        write_path = new_letter_dir + '/' + split_path[len(split_path)-1]
    #        #print(write_path)
    #        old_image = cv2.imread(formatted_pic_path, -1)
    #        formatted_image = cv2.Canny(old_image, 150, 250)
    #        write_image(write_path, formatted_image)
#
    #        #print(formatted_pic_path)
#
    #new_test_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train_canny'
    #make_dir(new_train_dataset_dir)

def make_dataset(old_dir, new_dir):
    #new_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train_canny'
    make_dir(new_dir)

    #old_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train'
    dir_list = os.scandir(old_dir)
    for entry in dir_list:
        # formatted_path: str = entry.path
        formatted_path: str = replace_path_backslash(entry.path)
        formatted_path_split = formatted_path.split('/')
        # print(formatted_path)
        # print(formatted_path_split)
        new_letter_dir = new_dir + '/' + formatted_path_split[len(formatted_path_split) - 1]
        # print(new_letter_dir)
        make_dir(new_letter_dir)
        # print(formatted_path)
        pic_list = os.scandir(formatted_path)
        for pic in pic_list:
            formatted_pic_path = replace_path_backslash(pic.path)
            split_path = formatted_pic_path.split('/')
            # print(split_path)
            # print(len(split_path))
            write_path = new_letter_dir + '/' + split_path[len(split_path) - 1]
            # print(write_path)
            old_image = cv2.imread(formatted_pic_path, -1)
            formatted_image = cv2.Canny(old_image, 150, 250)
            write_image(write_path, formatted_image)

            # print(formatted_pic_path)


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
