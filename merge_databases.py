import cv2, sys, os
import re
import numpy as np
# lmao gonna copy the whole database but make it canny god help me

# READ ME BEFORE YOU RUN THIS
# The 'merged' databases being written to must be empty or this will break
# You will have to change the name of the databases you want to use at the top of the 'run' function
# it doesn't really matter which you use for first and second so long as they are both databases you want to merge


def run():

    exec_path: str = os.path.dirname(sys.executable)

    # SPECIFY TRAIN DATASETS HERE
    first_train_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_train'
    second_train_dataset_dir: str = 'training_datasets/datasets/custom_dataset_train'
    merged_train_dataset_dir: str = 'training_datasets/datasets/merged_dataset_train'
    merge_datasets(first_train_dataset_dir, second_train_dataset_dir, merged_train_dataset_dir)

    # SPECIFY TEST DATASETS HERE
    first_test_dataset_dir: str = 'training_datasets/datasets/asl_alphabet_test'
    second_test_dataset_dir: str = 'training_datasets/datasets/custom_dataset_test'
    merged_test_dataset_dir: str = 'training_datasets/datasets/merged_dataset_test'
    merge_datasets(first_test_dataset_dir, second_test_dataset_dir, merged_test_dataset_dir)

def merge_datasets(first_dir, second_dir, merge_dir):
    make_dir(merge_dir)

    dir_list = os.scandir(first_dir)
    pic_num_list: list = []
    for entry in dir_list:
        pic_nums: list = []
        formatted_path: str = replace_path_backslash(entry.path)
        formatted_path_split = formatted_path.split('/')
        new_letter_dir = merge_dir + '/' + formatted_path_split[len(formatted_path_split) - 1]
        print('Writing to ' + new_letter_dir)
        make_dir(new_letter_dir)
        pic_list = os.scandir(formatted_path)
        # writing pictures from first database
        for pic in pic_list:
            find_numbers = re.findall(r'\d+', pic.name)
            pic_number = list(map(int, find_numbers))
            pic_nums.append(pic_number[0])
            formatted_pic_path = replace_path_backslash(pic.path)
            split_path = formatted_pic_path.split('/')
            write_path = new_letter_dir + '/' + split_path[len(split_path) - 1]
            old_image = cv2.imread(formatted_pic_path, -1)
            write_image(write_path, old_image)

            # print(formatted_pic_path)
        pic_nums.sort()
        pic_num_list.append(pic_nums[len(pic_nums)-1])
    # performing same operation on second directory
    dir_list = os.scandir(second_dir)
    pic_num_index = 0
    for entry in dir_list:
        pic_num = pic_num_list[pic_num_index] + 1
        formatted_path: str = replace_path_backslash(entry.path)
        formatted_path_split = formatted_path.split('/')
        new_letter_dir = merge_dir + '/' + formatted_path_split[len(formatted_path_split)-1]
        make_dir(new_letter_dir)
        print('Writing to ' + new_letter_dir)

        pic_list = os.scandir(formatted_path)
        for pic in pic_list:
            #find_numbers = re.findall(r'\d+', pic.name)
            formatted_pic_path = replace_path_backslash(pic.path)
            write_path = new_letter_dir + '/' + formatted_path_split[len(formatted_path_split)-1] + str(pic_num) + '.jpg'
            old_image = cv2.imread(formatted_pic_path, -1)
            write_image(write_path, old_image)
            pic_num += 1
    pic_num_index += 1




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
