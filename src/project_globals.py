from torchvision import transforms


# file for important stuff shared by a number of files


TRAINING_PATH: str = 'training_datasets/datasets/merged_dataset_train'
TESTING_PATH: str = 'training_datasets/datasets/merged_dataset_test'
TRANSFORMS = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

CLASSES: list = ['A', 'B', 'C', 'D', 'E',
                 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O',
                 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y',
                 'Z', 'nothing']


#def get_train_dataloader():
#
#
#def get_test_dataloader():
#