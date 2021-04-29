# Python Sign Language Interpreter

## Instructions

### **Get the correct version of Python**

* [Python version 3.8.0](https://www.python.org/downloads/release/python-380/)
    * This is the version of Python we used while developing this project, so it's the version we recommend you use. It's possible that a newer version of Python would work, but we strongly recommend against using an older version

### **Install the necessary packages**

* Use ```python -m pip install -r packages.txt``` to install the necessary packages
  * opencv-python should automatically install numpy during its installation if you don't have it, but run ```pip install numpy``` in the event that it doesn't
* Go to [PyTorch's website](https://pytorch.org/get-started/locally/) and install the 'Stable' version of PyTorch
    * You must pick the version of PyTorch based on your package manager (I recommend pip) & whether you have a GPU with CUDA installed
        * Copy and paste the command from the website based on these
    
### Having trouble installing the necessary packages? (next step)
* Use ```pip install --upgrade pip``` to upgrade your version of pip. This is one of the issues that can prevent you from installing packages.
    * Sometimes there is an error message during this process that claims that pip was not able to upgrade, however sometimes the upgrade still works so it's worth checking your version of pip even if the upgrade 'fails'.
    * In the event that pip does not upgrade, try running the original upgrade command as an Adminstrator or adding ```--user``` to the end of the command.
    
### Run the Interpreter
* To run the interpreter on a Windows machine, open the Command Prompt in the ```project``` directory (should be the base directory) and type ```run``` to execute ```run.bat```, which will then launch the live interpreter
* To run the trainer on a Windows machine, follow the instructions above for running the interpreter, but run ```train.bat``` instead of ```run.bat```


### Using the Interpreter
 Once you have the interpreter running, you should be able to place your hand in the input box (blue box) on the screen to start interpreting. Press the ```esc``` key to exit the program.

## Datasets

[Dataset for Sign Language](https://www.kaggle.com/grassknoted/asl-alphabet) - Sign language dataset we found online that we merged with our own dataset to produce our final model



## Using the Kaggle dataset
The dataset that comes with this project is the merged dataset, consisting of the Kaggle dataset and our own pictures. If you just want to use that dataset, you can skip this section.
1. Download [this](https://www.kaggle.com/grassknoted/asl-alphabet) dataset
2. Unzip the dataset
3. Move both ```asl_alphabet_train``` and ```asl_alphabet_test``` into the ```training_datasets``` folder
    * __IMPORTANT!__ Make sure the path for both looks like this: ```training_datasets/asl_alphabet_<train OR test>/<data>``` as the dataset directories are structured like ```asl_alphabet_train/asl_alphabet_<train OR test>/<data>``` in the zipped folder
    * An example of what the path should look like starting from ```project``` is ```training_datasets/datasets/asl_alphabet_train```
    * Note: all letters in the dataset must be grouped into their own directories, in ```asl_alphabet_train``` this shouldn't be a problem, but for ```asl_alphabet_test``` a directory will need to be created for each picture
    * Note 2: You should delete the ```del``` and ```space``` entries from both the train and test datasets as they were not included in the final version of this project
4. You are now ready to use the dataset


## Making your own dataset
If you want to make your own dataset, follow these steps:
1. Move/copy the ```make_custom_dataset.py``` file out of the ```scripts``` directory and into the ```project``` directory (should be base directory)
2. Run the script with ```python make_custom_dataset.py```
3. Once running and with the window selected (not the console), press ```1``` on your keyboard to make a training dataset, or ```2``` to make a testing dataset
   * Once you have selected one of these options, you must exit the program to select a different one
4. Press the letter you want to create images for on your keyboard to start taking pictures and writing them to your custom dataset
   * Ex: Press `A` on your keyboard to start taking pictures for the letter ```A```
    * ```/``` is reserved for the ```nothing``` entry in the database, press it to make entries for ```nothing```
    * Note: only takes picture of the image in the input box
    
5. Press ```esc``` key to stop taking pictures and return to the letter selection menu
6. Press ```esc``` again to exit the program or press a different key to start recording pictures for that letter

## Useful Links
[Sign Language Interpreter using Deep Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning) - Very close to what we want to achieve, UI was inspired by this project

[PyTorch](https://pytorch.org/) - The library we will be using to train the model

[Webcam Tutorial](https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python) - Page with the example I followed tog et the webcam running

[Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) - Tutorial from the official PyTorch website that I used to learn how to train a model

[Another Training Tutorial](https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5) - Another tutorial that was very helpful for learning how to train a model
