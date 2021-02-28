# Python Sign Language Interpreter

## Instructions

### **Get the correct version of Python**

* [Python version 3.8.0](https://www.python.org/downloads/release/python-380/) is required for this project to work. Higher & lower versions of Python will not work. Installation will be different based on your platform, but you should install it in a custom location that is easy to access. For example, I installed it in 'C:\Python\Python38' on my machine (Windows).


### **Install the necessary packages**

* Use ```py -m pip install -r packages.txt``` to install some of the necessary packages
* Go to [PyTorch's website](https://pytorch.org/get-started/locally/) and install the correct version of PyTorch if you're going to be training any models
    * You must pick the version of PyTorch based on your package manager & whether you have a GPU with CUDA installed

## Useful Links
[Sign Language Interpreter using Deep Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning) - Very close to what we want to achieve

[Tensorflow](https://www.tensorflow.org/) - A library we will be using for this project

[Webcam Tutorial](https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python) - Page with the example I followed tog et the webcam running

## Datasets

[Dataset for Sign Language](https://www.kaggle.com/datamunge/sign-language-mnist) - Sign language dataset to train the model (we probably won't use this)

[Google Teachable Machine](https://teachablemachine.withgoogle.com/) - Thing we might have to use to create our own dataset

[Another Dataset](http://vlm1.uta.edu/~athitsos/asl_lexicon/) - Another dataset that we could possibly use

## Making your own dataset
1. Open the [Google Teachable Machine](https://teachablemachine.withgoogle.com/) website and click 'Get Started' in the upper right hand corner.
2. Pick the 'Image Project' from the list
3. Click the 'Webcam' button in either Class 1 or Class 2. Click and hold the 'Hold to Record' button to record a series of images.
4. Once you have the images you want, click the hamburger (three dot) menu in the upper right corner of the Class and click 'Download Samples'
5. You should now have a zip file with all your recorded images