# READ THE README.md

import numpy as np
import comtypes as com

def main():
    
    menuInput = None
    print("Welcome to the Sign Language Interpreter!\n")
    while(menuInput != 3):
        print("Enter the numbers below to select the shown menu")
        print("1. Run (interpret sign language)")
        print("2. Train (give input to teach the interpreter")
        print("3. Quit (exits the program)")
        userInput = input("Enter choice: ")
        userInput = int(userInput)

        if(userInput == 1):
            run()
        if(userInput == 2):
            train()
        if(userInput == 3):
            print("Goodbye!")
            exit()

def run():
    print('testing numpy')
    print(np.floor(5.5))
    bigInt = np.uint64(pow(2, 63) - 1) # should set to max value of 64 bit unsigned integer (18,446,744,073,709,551,615 )
    print(bigInt)
    testInt = pow(2,32) - 1 #should set testInt to max value of 32 bit integer 4,294,967,295
    print(testInt)
    testInt = np.left_shift(testInt, 32) # should set testInt to 0 (doesn't actually do this)
    print(testInt)
    print("Doesn't work yet\n\n")

def train():
    print("Doesn't work yet\n\n")

main()
