
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
    print("Doesn't work yet\n\n")

def train():
    print("Doesn't work yet\n\n")

main()
