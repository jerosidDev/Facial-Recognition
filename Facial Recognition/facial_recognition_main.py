from pre_input_training import addFaceData
from input_standardization import generateStandardInputs

while True:

    #generateStandardInputs()
    #addFaceData()


    print("Options available")
    print("     1: Add training data")
    print("     2: Train the current model")
    print("     3: Recognize faces in an image")
    print("     4: Exit")

    optionSelected = input("Selected option: ")
    print(optionSelected)

    if optionSelected == "1":
        addFaceData();
    if optionSelected == "2":
        generateStandardInputs();
    elif optionSelected == "4":
        break;
    else:
        print("Unrecognized option")
