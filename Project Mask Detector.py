# Importing Libraries.
from keras.models import load_model
import numpy as np
import cv2

'''
VERY IMPORTANT TO CHECK COMPATIBILITY BETWEEN LIBRARIES.
At the last moment I ran this code 03/20/2024 I'm using python 3.11.8 and the version 3.12.0 of both Tensorflow and Keras. 
'''

# Loading the model, (you'll have to change the path to where you have the model saved).
# Note to ME: in the path just change (marit) to (Personal) to run the code in the desktop PC.
model = load_model(r"C:/Users/marit/OneDrive/Programming/UADY programming stuff\Detector de Cubrebocas/keras_model.h5")

# Create the array of the right shape to feed into the keras model.
# The 'length' or number of images you can put into the array is, determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Initializing varibale (cam) to have an instance of the library (cv2) to have our camera working.
cam = cv2.VideoCapture(0)

# Initializing variable (text) to give it a string depending of what the model recognizes.
text = ""

while True:
    # Print to leave the user know that the code is working by the terminal.
    print('The code is working, press the (q) key to finish the code.')

    # If the user press the (q) key from the keyboard then the while loop will finish and so do the code.
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Capture the frame.
    ret, image = cam.read()
    
    # Resizing the image to be at least 224x224 and then cropping from the center.
    image = cv2.resize(image, (224, 224))

    # Turn the image into a numpy array.
    image_array = np.asarray(image)

    # Normalize the image.
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array.
    data[0] = normalized_image_array

    # run the inference.
    prediction = model.predict(data)

    # Print(prediction).
    for i in prediction:
        if i[0] > 0.7:
            text ="Well-placed Mask"
        if i[1] > 0.7:
            text ="Poorly placed Mask"
        if i[2] > 0.7:
            text ="No Mask"
            
        # Resizing the image of the frame.
        image = cv2.resize(image,(500, 500))

        # Put label text in the showed frame.
        cv2.putText(image, "Label: " + text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0), 2, 1)

    # Show frame.
    cv2.imshow('frame',image)

# When the code finish the while loop, it releases and destroy all the windows from the cv2 and then the code finish.
cam.release()
cv2.destroyAllWindows()