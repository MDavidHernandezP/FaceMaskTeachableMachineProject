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
model_path = r"C:/Users/marit/OneDrive/Programming/UADY programming stuff\Detector de Cubrebocas/keras_model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading the model:", e)
    exit()

# Create the array of the right shape to feed into the keras model.
# The 'length' or number of images you can put into the array is, determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Initializing varibale (cam) to have an instance of the library (cv2) to have our camera working.
cam = cv2.VideoCapture(0)

# Adjust camera resolution.
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Resolution width.
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Resolution height.

# Get camera resolution.
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with the same resolution as the camera.
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', width, height)

# Adjust window resolution.
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 800, 600)  # Change to desired resolution.

# Define the size and position of the text.
font_scale = 0.5
font_thickness = 1
text_color = (0, 255, 0)
text_org = (10, 30)  # Position of text in window.

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
    prediction = model.predict(data)[0]

    # Print(prediction).
    if prediction[0] > 0.7:
        text = "Well-placed Mask"
    elif prediction[1] > 0.7:
        text = "Poorly placed Mask"
    elif prediction[2] > 0.7:
        text = "No Mask"
    else:
        text = "Mask not detected"

    # Put label text in the showed frame.
    cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Show frame.
    cv2.imshow('Frame', image)

# When the code finish the while loop, it releases and destroy all the windows from the cv2 and then the code finish.
cam.release()
cv2.destroyAllWindows()