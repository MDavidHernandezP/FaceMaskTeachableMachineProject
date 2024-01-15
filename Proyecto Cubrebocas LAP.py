from keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model(r"C:/Users/marit/OneDrive/Programming/UADY programming stuff\Detector de Cubrebocas/keras_model.h5")

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# Camera
cam = cv2.VideoCapture(0)

text = ""

while True:
    print('Awebo nos estamos ejecutando presiona (q) para terminar')

    if cv2.waitKey(1) == ord('q'):
        break
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    #Capture the frame
    ret,image = cam.read()
    
    #resizing the image
    image = cv2.resize(image, (224, 224))

    #turn the image into a numpy array
    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # print(prediction)
    for i in prediction:
        if i[0] > 0.7:
            text ="Cubrebocas bien colocado"
        if i[1] > 0.7:
            text ="Cubrebocas mal colocado"
        if i[2] > 0.7:
            text ="Sin Cubrebocas"            
            
        #resizing the image
        image = cv2.resize(image,(500, 500))
        #Put label text
        cv2.putText(image, "Label: " + text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0), 2, 1)

    #Show frame
    cv2.imshow('frame',image)

#Release and destroy all windows
cam.release()
cv2.destroyAllWindows()