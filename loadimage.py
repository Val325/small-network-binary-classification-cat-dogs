import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

while True:
    image_size = (180, 180)
    print("------------------------------------------------------------------")
    print("|          NEURAL NETWORK, WHICH DETERMINE CAT OR DOG            |")
    print("------------------------------------------------------------------")
    print("You loading image and this neural network determine is cat or dogs")
    predict = input("Yours file: ") # input 
    model = load_model("model/CatsAndDogs")


    img = keras.utils.load_img(
        predict, target_size=image_size
    )
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
