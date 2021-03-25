import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import operator
import cv2
import sys, os
from PIL import Image

def facetones_model():
    # Loading the model
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")

    # Category dictionary
    categories = {0: 'dark', 1: 'deep', 2: 'fair', 3: 'light', 4: 'light_medium', 5: 'medium_tan'}

    test_image = cv2.imread('imgss.jpg')
    # img_array = cv2.imread('imgss.jpg')
    # plt.imshow(img_array)
    # plt.show()

    test_image= cv2.resize(test_image, (64, 64))

    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 3))
    prediction = {'dark': result[0][0],
                      'deep': result[0][1],
                      'fair': result[0][2],
                      'light': result[0][3],
                      'light_medium': result[0][4],
                      'medium_tan': result[0][5]}
     # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    print(prediction)
    ress = prediction[0]
    print(ress[0])

    return ress


facetones_model()