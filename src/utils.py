
import os
import numpy as np

from keras.utils import plot_model
from keras.preprocessing import image

def load_image_custom(image_path):
    img = image.load_img(image_path, target_size = (224, 224, 3))
    data = image.img_to_array(img)/255.0
    
    img_1 = np.empty([1,224,224,3], dtype=float)
    img_1[0] = data
    return img_1

def generateModelImages(modelList):
    for model in modelList:
        plot_model(model, to_file='model.png')    

