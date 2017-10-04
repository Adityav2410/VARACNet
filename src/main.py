#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:08:24 2017

@author: Vamshi, Akshaya, Rishabh, Aditya, Chetan
"""

import os
import numpy as np

from utils import *
from config import *
from models import Models

from os import walk
from os.path import join

import keras
from keras.optimizers import Adam
from keras.utils import plot_model
from model4_SqueezeNet import SqueezeNet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def loadModels():
  """Load and compile submodels"""
  modelHandler = Models()
  model1 = modelHandler.getModel1(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES)
  model2 = modelHandler.getModel1(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES)
  model3 = modelHandler.getModel1(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES)
  model4 = SqueezeNet()

  model1.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  model1.load_weights('./model1_Weights.hdf5')

  model2.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  model2.load_weights('./model2_Weights.hdf5')

  model3.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  model3.load_weights('model3_Weights.hdf5')

  model4.compile(optimizer=Adam(lr=0.0001, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  model4.load_weights('model4_Weight.hdf5')
  
  modelList = [model1 , model2, model3, model4]
  nModel = len(modelList)
  return modelList, nModel

  
def main():
  correctClassify = 0
  incorrectClassify = 0
  [modelList, nModels] = loadModels()

  for dir1 in dirs:
	    path = base_dir + training_dir + dir1
	    label = int(dir1[-1:])
	    for dirpath, dirnames, filenames in walk(path):
	        if (len(filenames) > 0):
	            for file_name in filenames:
	                file_path = join(path, file_name)
	                img = load_image_custom(file_path)
	                total_prob = np.zeros(3)
	                for model in modelList:
	                    prob = model.predict(img)[0]
	                    total_prob = total_prob + prob;
	                img_prob = total_prob/nModel
	                img_class = np.argmax(img_prob)    
	                if (img_class == label):
	                    correctClassify = correctClassify + 1
	                else:
	                    incorrectClassify = incorrectClassify + 1

  classifyAccuracy = 1.0*correctClassify/(correctClassify + incorrectClassify)
  print('Ensemble Validation Accuracy', classifyAccuracy)


if __name__ == '__main__':
    main()