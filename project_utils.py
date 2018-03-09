#general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
#keras specific imports
from keras import Model, Input
from keras.layers import Dense, Flatten,GlobalMaxPool2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from keras.applications import VGG16,vgg16
from keras.applications import ResNet50, resnet50
from keras.applications import InceptionV3, inception_v3


def prep_for_vgg(path):
    img = image.load_img(path, target_size = (224, 224) )
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = vgg16.preprocess_input(x, mode='tf')
    return x

def get_test():
    '''Creates a test set from a folder I set aside to test results after modeling'''
    
    citypaths = os.listdir(path='./images/test/city')
    earthpaths = os.listdir(path='./images/test/earth')
    X_test, y_test = [],[]
    for path in citypaths:
        x = prep_for_vgg('./images/test/city/'+path)
        X_test.append(x.reshape(224,224,3))
        y_test.append(0)
    for path in earthpaths:
        x = prep_for_vgg('./images/test/earth/'+path)
        X_test.append(x.reshape(224,224,3))
        y_test.append(1)
    X_test, y_test = np.array(X_test), keras.utils.to_categorical(np.array(y_test))
    
    return X_test, y_test