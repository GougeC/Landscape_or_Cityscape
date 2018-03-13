#general imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras

from sklearn.metrics import f1_score
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

def preprocess_IV3(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def prep_for_model(path, model):
    img = image.load_img(path, target_size = (224, 224) )
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    if model == 'vgg16':
        x = vgg16.preprocess_input(x, mode='tf')
    else:
        x = preprocess_IV3(x)
    return x

def get_test(suffix,model):
    '''Creates a test set from a folder I set aside to test results after modeling
model names are 'vgg16', 'inception_v3' and 'resnet50'
    
    '''
    
    earthpaths = os.listdir(path='./images/test/earthporn'+suffix)
    citypaths = os.listdir(path='./images/test/cityporn'+suffix)
    X_test, y_test = [],[]
    for path in citypaths:
        x = prep_for_model('images/test/cityporn'+suffix+'/'+path,model)
        X_test.append(x.reshape(224,224,3))
        y_test.append(0)
    for path in earthpaths:
        x = prep_for_model('images/test/earthporn'+suffix+'/'+path,model)
        X_test.append(x.reshape(224,224,3))
        y_test.append(1)
    X_test, y_test = np.array(X_test), keras.utils.to_categorical(np.array(y_test))
    
    return X_test, y_test

def create_confusion_matrix(y_true,predictions,sub_mapping):
    '''
    returns a pandas matrix of the confusion matrix for the parameters given
    '''
    prediction_classes = np.argmax(predictions,axis = 1)
    true_classes = np.argmax(y_true,axis = 1)
    num_classes = len(y_true[0])
    con_matrix = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            val = np.sum((prediction_classes == i) & (true_classes ==j))
            con_matrix[i,j] = val
    index = [sub_mapping[x] for x in range(num_classes)]
    con_table = pd.DataFrame(con_matrix,index = index,columns = index)
    row_summations = con_table.T.sum()
    print("columns are actual, rows are predicted")
    con_table['totals'] = row_summations
    col_sums = con_table.sum()
    con_table =con_table.T
    con_table['totals'] = col_sums
    print("f1 score, macro: ",f1_score(true_classes,prediction_classes,average = 'macro'))
    print("f1 score, weighted: ",f1_score(true_classes,prediction_classes,average='weighted'))
    print("f1 score, micro: ",f1_score(true_classes,prediction_classes,average = 'micro'))
    return con_table.T
