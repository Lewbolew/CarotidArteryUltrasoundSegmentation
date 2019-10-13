import pickle
import os
import cv2 as cv
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from unet_with_batch_norm_multiclass import UNet
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
load_src("data_loader", "../../utils/data_loader.py")
from data_loader import data_loader_softmax

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="5"

#----------------------------INPUT PARAMETERS---------------------------

DATASETS_PATH = '../../dataset/augmentation/softmax/'
WEIGHTS_PATH = '../../weights/Unet/augmentation_experiments/weightsSoftmax/'
HISTORIES_PATH = '../../weights/Unet/augmentation_experiments/history_of_trainingSoftmax/'
PATH_TO_SAVE = '../../predictions/augmentation_experiments_softmax/'

# input image properties
IMG_HEIGHT = 544
IMG_WIDTH = 544
CHANNELS_NUM = 3

# model parameters
MODEL_DEPTH = 5
BATCH_NORM = True
DROUPOUT = 0.5
MAIN_ACTIVATION_FUNCION = 'relu'

#-------------------------END_OF_INPUTING_PARAMETERS--------------------
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)



datasets_list = os.listdir(DATASETS_PATH)
# Iterate over each dataset
for i in range(len(datasets_list)):

    # Loading dataset
    X, y = data_loader_softmax(os.path.join(DATASETS_PATH, datasets_list[i]), is_test=True)

    # Creating model with proper weights
    model = UNet((IMG_HEIGHT,IMG_WIDTH, CHANNELS_NUM), start_ch=32, depth=MODEL_DEPTH, 
                 batch_norm=BATCH_NORM, dropout=DROUPOUT, activation=MAIN_ACTIVATION_FUNCION)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.load_weights(os.path.join(WEIGHTS_PATH, datasets_list[i])+'.hdf5')
    # Predict test set
    predictions = model.predict(X)

    # Create dictionary to save the results
    dict_with_prediction_data = dict()
    dict_with_prediction_data['name_of_the_dataset'] = datasets_list[i]
    dict_with_prediction_data['predictions'] = predictions
    dict_with_prediction_data['x'] = X
    dict_with_prediction_data['y'] = y


    if datasets_list[i] in ''.join(os.listdir(HISTORIES_PATH)):
        dict_with_prediction_data['history_of_training'] =  pickle.load(open(HISTORIES_PATH+datasets_list[i]+'.dat', "rb"))
    else:
        dict_with_prediction_data['history_of_training'] = None

    # Save the results
    
    save(dict_with_prediction_data, PATH_TO_SAVE+'{}.dat'.format(datasets_list[i]))
