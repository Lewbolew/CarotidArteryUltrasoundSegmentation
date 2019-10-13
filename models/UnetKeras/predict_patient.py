import pickle
import os
import cv2 as cv
import numpy as np
from unet_with_batch_norm_multiclass import UNet
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
load_src("data_loader", "../../utils/data_loader.py")
from data_loader import data_loader, data_loader_softmax, load_image_from_folder

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#----------------------------INPUT PARAMETERS---------------------------


PATIENTS_FOLDER = '../../dataset/patients_with_procentage/'
WEIGHT_PATH = '../../weights/Unet/augmentation_experiments/weightsSoftmax/rgb_color_aug.hdf5'
PATH_TO_SAVE = '../../predictions/patients/'
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

model = UNet((IMG_HEIGHT,IMG_WIDTH, CHANNELS_NUM), start_ch=32, depth=MODEL_DEPTH, 
             batch_norm=BATCH_NORM, dropout=DROUPOUT, activation=MAIN_ACTIVATION_FUNCION)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
model.load_weights(WEIGHT_PATH)

patients = os.listdir(PATIENTS_FOLDER)
patients = patients[81:]
for i in range(len(patients)):

    X, y = load_image_from_folder(PATIENTS_FOLDER+patients[i])


    predictions = model.predict(X)

    # Predict test set
    predictions = model.predict(X)

    # Create dictionary to save the results
    dict_with_prediction_data = dict()
    dict_with_prediction_data['name_of_the_dataset'] = patients[i]
    dict_with_prediction_data['predictions'] = predictions
    dict_with_prediction_data['x'] = X
    # dict_with_prediction_data['y'] = None
    # dict_with_prediction_data['history_of_training'] = None

    # Save the results
    save(dict_with_prediction_data, PATH_TO_SAVE+patients[i]+'.dat')
