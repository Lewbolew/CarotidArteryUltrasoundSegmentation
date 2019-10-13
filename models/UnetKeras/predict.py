import pickle
import os
import cv2 as cv
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from unet_with_batch_norm import UNet
def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
load_src("data_loader", "../../utils/data_loader.py")
from data_loader import data_loader

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#----------------------------INPUT PARAMETERS---------------------------

DATA_PATH = '../../dataset/augmentation/rgb_all_aug/'
WEIGHT_PATH = '../../weights/Unet/full_augmentation_data/augmentation_experiments/rgb/all-augmentation-32-0.99.hdf5'
PATH_TO_SAVE = '../../predictions/colored_masks/'

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



test_name_images = os.listdir(os.path.join(DATA_PATH, 'x_test'))
X_test = []
for i in range(len(test_name_images)):
    X_test.append(cv.imread(os.path.join(DATA_PATH, 'x_test', test_name_images[i])))
X_test = np.asarray(X_test)

# X, y = data_loader(DATA_PATH, is_test=True)


model = UNet((IMG_HEIGHT,IMG_WIDTH, CHANNELS_NUM), start_ch=32, depth=MODEL_DEPTH, 
             batch_norm=BATCH_NORM, dropout=DROUPOUT, activation=MAIN_ACTIVATION_FUNCION)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
model.summary()

model.load_weights(WEIGHT_PATH)

predictions = model.predict(X_test)
np.save(PATH_TO_SAVE+'predictions.npy', predictions)
# np.save(PATH_TO_SAVE+'corresponding_x.npy', X)
# np.save(PATH_TO_SAVE+'masks.npy', y)
