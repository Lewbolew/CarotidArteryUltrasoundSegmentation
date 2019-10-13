import pickle
import os
import cv2 as cv
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from unet_with_batch_norm import UNet

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
load_src("losses_for_segmentation", "../../utils/losses_for_segmentation.py")
load_src("data_loader", "../../utils/data_loader.py")

from losses_for_segmentation import jaccard_distance, dice_coef_loss, dice_coef
from data_loader import data_loader

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#----------------------------INPUT PARAMETERS---------------------------
WEIGHT_PATH = '../../weights/Unet/full_augmentation_data/unet_on_lumen/'
DATA_PATH = '../../dataset/augmentation/lumen_augmentation/'

# input image properties
IMG_HEIGHT = 544
IMG_WIDTH = 544
CHANNELS_NUM = 3

# model parameters
MODEL_DEPTH = 5
BATCH_NORM = True
DROUPOUT = 0.5
MAIN_ACTIVATION_FUNCION = 'relu'

#training
NUM_EPOCHS = 100
BATCH_SIZE = 4

#-------------------------END_OF_INPUTING_PARAMETERS--------------------

# load images
X, y = data_loader(DATA_PATH)
X_train, y_train, X_val, y_val = X[5:], y[5:], X[:5], y[:5]

# create model
model = UNet((IMG_HEIGHT,IMG_WIDTH, CHANNELS_NUM), start_ch=32, depth=MODEL_DEPTH, 
             batch_norm=BATCH_NORM, dropout=DROUPOUT, activation=MAIN_ACTIVATION_FUNCION)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy", dice_coef])
model.summary()

# create model callbacks
early_stopping = EarlyStopping(patience=10,verbose=1)
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = "best_weights_lumen.hdf5"
model_checkpoint = ModelCheckpoint(os.path.join(WEIGHT_PATH,filepath))
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# train prcess

history = model.fit(X_train, y_train,
                    validation_data=[X_val,y_val],
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping,model_checkpoint,reduce_lr]
                    )


# save history of the training process
def save(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)

save(history, 'lumen_history')


