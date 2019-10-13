from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose

def conv_block(m, dim, activ_func, batch_norm, res, dropout=0):
	n = Conv2D(dim, 3, activation=activ_func, padding='same') (m)
	n = BatchNormalization()(n) if batch_norm else n
	n = Dropout(dropout)(n) if dropout else n
	n = Conv2D(dim, 3, activation=activ_func, padding='same')(n)
	n = BatchNormalization()(n) if batch_norm else n
	return Concatenate()([m,n]) if res else n

def level_block(m, dim, depth, inc, activ_func, dropout, batch_norm, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, activ_func, batch_norm, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, activ_func, dropout, batch_norm, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=activ_func, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=activ_func, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, activ_func, batch_norm, res)
	else:
		m = conv_block(m, dim, activ_func, batch_norm, res, dropout)
	return m

def UNet(img_shape, out_ch=3, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batch_norm=True, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batch_norm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)


# import pickle
# import os
# import cv2 as cv
# import numpy as np

# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# from unet_with_batch_norm import UNet

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# #----------------------------INPUT PARAMETERS---------------------------
# WEIGHT_PATH = '../../weights/Unet'
# DATA_PATH = '../../dataset/augmented_data'

# # input image properties
# IMG_HEIGHT = 446
# IMG_WIDTH = 446
# CHANNELS_NUM = 3

# # model parameters
# MODEL_DEPTH = 5
# BATCH_NORM = True
# DROUPOUT = 0.5
# MAIN_ACTIVATION_FUNCION = 'relu'

# #training
# NUM_EPOCHS = 3
# BATCH_SIZE = 128

# #-------------------------END_OF_INPUTING_PARAMETERS--------------------



# # load images
# def data_loader(data_path, is_test=False):
#     """
#     Return loaded images. I didn`t do the DataGenerator as we had quite a few images.
#     """
#     if is_test:
#         # X_test and y_test loading
#         test_name_images = os.listdir(os.path.join(data_path, 'test', 'x'))
#         X_test = []
#         y_test = []
#         for i in range(len(test_name_images)):
#             X_test.append(cv.resize(cv.imread(os.path.join(data_path, 'test', 'x', test_name_images[i])), (446,446)))
#             y_test.append(cv.resize(cv.imread(os.path.join(data_path, 'test', 'y', test_name_images[i])),(446,446)))
#         return np.asarray(X_test), np.asarray(y_test)
#     else:
#         # X_train and y_train loading
#         train_name_images = os.listdir(os.path.join(data_path, 'train', 'x'))
#         X_train = []
#         y_train = []
#         for i in range(len(train_name_images)):
#             X_train.append(cv.resize(cv.imread(os.path.join(data_path, 'train', 'x', train_name_images[i])),(446,446)))
#             y_train.append(cv.resize(cv.imread(os.path.join(data_path, 'train', 'y', train_name_images[i])),(446,446)))
#         return np.asarray(X_train), np.asarray(y_train)

# X, y = data_loader(DATA_PATH)
# X_train, y_train, X_val, y_val = X[10:], y[10:], X[:10], y[:10]
# # create model
# model = UNet((IMG_HEIGHT,IMG_WIDTH, CHANNELS_NUM), start_ch=32, depth=MODEL_DEPTH, 
# 			 batch_norm=BATCH_NORM, dropout=DROUPOUT, activation=MAIN_ACTIVATION_FUNCION)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()

# # create model callbacks
# early_stopping = EarlyStopping(patience=10,verbose=1)
# model_checkpoint = ModelCheckpoint(WEIGHT_PATH)
# reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# history = model.fit(X_train, y_train,
# 					validation_data=[X_val,y_val],
# 					epochs=NUM_EPOCHS,
# 					batch_size=BATCH_SIZE,
# 					callbacks=[early_stopping,model_checkpoint,reduce_lr]
# 					)


# # # save history of the training process
# # with open('/trainHistoryDict', 'wb') as file_pi:
# #         pickle.dump(history.history, file_pi)





