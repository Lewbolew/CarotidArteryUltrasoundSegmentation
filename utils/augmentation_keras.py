import numpy as np
import os
import pandas as pd
import cv2 as cv

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from data_preprocessing import save_images


# INPUT PARAMETERS
PATH_TO_X = '../dataset/preprocessed_data/train_x/'
PATH_TO_Y = '../dataset/preprocessed_data/train_y/'
PATH_TO_SAVE_AUGMENTED_X = '../dataset/augmented_data/train_x/'
PATH_TO_SAVE_AUGMENTED_Y = '../dataset/augmented_data/train_y/'

names_list = ['png_13_5691.png', 'png_6_3078.png', 'png_2_8486.png', 'png_10_9933.png', 'png_0_4895.png', 'png_9_1892.png', 
'png_10_4632.png', 'png_16_5198.png', 'png_4_9518.png', 'png_10_4380.png', 'png_4_1538.png', 'png_13_3789.png',
 'png_14_2384.png', 'png_10_5556.png', 'png_4_8124.png', 'png_15_7559.png', 'png_8_1219.png', 'png_0_5429.png', 
 'png_3_3446.png', 'png_16_7057.png', 'png_0_212.png', 'png_16_3718.png', 'png_9_4593.png', 'png_3_4655.png', 
 'png_7_1794.png', 'png_11_3976.png', 'png_8_9144.png', 'png_14_2099.png', 'x5682.png', 'x3602.png', 'png_7_3496.png',
  'png_9_6470.png', 'png_9_7254.png', 'png_9_9366.png', 'png_12_6446.png', 'png_9_3459.png', 'png_13_3368.png', 
  'png_11_238.png', 'png_5_9330.png', 'png_12_7110.png', 'png_1_2594.png', 'png_12_7282.png', 'png_9_4624.png', 
  'png_8_8868.png', 'png_9_3375.png', 'png_14_9565.png', 'png_9_4115.png', 'png_3_7976.png', 'x391.png', 'x909.png',
   'png_0_9482.png', 'png_9_8001.png', 'png_1_158.png', 'png_7_1864.png', 'png_15_8794.png', 'png_7_1765.png', 
   'png_4_5120.png', 'png_1_8538.png', 'png_16_7786.png', 'png_4_2969.png', 'png_2_9237.png', 'png_2_6854.png',
    'png_4_5960.png', 'png_1_8772.png', 'png_5_6637.png', 'x260.png', 'png_16_273.png', 'png_11_5672.png', 
    'png_7_6359.png', 'png_16_3497.png', 'png_5_5453.png', 'png_6_8112.png', 'png_5_4085.png', 'png_15_1504.png', 
    'png_6_2592.png', 'x61.png', 'png_11_5984.png', 'png_15_878.png', 'png_5_8836.png', 'png_14_927.png', 'x3060.png', 
    'png_2_8672.png', 'png_15_1383.png', 'png_9_9677.png', 'png_2_9804.png', 'png_6_981.png', 'png_8_854.png', 'x470.png',
     'png_15_1094.png', 'png_14_6121.png', 'png_12_4130.png', 'png_8_7229.png', 'png_1_274.png', 'png_13_5590.png',
      'png_1_3014.png', 'png_8_7840.png', 'png_14_8889.png', 'png_13_5677.png', 'png_7_6309.png', 'png_11_3912.png',
       'png_4_1942.png', 'png_11_9657.png', 'png_3_6872.png', 'png_16_8546.png', 'png_1_2307.png', 'png_10_6677.png',
        'png_16_3645.png', 'png_12_2679.png', 'png_4_3384.png', 'png_5_4755.png', 'png_13_8712.png', 'png_13_4180.png', 
        'png_1_3069.png', 'png_6_2401.png', 'png_16_6122.png', 'png_16_6879.png', 'png_12_1339.png', 'png_16_9126.png',
         'x25.png', 'x5536.png', 'png_13_2805.png', 'png_5_5678.png', 'png_12_3950.png', 'x885.png', 'png_2_9352.png', 
         'png_7_2628.png', 'png_10_8564.png', 'png_13_346.png', 'png_11_1438.png', 'png_5_1462.png', 'png_0_1934.png',
          'png_0_499.png', 'png_3_3063.png', 'png_10_5831.png', 'png_5_7888.png', 'png_12_3615.png', 'png_6_5226.png', 
          'png_2_4726.png', 'png_7_8670.png', 'png_5_6349.png', 'png_14_7930.png', 'png_14_3709.png', 'png_13_8861.png',
           'png_9_9391.png', 'png_6_4555.png', 'png_1_794.png', 'png_6_681.png', 'png_14_8598.png', 'png_0_2832.png',
            'x639.png', 'png_0_9759.png', 'png_11_1877.png', 'x546.png', 'png_6_2438.png', 'png_1_7662.png', 
            'png_4_2274.png', 'png_2_789.png', 'png_8_6008.png', 'png_14_620.png', 'png_1_8192.png',
             'png_3_1248.png', 'png_14_6721.png', 'png_4_8853.png', 'png_10_6264.png', 'png_13_1629.png',
              'png_2_5365.png', 'png_3_2788.png', 'x713.png', 'png_3_5765.png', 'png_11_9154.png',
               'png_13_5127.png', 'png_0_1805.png', 'png_5_7716.png', 'png_7_4904.png', 'png_7_311.png', 
            'png_6_2273.png', 'png_8_6515.png', 'png_12_3274.png', 'png_5_9568.png', 'png_6_2901.png',
     'png_0_5112.png', 'png_12_7125.png', 'png_7_4063.png', 'png_10_7417.png', 'png_0_4428.png', 'png_14_4711.png', 
'png_15_2961.png', 'png_12_9008.png', 'png_8_1408.png', 'png_15_4479.png', 'png_15_2597.png', 'png_4_5578.png', 
'png_16_5180.png', 'png_7_2452.png', 'png_8_597.png', 'png_1_5241.png', 'png_2_2417.png', 
'png_0_9533.png', 'png_15_6951.png',
 'png_3_4366.png', 'png_8_5338.png', 'x2628.png', 'x359.png', 'png_11_6361.png', 'x3976.png', 'png_10_7815.png', 
 'png_6_9797.png', 'png_15_1558.png', 'png_10_8419.png', 'png_2_3650.png', 'png_12_5907.png', 'png_10_5700.png',
  'png_15_2780.png', 'png_11_3315.png', 'png_3_9649.png', 'png_3_4086.png', 'png_8_9993.png', 'png_11_9180.png']


NUMBER_OF_AUGMENTATIONS = 200

# data reading
X_names = os.listdir(PATH_TO_X)
y_names = os.listdir(PATH_TO_Y) # equal to X_names in the case of my dataset

X_train = np.asarray([cv.imread(os.path.join(PATH_TO_X, img_name)) for img_name in X_names])
y_train = np.asarray([cv.imread(os.path.join(PATH_TO_Y, img_name)) for img_name in y_names])

# augmentation process
aug_imgs = []
aug_masks = []
# Here I used trick a bit. To use Kerases Generators for Augmentation in Segmentation problem(to make the same 
# transformation with mask as with image) you should use batch_size=1 and equall seed arguments to the flow 
# method of DataGenerator

# augment images
datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True)
datagen.fit(X_train)
i = 0
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=1,seed=1337
                                     # save_to_dir=PATH_TO_SAVE_AUGMENTED_X,
                                     # save_prefix='png'
                                     ):
    aug_imgs.append(X_batch[0].astype(np.uint8))
    i+=1
    if i > NUMBER_OF_AUGMENTATIONS:
        break

# augment masks
datagen = ImageDataGenerator(rotation_range=360, horizontal_flip=True, vertical_flip=True)
datagen.fit(y_train)
i = 0
for X_batch, y_batch in datagen.flow(y_train, X_train, batch_size=1,seed=1337
                                     # save_to_dir=PATH_TO_SAVE_AUGMENTED_Y,
                                     # save_prefix='png'
                                    ):
    aug_masks.append(X_batch[0].astype(np.uint8))
    i+=1
    if i > NUMBER_OF_AUGMENTATIONS:
        break

# saving the results
# save_images(X_train, PATH_TO_SAVE_AUGMENTED_X, X_names)
# save_images(y_train, PATH_TO_SAVE_AUGMENTED_Y, y_names)
save_images(aug_imgs, PATH_TO_SAVE_AUGMENTED_X, names_list)
save_images(aug_masks, PATH_TO_SAVE_AUGMENTED_Y, names_list)
