import os
import cv2 as cv
import numpy as np

def data_loader(data_path, is_test=False):
    """
    Return loaded images. I didn`t do the DataGenerator as we had quite a few images.
    """
    if is_test:
        folder_for_x = 'x_test'
        folder_for_y = 'y_test'
    else:
        folder_for_x = 'train_x'
        folder_for_y = 'train_y'

    images_names = os.listdir(os.path.join(data_path, folder_for_x))
    X = []
    y = []
    for i in range(len(images_names)):
        X.append(cv.imread(os.path.join(data_path, folder_for_x, images_names[i])))
        y.append(cv.imread(os.path.join(data_path, folder_for_y, images_names[i])))
    return np.asarray(X), np.asarray(y)


def data_loader_softmax(data_path, is_test=False, is_val=False):
    if is_test:
        folder_for_x = 'x_test'
        folder_for_y = 'y_test'
    elif is_val:
        folder_for_x = 'x_val'
        folder_for_y = 'y_val'
    else:
        folder_for_x = 'train_x'
        folder_for_y = 'train_y'


    images_names = os.listdir(os.path.join(data_path, folder_for_x))
    X=[]
    y=[]

    for i in range(len(images_names)):
        X.append(cv.imread(os.path.join(data_path, folder_for_x, images_names[i])))
        y.append(np.load(os.path.join(data_path, folder_for_y, images_names[i][:-4]+'.npy')))
    return np.asarray(X), np.asarray(y)

def preprocess_image(inp_img):
    """
    Crop inputed img + make redundant pixels darker
    Return:
     - img of size 544x544
    """
    img = np.copy(inp_img)
    img = img[12:,192:-64, :] # crop
    img[:105,:,:] = 0 # make pixel area dark
    img[-10:,:,:] = 0 # make pixel area dark
    
    return img.astype(np.uint8)

def preprocess_images(list_of_images):
    """
    Preprocess batch of images
    """
    return [preprocess_image(img) for img in list_of_images]

def load_image_from_folder(folder_path):
    images_names = os.listdir(os.path.join(folder_path))
    imgs = []

    for i in range(len(images_names)):
        imgs.append(cv.imread(os.path.join(folder_path, images_names[i])))
    imgs = preprocess_images(imgs)
    
    return np.asarray(imgs), None