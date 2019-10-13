from keras import backend as K
import cv2 as cv
import numpy as np
import os

def draw_cropped_img(img):
    """
    Draw the borders where the image will be cropped. 
    """
    plt.figure(figsize=(20,10))
    img[103:104,:,0] = 255
    img[:,192:193,0] = 255
    img[:,-65:-64,0] = 255
    img[-10:-9,:,0] = 255
    plt.imshow(img)
    plt.axis('Off')

def preprocess_image(inp_img):
    """
    Crop inputed img + make redundant pixels darker
    Return:
     - img of size 544x544
    """
    img = np.copy(inp_img)
    img = img[12:,192:-64, :]
    img[:105,:,:] = 0
    img[-10:,:,:] = 0
    return img.astype(np.uint8)

def preprocess_images(list_of_images):
    """
    Preprocess batch of images
    """
    return [preprocess_image(img) for img in list_of_images]

def preprocess_mask(mask):
    """
    Crop the mask + set all values into the set of values [0, 1]
    """
    new_mask = np.copy(mask)
    new_mask = new_mask[12:,192:-64, :]
    new_mask[new_mask!=0] = 1
    new_mask = new_mask.astype(np.uint8)
    return new_mask

def preprocess_masks(list_of_masks):
    return [preprocess_mask(mask) for mask in list_of_masks]

def save_images(imgs, folder_to_save, filenames):
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    for i in range(len(imgs)):
        cv.imwrite(os.path.join(folder_to_save, filenames[i]), imgs[i])

if __name__ == '__main__':
	# input parameters
	PATH_TO_X = '../dataset/data/x_test/'
	PATH_TO_Y = '../dataset/data/y_test/'
	PATH_TO_SAVE_X = '../dataset/preprocessed_data/test_x/'
	PATH_TO_SAVE_Y = '../dataset/preprocessed_data/test_y/'
	img_names = os.listdir(PATH_TO_X)

	# read images
	imgs = [cv.imread(PATH_TO_X+'/'+img_names[i]) for i in range (len(img_names))]
	masks = [cv.imread(PATH_TO_Y+'/'+img_names[i]) for i in range (len(img_names))]

	# do the cropping and risizing
	imgs = preprocess_images(imgs)
	masks = preprocess_masks(masks)

	# save images
	save_images(imgs, PATH_TO_SAVE_X, img_names)
	save_images(masks, PATH_TO_SAVE_Y, img_names)
