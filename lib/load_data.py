# Method to load data; both images and their masks
import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Get and resize train images and masks
def get_data(ids, path, im_height, im_width, train):
    
    #ids = sorted(next(os.walk(path + "/images"))[2])
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    print(X.shape)

    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
        
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        id_ = id_.split('/')[-1]
        # Load images
        img = cv2.imread(path + '/images/'+ '{}'.format(id_), cv2.IMREAD_COLOR)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = cv2.imread(path + '/masks/' + '{}'.format(id_.split('.')[0]+'.png'), cv2.IMREAD_GRAYSCALE)
            mask = img_to_array(mask)
            mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        #print(X[n, ..., 0].shape)
        #print(y[n].shape)
        X[n] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        print(X.shape)
        return X, y
    else:
        return X
