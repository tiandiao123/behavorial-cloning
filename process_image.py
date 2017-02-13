import pandas as pd
import os
import json
from skimage.exposure import adjust_gamma
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy import ndimage
from scipy.misc import imresize



import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def process_images(image):
    imagearray=mpimg.imread(image)
    return imagearray

#it turns out that we cannot process data using get_data functions even though I feel it is correct implementation for geting image data
#The reason is that the driving_log.csv contains image file names that don't exist in the IMG fold.
#Thus, I changed the get_data into precess_data functions
def get_data(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            directory = "../IMG"
            img_center = process_image(directory + row[0])
            img_left = process_image(directory + row[1])
            img_right = process_image(directory + row[2])

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)

def process_data(csv_file, angles):
    images = np.asarray(os.listdir(csv_file))
    center = np.ndarray(shape=(len(angles), 32, 64, 3))
    right = np.ndarray(shape=(len(angles), 32, 64, 3))
    left = np.ndarray(shape=(len(angles), 32, 64, 3))
    count = 0
    for image in images:
        image_file = os.path.join('../IMG', image)
        if image.startswith('center'):
            image_data = ndimage.imread(image_file).astype(np.float32)
            center[count % len(angles)] = imresize(image_data, (32,64,3))#[12:,:,:]
        elif image.startswith('right'):
            image_data = ndimage.imread(image_file).astype(np.float32)
            right[count % len(angles)] = imresize(image_data, (32,64,3))#[12:,:,:]
        elif image.startswith('left'):
            image_data = ndimage.imread(image_file).astype(np.float32)
            left[count % len(angles)] = imresize(image_data, (32,64,3))#[12:,:,:]
            count += 1
        return (center,right,left)
