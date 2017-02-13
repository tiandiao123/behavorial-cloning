import numpy as np
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


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
%matplotlib inline

angles = pd.read_csv('../driving_log.csv',header=None)
angles.columns = ('Center Image','Left Image','Right Image','Steering Angle','Throttle','Brake','Speed')
angles = np.array(angles['Steering Angle'])

recovery_angles = pd.read_csv('../Recovery/driving_log.csv', header = None)
recovery_angles.columns = ('Center Image','Left Image','Right Image','Steering Angle','Throttle','Brake','Speed')
recovery_angles = np.array(recovery_angles['Steering Angle'])


images = np.asarray(os.listdir("../IMG/"))
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



recovery_images = np.asarray(os.listdir("../Recovery/IMG/"))
recovery = np.ndarray(shape=(len(recovery_images), 32, 64, 3))

# Populate recovery driving dataset
count = 0
for image in recovery_images:
    image_file = os.path.join('../Recovery/IMG', image)
    image_data = ndimage.imread(image_file).astype(np.float32)
    recovery[count] = imresize(image_data, (32,64,3))
    count += 1

X_train = np.concatenate((center, right, left, recovery[0:1012]), axis=0)
y_train = np.concatenate((angles, (angles - .08), (angles + .08), recovery_angles[0:1012]),axis=0)


mirror = np.ndarray(shape=(X_train.shape))
count = 0
for i in range(len(X_train)):
    mirror[count] = np.fliplr(X_train[i])
    count += 1


mirror_angles = y_train * -1
X_train = np.concatenate((X_train, mirror), axis=0)
y_train = np.concatenate((y_train, mirror_angles),axis=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.05)


# Establish model architecture
model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(32,64,3)))
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()


adam = Adam(lr=0.0001)
model.compile(loss='mse',optimizer=adam)

checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)


model.fit(X_train,
        y_train,
        nb_epoch=20,
        verbose=1,
        batch_size=128,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, callback])

#print("Weights Saved")
json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
print("Model Saved")






