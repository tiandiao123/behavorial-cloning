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
angles.columns = ('Center','Left','Right','Steering Angle','Throttle','Brake','Speed')
angles = np.array(angles['Steering Angle'])
recovery_angles = pd.read_csv('../Recovery/driving_log.csv', header = None)
recovery_angles.columns = ('Center','Left','Right','Steering Angle','Throttle','Brake','Speed')
recovery_angles = np.array(recovery_angles['Steering Angle'])
center_angles=angles
right_angles=angles-0.2
left_angles=angles+0.2
labels=np.concatenate((center_angles,right_angles,left_angles,recovery_angles),axis=0)
print(len(recovery_angles))



images = np.asarray(os.listdir("../IMG/"))
center_images = np.ndarray(shape=(len(center_angles), 32, 64, 3))
right_images = np.ndarray(shape=(len(right_angles), 32, 64, 3))
left_images = np.ndarray(shape=(len(left_angles), 32, 64, 3))

count1 = 0
count2 = 0
count3 = 0

#I referenced some codes from Forums to get data(since originally I use my own functions in process_image.py file but didn't work)
#The reason why my get_data function didn't work is that the log file's image files names don't fit with the image files names in the IMG File
for image in images:
    image_file = os.path.join('../IMG', image)
    if image.startswith('center'):
        image_data = ndimage.imread(image_file).astype(np.float32)
        center_images[count1] = imresize(image_data, (32,64,3))
        count1+=1
    elif image.startswith('right'):
        image_data = ndimage.imread(image_file).astype(np.float32)
        right_images[count2] = imresize(image_data, (32,64,3))
        count2+=1
    elif image.startswith('left'):
        image_data = ndimage.imread(image_file).astype(np.float32)
        left_images[count3] = imresize(image_data, (32,64,3))
        count3 += 1

plt.imshow(center_images[6])
plt.imshow(left_images[0])

recovery_images = np.asarray(os.listdir("../Recovery/IMG/"))
recovery = np.ndarray(shape=(len(recovery_angles), 32, 64, 3))
count = 0
for image in recovery_images:
    if image.startswith('center'):
        image_file = os.path.join('../Recovery/IMG', image)
        image_data = ndimage.imread(image_file).astype(np.float32)
        recovery[count] = imresize(image_data, (32,64,3))
        count += 1

data=np.concatenate((center_images,right_images,left_images,recovery),axis=0)

flip_images = np.ndarray(shape=(data.shape))
count = 0
for i in range(len(data)):
    flip_images[count] = np.fliplr(data[i])
    count += 1
print(flip_images.shape)

flip_images_angles = labels * -1
training_data = np.concatenate((data, flip_images), axis=0)
training_angles = np.concatenate((labels, flip_images_angles),axis=0)

X_train, X_validation, y_train, y_validation = train_test_split(training_data, training_angles, test_size=.1)

# Construct model architecture
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((10,0), (0,0)), input_shape=(32,64,3)))

model.add(BatchNormalization(axis=1))
model.add(Convolution2D(8,3,3,border_mode='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(8,3,3,border_mode='valid',activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(24, 3, 3, border_mode='valid', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(45, 2, 2, border_mode='valid', activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(45, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())

model.add(Dense(560))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

adam = Adam(lr=0.0001)
model.compile(loss='mse',optimizer='adam')

checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

history_object=model.fit(X_train,y_train,nb_epoch=20,verbose=1,batch_size=128,shuffle=True,validation_data=(X_validation, y_validation),callbacks=[checkpoint, callback])
json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(json, f)
print("Constructed Model has been saved")


#draw the pictures to see the loss and val_loss changes over training
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()