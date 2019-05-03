import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from numpy import load
import glob
import cv2

#CNN architecture
model = Sequential()
model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(96, 96, 1), dim_ordering="tf"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Convolution2D(36, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Convolution2D(48, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(GlobalAveragePooling2D());
model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(30))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="face_model_kaggle.h5", verbose=1, save_best_only=True)
epochs = 30

data = load('face_images.npz')
lst = data.files

'''
i = 0
for item in lst:
	for i in range(data[item].shape[2]):
		cv2.imwrite( 'dataset2/' + str(i) +'zz.png',data[item][:,:,i])
'''

#data retrival
X_train = []
for item in lst:
	X_train = data[item][:,:,0:2281]
X_train = np.rollaxis(X_train, 2).reshape(2281,96,96,1)
print(X_train.shape)

#label retrival
with open("E:/IIS/Project/UU_IIS_Project/facial_keypoints.csv") as f:
    lines = f.readlines()
split_lines = []
for line in lines:
    split_lines.append(line.replace("\n", "").split(','))
landmarks= []
for line in split_lines[1:]:
    landmarks.append(line)
converted_landmarks = []
for data in landmarks[0:2281]:
	converted_landmarks.append([float(x) if x!='' else 0 for x in data])
y_train = np.array(converted_landmarks)
print(y_train.shape)

#train
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)