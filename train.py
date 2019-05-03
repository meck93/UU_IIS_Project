import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
import glob
import cv2


#CNN architecture
model = Sequential()
model.add(BatchNormalization(input_shape=(480, 640, 1)))
model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(480, 640, 1), dim_ordering="tf"))
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
model.add(Dense(300, activation="relu"))
model.add(Dense(152))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="face_model2.h5", verbose=1, save_best_only=True)
epochs = 30

#data retrieval
X_data = []
files = glob.glob("E:/IIS/Project/UU_IIS_Project/dataset/muct/jpg/*.jpg")
for file in files:
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X_data.append(gray)
X_train = np.array(X_data).reshape(3755,480,640,1)
print(X_train.shape)

#label retrival
with open("E:/IIS/Project/UU_IIS_Project/dataset/muct/muct-landmarks/muct76-opencv.csv") as f:
    lines = f.readlines()
split_lines = []
for line in lines:
    split_lines.append(line.replace("\n", "").split(','))
landmarks= []
for line in split_lines[1:]:
    landmarks.append(line[2:])
converted_landmarks = []
for data in landmarks:
    converted_landmarks.append([float(x) for x in data])
y_train = np.array(converted_landmarks[0:3755])
print(y_train.shape)

#train
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)