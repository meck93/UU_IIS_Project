import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
import glob
import cv2

#CNN architecture
model = Sequential()
model.add(BatchNormalization(input_shape=(640,480, 1)))
model.add(Convolution2D(24, 5, 5, border_mode="same", init="he_normal", input_shape=(640,480, 1), dim_ordering="tf"))
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

model.load_weights("face_model2.h5")

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

#image to predict on
X_data = []
image = cv2.imread("Obama2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
X_data.append(gray)
print(np.array(X_data).shape)
X_pred = np.array(X_data).reshape(1,640,480,1)

features = model.predict(X_pred, batch_size=1)
print(features)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#display features
cv2.namedWindow( "image", cv2.WINDOW_AUTOSIZE );
for f in features:
	for i in range(f.size-1):
		for (x,y,w,h) in faces:
			cv2.circle(image,(f[i],f[i+1]), 2, (0,0,255), -1)
cv2.imshow( "image", image);
cv2.waitKey(0)
cv2.destroyAllWindows()