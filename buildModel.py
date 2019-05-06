import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from read_bosphorus import readBosphorus


def getModel():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(96, 96, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", kernel_initializer="he_normal", input_shape=(96, 96, 1), data_format="channels_last"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(36, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(48, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(90, activation="relu"))
    model.add(Dense(30))

    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    return model


def getDataset():
    datasetlocation = "./datasets/bosphorus/data/"
    index = readBosphorus.makeIndex(datasetlocation)
    X = []
    for elem in index:
        x, y = readBosphorus.getFeatureVector(elem)
        X.append(x)

    X = np.asarray(X)
    print(X.shape)

def main():
    model = getModel()
    dataset = getDataset()


if __name__ == "__main__":
    main()