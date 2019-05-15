import random

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, GlobalAveragePooling2D, MaxPooling2D)
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

import cv2 as cv
from sources import readBosphorus
from constants import FACIAL_LANDMARKS


# calculate the average distance between the true points and the predicted points
def metric_avg_distance(y_true, y_pred):
    A=K.reshape(y_true, (22, 2, -1))
    B=K.reshape(y_pred, (22, 2, -1))
    Z=A-B
    z=K.abs(K.sqrt(K.sum(K.square(Z),axis=-1,keepdims=False)))
    avg=K.mean(z, axis=None, keepdims=False)
    return avg


def getModel():
    model = Sequential()
    #model.add(BatchNormalization(input_shape=(96, 96, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", kernel_initializer="he_normal",
                     input_shape=(2, 128, 128), data_format="channels_first"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(36, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(48, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    #model.add(Conv2D(64, (3, 3)))
    # model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    #model.add(Conv2D(64, (3, 3)))
    # model.add(Activation("relu"))
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(90, activation="relu"))
    model.add(Dense(44))

    model.compile(optimizer="rmsprop", loss="mse", metrics=[metric_avg_distance])

    print(model.summary())
    return model


def getDataset():
    datasetlocation = "./datasets/bosphorus"
    try:
        X = np.load(datasetlocation+"_X.npy")
        y = np.load(datasetlocation+"_y.npy")
        return X, y
    except IOError:
        pass

    print("Creating dataset...")
    index = readBosphorus.makeIndex()
    X = []
    y = []
    for i, elem in enumerate(index):
        if i % 10 == 0:
            print("{:d}/{:d}".format(i, len(index)))
        x, y_ = getFeatureVector(elem)
        X.append(x)
        y.append(y_)

    X = np.asarray(X)
    y = np.asarray(y)

    np.save(datasetlocation+"_X.npy", X)
    np.save(datasetlocation+"_y.npy", y)

    return X, y


def getFeatureVector(id):
    nrows, ncols, zmin, imfile, data = readBosphorus.readBNTFile(id+".bnt")
    points, labels = readBosphorus.readLM2File(id+".lm2")
    image = cv.imread(id+".png")
    depth = data[:, 2]
    depth = depth.reshape((nrows, ncols))
    depth = np.flip(depth, 0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    im = cv.resize(gray, (128, 128))
    im = np.asarray(im, dtype="float")
    im /= 255.0

    depth[depth == zmin] = np.nan
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth = np.nan_to_num(depth)

    dep = cv.resize(depth, (128, 128))

    x = [im, dep]
    x = np.asarray(x, dtype='float')

    y = []

    for l in FACIAL_LANDMARKS:
        pos = labels.index(l)
        p = points[pos]
        p[0] = p[0] / gray.shape[1]
        p[1] = p[1] / gray.shape[0]
        y.append(p)

    y = np.asarray(y, dtype='float')
    y = y.flatten()

    return x, y


def visualize(x, y, plot_landmarks=True, annotate_landmarks=True):
    # plot the RGB image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(x[0, :, :], cmap='gray')
    y *= 128
    y = y.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        plt.scatter(y[:, 0], y[:, 1], s=20, c="red", alpha=1.0, edgecolor='black')

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y[i, 0], y[i, 1]), color="white", fontsize="small")

    plt.subplot(1, 2, 2)
    plt.title("Depth")
    plt.imshow(x[1, :, :], cmap='gray')
    plt.show()


def main():
    model = getModel()
    X, y = getDataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    tensorboard = TensorBoard(update_freq='batch')

    model.fit(X_train, y_train, verbose=True, validation_data=(X_test, y_test), epochs=4, callbacks=[tensorboard])
    model.save("network.hdf5")
    #model.load_weights("./network.hdf5")

    while True:
        i = random.randint(0, X_test.shape[0]-1)
        y_pred = model.predict(np.asarray([X_test[i]]))
        visualize(X_test[i], y_pred)
        visualize(X_test[i], y_test[i])


if __name__ == "__main__":
    main()
