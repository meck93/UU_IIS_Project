import os
import random

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import cv2 as cv
from constants import FACIAL_LANDMARKS
from sources import readBosphorus
from augment import getAugmentedDataset

# calculate the average distance between the true points and the predicted points
def avg_dist_coordinates(y_true, y_pred):
    A = K.reshape(y_true, (22, 2, -1))
    B = K.reshape(y_pred, (22, 2, -1))
    Z = A-B
    z = K.abs(K.sqrt(K.sum(K.square(Z), axis=-1, keepdims=False)))
    avg = K.mean(z, axis=None, keepdims=False)
    return avg


def getModel():
    model = Sequential()

    # 1st convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_normal",
                     input_shape=(2, 128, 128), data_format="channels_first"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Dropout(rate=0.0))

    # 2nd convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Dropout(rate=0.0))

    # 3rd convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Dropout(rate=0.0))

    # flatten output
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(512, activation="relu"))

    # output layer
    model.add(Dense(44))

    # compile the model with Adam optimizer + use custom metric as loss and metric
    # TODO: experiment with "MSE" loss and custom metric + custom metric as loss & metric
    model.compile(optimizer=keras.optimizers.Adam(), loss=avg_dist_coordinates, metrics=["acc", avg_dist_coordinates])

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
    y = y.copy()
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


def visualize_prediction(x, y_pred, y_true, plot_landmarks=True, annotate_landmarks=False):
    # plot the RGB image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Predicted Landmarks")
    plt.imshow(x[0, :, :], cmap='gray')
    y_pred *= 128
    y_pred = y_pred.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        plt.scatter(y_pred[:, 0], y_pred[:, 1], s=20, c="red", alpha=1.0, edgecolor='black')

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y_pred[i, 0], y_true[i, 1]), color="white", fontsize="small")

    plt.subplot(1, 2, 2)
    plt.title("True Landmarks")
    plt.imshow(x[0, :, :], cmap='gray')
    y_true *= 128
    y_true = y_true.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        plt.scatter(y_true[:, 0], y_true[:, 1], s=20, c="red", alpha=1.0, edgecolor='black')

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y_true[i, 0], y_true[i, 1]), color="white", fontsize="small")

    plt.show()


def main(name, plot_graph=False):
    # build model
    model = getModel()

    # retrieve dataset
    X, y = getDataset()

    # split into train and tests
    # TODO: think about stratifying according to the 6 basic emotions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_aug, y_aug = getAugmentedDataset(X_train, y_train)

    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    tensorboard = TensorBoard(update_freq='batch', log_dir="./logs/{}/".format(name))
    early = EarlyStopping(monitor='val_avg_dist_coordinates', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_avg_dist_coordinates', verbose=True, factor=0.5, patience=4)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30,
                        batch_size=32, callbacks=[tensorboard, early, reduce_lr], verbose=True)
    model.save("./network.hdf5")

    if plot_graph:
        # summarize history for avg_dist_coordinates
        plt.plot(history.history['avg_dist_coordinates'])
        plt.plot(history.history['val_avg_dist_coordinates'])
        plt.title('Mean Euclidean Distance: {}'.format(name))
        plt.ylabel('Mean Euclidean Distance')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    while True:
        i = random.randint(0, X_test.shape[0]-1)
        y_pred = model.predict(np.asarray([X_test[i]]))  # predict the facial landmarks
        visualize_prediction(X_test[i], y_pred, y_test[i])  # visualize the predictions next to the true landmarks

if __name__ == "__main__":
    main("Test")
