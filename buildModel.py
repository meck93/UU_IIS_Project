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
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

import cv2 as cv
from constants import FACIAL_LANDMARKS
from sources import readBosphorus
from augment import getAugmentedDataset
from visualization import visualize_prediction

def avg_l2_dist(y_true, y_pred):
    """
    Calculate the average distance between the true points and the predicted points using the L2-Norm.

    Inputs:
    - y_true: has shape (44,1) a vector of coordinates e.g., (x1,y1,x2,y2,...). 
              reshaped such that each cooridnate pairs are together e.g., [[x1,y1], [x2, y2], ...]
    - y_pred: has shape (44,1) a vector of coordinates e.g., (x1,y1,x2,y2,...). 
              reshaped such that each cooridnate pairs are together e.g., [[x1,y1], [x2, y2], ...]
    """
    A = K.reshape(y_true, (22, 2, -1)) 
    B = K.reshape(y_pred, (22, 2, -1))
    Z = A-B
    z = K.abs(K.sqrt(K.sum(K.square(Z), axis=-1, keepdims=False)))
    avg = K.mean(z, axis=None, keepdims=False)
    return avg

def avg_l1_dist(y_true, y_pred):
    """
    Calculate the average distance between the true points and the predicted points using the L2-Norm.

    Inputs:
    - y_true: has shape (44,1) a vector of coordinates e.g., (x1,y1,x2,y2,...). 
              reshaped such that each cooridnate pairs are together e.g., [[x1,y1], [x2, y2], ...]
    - y_pred: has shape (44,1) a vector of coordinates e.g., (x1,y1,x2,y2,...). 
              reshaped such that each cooridnate pairs are together e.g., [[x1,y1], [x2, y2], ...]
    """
    A = K.reshape(y_true, (22, 2, -1))
    B = K.reshape(y_pred, (22, 2, -1))
    Z = K.abs(A-B)
    z = K.sum(Z, axis=-1, keepdims=False)
    avg = K.mean(z, axis=None, keepdims=False)
    return avg

def createModel(hasDepthData=True, l2_loss=True):
    if hasDepthData:
        in_shape = (2,128,128)
    else:
        in_shape = (1,128,128)

    model = Sequential()

    # 1st convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal",
                     input_shape=in_shape, data_format="channels_first"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first"))
    model.add(Dropout(rate=0.05))

    # 2nd convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first"))
    model.add(Dropout(rate=0.05))

    # 3rd convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format="channels_first"))
    model.add(Dropout(rate=0.05))

    # flatten output
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(256, activation="relu"))

    # output layer
    model.add(Dense(44))

    # compile the model with Adam optimizer + use custom metric as loss and metric
    if l2_loss:
        model.compile(optimizer=keras.optimizers.Adam(), loss=avg_l2_dist, metrics=["acc", avg_l2_dist])
    else:
        model.compile(optimizer=keras.optimizers.Adam(), loss=avg_l1_dist, metrics=["acc", avg_l1_dist])

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
    x = np.asarray(x, dtype="float")

    y = []

    for l in FACIAL_LANDMARKS:
        pos = labels.index(l)
        p = points[pos]
        p[0] = p[0] / gray.shape[1]
        p[1] = p[1] / gray.shape[0]
        y.append(p)

    y = np.asarray(y, dtype="float")
    y = y.flatten()

    return x, y


def _check_dir_or_create(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def train_model(name, plot_graph=False, hasDepthData=True, l2_loss=True):
    # build model
    model = createModel(hasDepthData, l2_loss)

    # retrieve dataset
    X, y = getDataset()

    # split into train and tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # augment the training data set
    X_aug, y_aug = getAugmentedDataset(X_train, y_train)

    # drop the depth channel - only use the grayscale information
    if not hasDepthData:
        X_test = X_test[:,0,:,:]
        X_test = np.expand_dims(X_test, axis=1)
        X_train = X_train[:,0,:,:]
        X_train = np.expand_dims(X_train, axis=1)
        X_aug = X_aug[:,0,:,:]
        X_aug = np.expand_dims(X_aug, axis=1)

    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # setup the correct validation metric
    val_metric = "val_avg_l2_dist" if l2_loss else "val_avg_l1_dist"

    # set up tensorboard, early stopping and learning rate reducer as callbacks
    tb = TensorBoard(update_freq="batch", log_dir="./logs/{}/".format(name))
    reduce_lr = ReduceLROnPlateau(monitor=val_metric, verbose=True, factor=0.5, patience=3)

    # setup checkpoint callback
    cp_filepath = "./models/{}/".format(name)
    _check_dir_or_create(cp_filepath)
    cp = ModelCheckpoint(filepath="./models/{}/model.hdf5".format(name), monitor=val_metric, verbose=True, save_best_only=True)

    # train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40,
                        batch_size=32, callbacks=[tb, reduce_lr, cp], verbose=True)

    if plot_graph:
        # summarize history for avg_l2_dist
        plt.plot(history.history[val_metric[4:]])
        plt.plot(history.history[val_metric])
        plt.title("Mean Euclidean Distance: {}".format(name))
        plt.ylabel("Mean Euclidean Distance")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    while True:
        i = random.randint(0, X_test.shape[0]-1)
        y_pred = model.predict(np.asarray([X_test[i]]))  # predict the facial landmarks
        visualize_prediction(X_test[i], y_pred, y_test[i])  # visualize the predictions next to the true landmarks


def use_pretrained_model(name, hasDepthData=True, l2_loss=True):
    # build model
    model = createModel(hasDepthData, l2_loss)

    # retrieve dataset
    X, y = getDataset()

    # split into train and tests
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # drop the depth channel - only use the grayscale information
    if not hasDepthData:
        X_test = X_test[:, 0, :, :]
        X_test = np.expand_dims(X_test, axis=1)

    # load a trained model from file
    if l2_loss:
        model = load_model("./models/{}/model.hdf5".format(name), custom_objects={"avg_l2_dist": avg_l2_dist})
    else:
        model = load_model("./models/{}/model.hdf5".format(name), custom_objects={"avg_l1_dist": avg_l1_dist})

    while True:
        i = random.randint(0, X_test.shape[0]-1)
        y_pred = model.predict(np.asarray([X_test[i]]))  # predict the facial landmarks
        visualize_prediction(X_test[i], y_pred, y_test[i])  # visualize the predictions next to the true landmarks

if __name__ == "__main__":
    # use_pretrained_model("", hasDepthData=True, l2_loss=True)
    train_model("ReplaceMeWithModelName", plot_graph=True, hasDepthData=True, l2_loss=True)
