import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
from read_bosphorus import readBosphorus
from read_bosphorus.constants import FACIAL_LANDMARKS
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import keras.backend as K


def metric_avg_distance(y_true, y_pred):
    K.reshape(y_true, (22,2))
    K.reshape(y_pred, (22,2))
    avg = 0
    for i in range(22):
        avg += K.l2_normalize(y_true[i] - y_pred[i])
    avg /= 22.0
    return avg


def getModel():
    model = Sequential()
    #model.add(BatchNormalization(input_shape=(96, 96, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", kernel_initializer="he_normal", input_shape=(2, 128, 128), data_format="channels_first"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(36, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(48, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation("relu"))
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(90, activation="relu"))
    model.add(Dense(44))

    model.compile(optimizer="rmsprop", loss="mse", metrics=["acc"])

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
        x, y_ = readBosphorus.getFeatureVector(elem)
        X.append(x)
        y.append(y_)
        #visualize(x, y_, annotate_landmarks=True)

    X = np.asarray(X)
    y = np.asarray(y)

    np.save(datasetlocation+"_X.npy", X)
    np.save(datasetlocation+"_y.npy", y)

    return X, y


def visualize(x, y, plot_landmarks=True, annotate_landmarks=True):
    # plot the RGB image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(x[0, :, :], cmap='gray')
    y *= 128
    y = y.reshape((22,2))

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model.fit(X_train, y_train, verbose=True, validation_data=(X_test, y_test), epochs=1)
    model.save("network.hdf5")

    while True:
        i = random.randint(0, X_test.shape[0]-1)
        y_pred = model.predict(np.asarray([X_test[i]]))
        visualize(X_test[i], y_pred)
        visualize(X_test[i], y_test[i])



if __name__ == "__main__":
    main()