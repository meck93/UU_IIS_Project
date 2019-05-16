from buildModel import getDataset, visualize
from constants import FACIAL_LANDMARKS
import random
import numpy as np

def augmentDataset(X, Y, factor=3):
    size = round(len(X)*factor)
    X_aug = []
    y_aug = []
    
    while len(X_aug) < size:
        i = random.randint(0, len(X)-1)
        x = X[i].copy()
        y = Y[i].copy()
        visualize(x, y)
        print(y.shape)
        x, y = randomFlip(x, y)
        #x, y = randomNoise(x, y)
        print(y.shape)
        visualize(x, y)

    print(size)
    return X, y

def randomFlip(x, y):
    if random.choice([True, False]):
        return x, y
    x_aug = np.asarray([np.flip(x[0], 1), np.flip(x[1], 1)], dtype='float')
    y_aug = flipLandmarks(y)
    return x_aug, y_aug


def flipLandmarks(y):
    y = y.reshape((22,2))
    # flip
    y[:, 0] = 1 - y[:, 0]
    # switch left <-> right
    landmarks_lower = list(map(str.lower, FACIAL_LANDMARKS))
    y_new = []
    for i, l in enumerate(landmarks_lower):
        if "left" in l:
            right = l.replace("left", "right")
            idx = landmarks_lower.index(right)
            y_new.append(y[idx])
        elif "right" in l:
            left = l.replace("right", "left")
            idx = landmarks_lower.index(left)
            y_new.append(y[idx])
        else:
            y_new.append(y[i])
    y_new = np.asarray(y_new)
    y_new = y_new.flatten()
    return y_new


def randomNoise(x, y):
    noise_spread_im = random.uniform(0, 0.05)
    noise_spread_dep = random.uniform(0, 0.05)
    noise_im = np.random.normal(0, noise_spread_im, x[0].shape)
    noise_dep = np.random.normal(0, noise_spread_dep, x[1].shape)
    x[0] += noise_im
    x[1] += noise_dep
    return x, y


def main():
    X, y = getDataset()
    X_aug, y_aug = augmentDataset(X, y)


if __name__ == "__main__":
    main()