from constants import FACIAL_LANDMARKS
import random
import numpy as np
import cv2


def getAugmentedDataset(X, y, factor=3):
    datasetlocation = "./datasets/bosphorus"
    try:
        X = np.load(datasetlocation+"_X_aug.npy")
        y = np.load(datasetlocation+"_y_aug.npy")
        return X, y
    except IOError:
        pass

    print("Creating augmented dataset...")
    X, y = augmentDataset(X, y, factor)
    np.save(datasetlocation+"_X_aug.npy", X)
    np.save(datasetlocation+"_y_aug.npy", y)
    return X, y


def augmentDataset(X, Y, factor=3):
    size = round(len(X)*factor)
    X_aug = []
    y_aug = []
    i = 0
    
    while len(X_aug) < size:
        x = X[i].copy()
        y = Y[i].copy()
        #visualize(x, y)
        x, y = randomFlip(x, y)
        x, y = randomTranslation(x, y)
        x, y = randomRotation(x, y)
        x, y = randomBlur(x, y)
        x, y = randomNoise(x, y)
        if not all(i<1 and i>0 for i in y):
            # some y is outside image boundaries
            continue
        #visualize(x, y)
        X_aug.append(x)
        y_aug.append(y)

        if len(X_aug) % 10 == 0:
            print("{:d}/{:d}".format(len(X_aug), size))

        i = (i+1) % len(X)

    X_aug = np.asarray(X_aug)
    y_aug = np.asarray(y_aug)
    return X_aug, y_aug


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


def randomRotation(x, y):
    rot = random.choice([-9, -6. -3, 0, 3, 6, 9])
    if rot == 0:
        return x, y

    # rotate x
    im = x[0]
    dep = x[1]
    h,w = im.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    im2 = cv2.warpAffine(im, M, (w,h), borderMode=cv2.BORDER_REPLICATE)
    dep2 = cv2.warpAffine(dep, M, (w,h))

    # rotate y
    y = y.reshape((22,2))
    M2 = cv2.getRotationMatrix2D((0.5, 0.5), rot, 1)
    for i, p in enumerate(y):
        px = M2[0,0]*p[0] + M2[0,1]*p[1] + M2[0,2]
        py = M2[1,0]*p[0] + M2[1,1]*p[1] + M2[1,2]
        y[i] = np.asarray([px, py])
    y = y.flatten()
    return np.asarray([im2, dep2]), y


def randomTranslation(x, y):
    t_x = random.randint(-7,7)
    t_y = random.randint(-7,7)
    if t_x == 0 and t_y == 0:
        return x, y

    # translate x
    im = x[0]
    dep = x[1]
    h,w = im.shape
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    im2 = cv2.warpAffine(im, M, (w,h), borderMode=cv2.BORDER_REPLICATE)
    dep2 = cv2.warpAffine(dep, M, (w,h))

    # translate y
    y = y.reshape((22,2))
    y[:,0] += t_x/128.0
    y[:,1] += t_y/128.0
    y = y.flatten()
    return np.asarray([im2, dep2]), y


def randomBlur(x, y):
    if random.choice([True, False]):
        return x, y
    blur_im_x = random.choice([1, 3])
    blur_im_y = random.choice([1, 3])
    blur_dep_x = random.choice([1, 3, 5, 7])
    blur_dep_y = random.choice([1, 3, 5, 7])
    im = x[0]
    dep = x[1]
    im2 = cv2.GaussianBlur(im, (blur_im_x, blur_im_y), 0)
    dep2 = cv2.GaussianBlur(dep, (blur_dep_x, blur_dep_y), 0)
    return np.asarray([im2, dep2]), y


def randomNoise(x, y):
    if random.choice([True, False, False]):
        return x, y
    noise_spread_im = random.uniform(0, 0.03)
    noise_spread_dep = random.uniform(0, 0.05)
    noise_im = np.random.normal(0, noise_spread_im, x[0].shape)
    noise_dep = np.random.normal(0, noise_spread_dep, x[1].shape)
    x[0] += noise_im
    x[1] += noise_dep
    return x, y