import struct
from array import array
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
import random


# reads a BNT file
# returns num rows, num cols, zmin value, image filename, data as np array
def readBNTFile(filepath):
    with open(filepath, "rb") as f:
        nrows, = struct.unpack('H', f.read(2))
        ncols, = struct.unpack('H', f.read(2))
        zmin, = struct.unpack('d', f.read(8))

        str_len, = struct.unpack('H', f.read(2))
        b = f.read(str_len)
        imfile = b.decode("utf-8")

        data_len, = struct.unpack('I', f.read(4))
        data = array('d')
        data.fromfile(f, data_len)
        data = np.asarray(data, dtype='double')
        data = data.reshape((5, data_len//5)).T

        if not nrows * ncols == data.shape[0]:
            # something went wrong
            raise Exception("Error reading bnt file " + filepath + "\nnrows * ncols does not match data size")

        return nrows, ncols, zmin, imfile, data


# reads a LM2 file
# returns points array with 2D landmark positions and label names
def readLM2File(filepath):
    with open(filepath, "r") as f:
        f.readline()
        f.readline()

        line = f.readline().strip()
        numLandmarks = int(line.split(" ")[0])

        f.readline()
        f.readline()

        labels = []
        for _ in range(numLandmarks):
            labels.append(f.readline().strip())

        f.readline()
        f.readline()

        points = []
        for _ in range(numLandmarks):
            parts = f.readline().strip().split(" ")
            points.append([float(parts[0]), float(parts[1])])

        points = np.asarray(points, dtype="double")

        return points, labels


# return a list of IDs of all samples in the dataset.
# an ID is a full path to the sample without the file extension
def makeIndex(datasetlocation):
    ids = []
    for dirpath, dirnames, filenames in os.walk(datasetlocation):
        for f in filenames:
            if f.endswith(".lm2"):
                ids.append(dirpath + "/" + f[:-4])
    return ids


# visualize the image data and depth information of one sample
def visualizeSample(id, plot_landmarks=True, annotate_landmarks=True):
    nrows, ncols, zmin, imfile, data = readBNTFile(id+".bnt")
    points, labels = readLM2File(id+".lm2")
    image = io.imread(id+".png")

    # plot the RGB image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(id.split("/")[-1])
    plt.imshow(image)

    if plot_landmarks:  # plot facial landmarks as points
        plt.scatter(points[:, 0], points[:, 1], s=20, c="red", alpha=1.0, edgecolor='black')

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(labels):
            plt.annotate(label, (points[i, 0], points[i, 1]), color="white", fontsize="large")

    # plot the depth data
    plt.subplot(1, 2, 2)
    depth = data[:, 2]
    depth = depth.reshape((nrows, ncols))
    depth = np.flip(depth, 0)
    depth[depth == zmin] = np.nan  # replace background values with nan
    plt.imshow(depth, cmap='gray', vmin=np.nanmin(depth), vmax=np.nanmax(depth))
    plt.show()


def main():
    datasetlocation = "../datasets/bosphorus/data/"
    filepath = datasetlocation + "bs000/bs000_CAU_A22A25_0.lm2"

    for dirpath, dirnames, filenames in os.walk(datasetlocation):
        for f in filenames:
            if f.endswith(".lm2"):
                points, labels = readLM2File(dirpath + "/" + f)

    ids = makeIndex(datasetlocation)
    visualizeSample(ids[0])


if __name__ == "__main__":
    main()
