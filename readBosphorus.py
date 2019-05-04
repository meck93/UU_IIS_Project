import struct
from array import array
import numpy as np
import os

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
        data = data.reshape((5,data_len//5)).T

        if not nrows * ncols == data.shape[0]:
            # something went wrong
            raise Exception("Error reading bnt file "+ filepath + "\nnrows * ncols does not match data size")

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

def main():
    datasetlocation = "C:/Users/Lukas/Desktop/code/Bosphorus/data/"
    filepath = datasetlocation + "bs000/bs000_CAU_A22A25_0.lm2"

    for dirpath, dirnames, filenames in os.walk(datasetlocation):
        for f in filenames:
            if f.endswith(".lm2"):
                points, labels = readLM2File(dirpath + "/" + f)
                print(points.shape)


if __name__ == "__main__":
    main()
