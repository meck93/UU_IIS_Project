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


def readLM2File(filepath):
    with open(filepath, "r") as f:
        pass

def main():
    datasetlocation = "C:/Users/Lukas/Desktop/code/Bosphorus/data/"
    filepath = datasetlocation + "bs000/bs000_CAU_A22A25_0.lm2"
    #readLM2File(filepath)

    for dirpath, dirnames, filenames in os.walk(datasetlocation):
        for f in filenames:
            if f.endswith(".bnt"):
                nrows, ncols, zmin, imfile, data = readBNTFile(dirpath + "/" + f)
                print(imfile)



if __name__ == "__main__":
    main()
