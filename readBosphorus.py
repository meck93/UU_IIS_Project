import struct
from array import array
import numpy as np

datasetlocation = "C:/Users/Lukas/Desktop/code/Bosphorus/data/"

filepath = datasetlocation + "bs000/bs000_CAU_A22A25_0.bnt"

with open(filepath, "rb") as f:
    # nrows = fread(fid, 1, 'uint16');
    # ncols = fread(fid, 1, 'uint16');
    # zmin = fread(fid, 1, 'float64');

    # len = fread(fid, 1, 'uint16');
    # imfile = fread(fid, [1 len], 'uint8=>char');

    # % normally, size of data must be nrows*ncols*5
    # len = fread(fid, 1, 'uint32');
    # data = fread(fid, [len/5 5], 'float64');

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


    print(nrows, ncols, zmin)
    print(str_len)
    print(imfile)

    print(data_len)
    #print(data.shape)
    print(data[-1])