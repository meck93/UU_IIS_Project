from .camera import Source
import cv2
import os

class Dataset(Source):
    def __init__(self, path='./datasets/self_created/dataset/'):
        self.path = path
        self.i = 0
        if not os.path.isfile(self.path + "RGB_{:04d}.png".format(self.i)):
            raise Exception("Dataset source file does not exist:\n" + self.path + "RGB_{:04d}.png".format(self.i))

    def getFrame(self):
        path_rgb = self.path + "RGB_{:04d}.png".format(self.i)
        image = cv2.imread(path_rgb)
        path_depth = self.path + "D_{:04d}.png".format(self.i)
        depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)
        if image is None or depth is None:
            # restart
            self.i = 0
            return self.getFrame()
        self.i += 1
        return image, depth
