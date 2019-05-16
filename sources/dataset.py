from .camera import Source
import cv2
import os
import numpy as np
import pandas as pd


class Dataset(Source):
    def __init__(self, path='./datasets/self_created/dataset/'):
        self.path = path
        self.i = 0
        if not os.path.isfile(self.path + "frames_used.csv"):
            raise Exception("Dataset source file does not exist:\n" + self.path + "frames_used.csv is missing!")
        self.frames = self._get_processed_frames()

    def _get_processed_frames(self):
        frames = pd.read_csv(self.path + "frames_used.csv")
        return frames['filename'].values

    def getFrame(self):
        # create the filename of the current frame
        image_index = self.frames[(self.i % len(self.frames))]

        # read rgb image
        path_rgb = self.path + "RGB_{:04d}.png".format(image_index)
        image = cv2.imread(path_rgb)

        # read depth image
        path_depth = self.path + "D_{:04d}.png".format(image_index)
        depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)

        if image is None or depth is None:
            # restart
            self.i = 0
            return self.getFrame()
        self.i += 1
        return image, depth

    def convertRowStringToCoordinates(self, string):
        row = []
        elements = string.strip().split(",")
        for element in elements:
            if element[0] == "[" and element[-1] == "]":
                # TODO: theoretially, I could remove the magic numbers and replace them by regex terms
                val1 = int(element[1:4])
                val2 = int(element[-5:-2])
                row.append([val1, val2])
            else:
                row.append([element])

        return row

    def getLandmarks(self, print_preview=True):
        path_landmarks = self.path + "landmarks_dataset.csv"
        landmarks = list()
        filenames = list()

        with open(path_landmarks, "r") as f:
            landmark_names = f.readline().strip().split(",")

            for line in f:
                row = self.convertRowStringToCoordinates(line)
                landmarks.append(np.asarray(row[:-1]))
                filenames.append(row[-1])

        if print_preview:
            print("Landmark Names\n", landmark_names, "\n")
            for landmark, filename in zip(landmarks[:2], filenames[:2]):
                print("Landmark Coordinates\n", landmark, "\nFilename\n", filename, "\n")

        return landmarks, filenames, landmark_names
