from sources.camera import RealSenseCam
from sources.dataset import Dataset
import matplotlib.pyplot as plt
from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from buildModel import visualize

def main():

    source = Dataset() # for camera use RealsenseCam()

    landmarkDetector = LandmarkDetector()

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        landmarks = landmarkDetector.detectLandmarks(faces)

        for f, l in zip(faces, landmarks):
            visualize(f, l, annotate_landmarks=False)


if __name__ == "__main__":
    main()
