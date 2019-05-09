from sources.camera import RealSenseCam
import matplotlib.pyplot as plt
from faceDetection import detectFaces
from landmarkDetection import detectLandmarks

def main():

    source = RealSenseCam()

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        landmarks = detectLandmarks(faces)


if __name__ == "__main__":
    main()
