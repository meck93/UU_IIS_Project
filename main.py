from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from sources.camera import RealSenseCam
from sources.dataset import Dataset
from visualization import initVisualization, updateVisualization


def main(modelname):
    source = Dataset()  # for camera use RealsenseCam()

    modelpath = "./datasets/models/{}/model.hdf5".format(modelname)
    landmarkDetector = LandmarkDetector(modelpath)

    vis = initVisualization()

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        landmarks = landmarkDetector.detectLandmarks(faces)

        for f, l in zip(faces, landmarks):
            updateVisualization(f, l, vis)


if __name__ == "__main__":
    main("Test")
