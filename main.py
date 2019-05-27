import numpy as np
from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from sources.camera import RealSenseCam
from sources.dataset import Dataset
from visualization import (initVisualization, updateVisualization,
                           visualise_and_compare)
from constants import BEST_MODEL_NAME, BEST_MODEL_ISL2LOSS

def main(modelname=BEST_MODEL_NAME, hasDepthData=True, l2_loss=BEST_MODEL_ISL2LOSS):
    source = Dataset()  # for camera use RealsenseCam()

    landmarkDetector = LandmarkDetector(modelname, hasDepthData, l2_loss)

    vis = initVisualization()

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        landmarks = landmarkDetector.detectLandmarks(faces)

        for f, l in zip(faces, landmarks):
            updateVisualization(f, l, vis)


def compare_models(modelname1, modelname2, m1_depth, m2_depth, m1_l2_loss, m2_l2_loss):
    source = Dataset()  # for camera use RealsenseCam()

    # load both models
    model_depth = LandmarkDetector(modelname1, m1_depth, m1_l2_loss)

    model_no_depth = LandmarkDetector(modelname2, m2_depth, m2_l2_loss)

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        # predict the facial landmarks for the depth model
        y_depth = model_depth.detectLandmarks(faces)

        # predict the facial landmarks for the no depth model
        y_no_depth = model_no_depth.detectLandmarks(faces)

        # reshape for visualizing
        faces = faces[0, :, :, :]

        # visualize the predictions next to the true landmarks
        visualise_and_compare(faces, y_depth, y_no_depth, "With Depth Data", "Without Depth Data")


if __name__ == "__main__":
    # main("Aug4L2Drop0.05", hasDepthData=True, l2_loss=True)
    # compare_models("Aug4L2Drop0.05_V2", "Aug4L2Drop0.05GrayOnly_V2", m1_depth=True, m2_depth=False, m1_l2_loss=True, m2_l2_loss=True)
    main()
