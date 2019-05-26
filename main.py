import numpy as np

from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from sources.camera import RealSenseCam
from sources.dataset import Dataset
from visualization import (initVisualization, updateVisualization,
                           visualise_and_compare)


def main(modelname, hasDepthData=True, l2_loss=True):
    source = Dataset()  # for camera use RealsenseCam()

    modelpath = "./datasets/models/{}/model.hdf5".format(modelname)
    landmarkDetector = LandmarkDetector(modelpath, hasDepthData, l2_loss)

    vis = initVisualization(hasDepthData)

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        if faces.shape != (1,2,128,128) or faces.shape != (1,1,128,128):
            continue # sometimes the frame has shape (0,) because no faces was detected, skip these frames

        if not hasDepthData:  # remove the depth channel
            faces = faces[:, 0, :, :]
            faces = np.expand_dims(faces, axis=1)

        landmarks = landmarkDetector.detectLandmarks(faces)

        for f, l in zip(faces, landmarks):
            updateVisualization(f, l, vis, hasDepthData)


def compare_models(modelname1, modelname2, m1_depth, m2_depth, m1_l2_loss, m2_l2_loss):
    source = Dataset()  # for camera use RealsenseCam()

    # load both models
    modelpath1 = "./datasets/models/{}/model.hdf5".format(modelname1)
    model_depth = LandmarkDetector(modelpath1, m1_depth, m1_l2_loss)

    modelpath2 = "./datasets/models/{}/model.hdf5".format(modelname2)
    model_no_depth = LandmarkDetector(modelpath2, m2_depth, m2_l2_loss)

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        # remove the depth channel
        faces_no_depth = faces[:, 0, :, :]
        faces_no_depth = np.expand_dims(faces_no_depth, axis=1)

        # predict the facial landmarks for the depth model
        y_depth = model_depth.detectLandmarks(faces)

        # predict the facial landmarks for the no depth model
        y_no_depth = model_no_depth.detectLandmarks(faces_no_depth)

        # reshape for visualizing
        faces = faces[0, :, :, :]

        # visualize the predictions next to the true landmarks
        visualise_and_compare(faces, y_depth, y_no_depth, "With Depth Data", "Without Depth Data")


if __name__ == "__main__":
    main("Aug4L2Drop0.05GrayOnly_V2", hasDepthData=False, l2_loss=True)
    # compare_models("Aug4L2Drop0.05_V2", "Aug4L2Drop0.05GrayOnly_V2", m1_depth=True, m2_depth=False, m1_l2_loss=True, m2_l2_loss=True)
