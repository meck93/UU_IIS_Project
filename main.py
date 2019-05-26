import numpy as np
import face_alignment

from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from sources.camera import RealSenseCam
from sources.dataset import Dataset
from visualization import (initVisualization, updateVisualization,
                           visualise_and_compare, visualise_bulat)
import matplotlib.pyplot as plt


def main(modelname, hasDepthData=True, l2_loss=True):
    source = Dataset()  # for camera use RealsenseCam()

    modelpath = "./datasets/models/{}/model.hdf5".format(modelname)
    landmarkDetector = LandmarkDetector(modelpath, hasDepthData, l2_loss)

    vis = initVisualization(hasDepthData)

    while True:
        image, depth = source.getFrame()

        faces = detectFaces(image, depth)

        if faces.shape != (1, 2, 128, 128) or faces.shape != (1, 1, 128, 128):
            continue  # sometimes the frame has shape (0,) because no faces was detected, skip these frames

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


def compare_to_bulat(modelname, hasDepthData, l2_loss):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=True)

    source = Dataset()  # for camera use RealsenseCam()

    # load both models
    modelpath = "./datasets/models/{}/model.hdf5".format(modelname)
    model = LandmarkDetector(modelpath, hasDepthData, l2_loss)

    while True:
        image, depth = source.getFrame()
        print(image.shape)
        print(image)

        faces = detectFaces(image, depth)
        # print(faces.shape)
        # bulat_in = faces[0, 0, :, :]
        # bulat_in = np.expand_dims(bulat_in, axis=2)
        # print(bulat_in.shape)
        # bulat_in *= 128
        # print(faces[0, 0, :, :])
        # plt.imshow(faces[0, 0, :, :], cmap="gray")
        # plt.show()

        # predict the facial landmarks using Bulat et al. method
        bulat_pred = fa.get_landmarks(image)[0]

        if bulat_pred is None:
            continue

        # remove the depth channel
        if not hasDepthData:
            faces = faces[:, 0, :, :]
            faces = np.expand_dims(faces, axis=1)

        # predict the facial landmarks for the depth model
        our_pred = model.detectLandmarks(faces)

        # reshape for visualizing
        faces = faces[0, :, :, :]

        # visualize the predictions next to the true landmarks
        visualise_bulat(faces, our_pred, image, bulat_pred, "Our Model", "Bulat et al.")


if __name__ == "__main__":
    # main("Aug4L2Drop0.05GrayOnly_V2", hasDepthData=False, l2_loss=True)
    # compare_models("Aug4L2Drop0.05_V2", "Aug4L2Drop0.05GrayOnly_V2", m1_depth=True, m2_depth=False, m1_l2_loss=True, m2_l2_loss=True)
    compare_to_bulat("Aug4L2Drop0.05GrayOnly_V2", hasDepthData=False, l2_loss=True)
