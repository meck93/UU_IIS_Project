from keras.models import load_model
import numpy as np
from buildModel import avg_l2_dist, avg_l1_dist
from constants import BEST_MODEL_NAME, BEST_MODEL_ISL2LOSS

class LandmarkDetector():
    def __init__(self, modelname=BEST_MODEL_NAME, hasDepthData=True, l2_loss=BEST_MODEL_ISL2LOSS):
        self.l1_loss = {"avg_l1_dist": avg_l1_dist}
        self.l2_loss = {"avg_l2_dist": avg_l2_dist}
        self.hasDepthData = hasDepthData

        modelpath = "./datasets/models/{}/model.hdf5".format(modelname)

        if l2_loss:
            self.model = load_model(modelpath, custom_objects=self.l2_loss)
        else:
            self.model = load_model(modelpath, custom_objects=self.l1_loss)

    def detectLandmarks(self, input_vector):
        if not self.hasDepthData:  # remove the depth channel
            input_vector = input_vector[:, 0, :, :]
            input_vector = np.expand_dims(input_vector, axis=1)

        if self.hasDepthData and input_vector.shape == (1,2,128,128): # grayscale and depth input
            return self.model.predict(input_vector)
        elif not self.hasDepthData and input_vector.shape == (1,1,128,128): # only grayscale input
            return self.model.predict(input_vector)
        else:
            return []
