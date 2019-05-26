from keras.models import load_model
from buildModel import avg_l2_dist, avg_l1_dist
from constants import BEST_MODEL_PATH

class LandmarkDetector():
    def __init__(self, modelpath=BEST_MODEL_PATH, hasDepthData=False, l2_loss=True):
        self.l1_loss = {"avg_l1_dist": avg_l1_dist}
        self.l2_loss = {"avg_l2_dist": avg_l2_dist}
        self.hasDepthData = hasDepthData

        if l2_loss:
            self.model = load_model(modelpath, custom_objects=self.l2_loss)
        else:
            self.model = load_model(modelpath, custom_objects=self.l1_loss)

    def detectLandmarks(self, input_vector):
        if self.hasDepthData and input_vector.shape == (1,2,128,128): # grayscale and depth input
            return self.model.predict(input_vector)
        elif not self.hasDepthData and input_vector.shape == (1,1,128,128): # only grayscale input
            return self.model.predict(input_vector)
        else:
            return []
