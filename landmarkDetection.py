from keras.models import load_model
from buildModel import avg_l2_dist, avg_l1_dist

class LandmarkDetector():
    def __init__(self, modelpath="./network.hdf5"):
        self.l1_loss = {"avg_l1_dist": avg_l1_dist}
        self.l2_loss = {"avg_l2_dist": avg_l2_dist}
        self.model = load_model(modelpath, custom_objects=self.l2_loss)

    def detectLandmarks(self, input_vector):
        if input_vector.shape != (1,2,128,128):
            return []
        return self.model.predict(input_vector)
