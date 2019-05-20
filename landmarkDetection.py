from keras.models import load_model
from buildModel import avg_l2_dist

class LandmarkDetector():
    def __init__(self, modelpath="./network.hdf5"):
        self.model = load_model(modelpath, custom_objects={'avg_dist_coordinates': avg_l2_dist})

    def detectLandmarks(self, input_vector):
        if input_vector.shape != (1,2,128,128):
            return []
        return self.model.predict(input_vector)
