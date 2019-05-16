from keras.models import load_model
from buildModel import avg_dist_coordinates

class LandmarkDetector():
    def __init__(self, modelpath="./network.hdf5"):
        self.model = load_model(modelpath, custom_objects={'avg_dist_coordinates': avg_dist_coordinates})

    def detectLandmarks(self, input_vector):
        return self.model.predict(input_vector)
