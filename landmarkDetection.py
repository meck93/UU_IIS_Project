from keras.models import load_model

class LandmarkDetector():
    def __init__(self, modelpath="./network.hdf5"):
        self.model = load_model(modelpath)

    def detectLandmarks(self, input_vector):
        return self.model.predict(input_vector)
