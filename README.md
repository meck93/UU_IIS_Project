# UU_IIS_Project
Uppsala University - Intelligent Interactive System Project


## Getting started
- Copy the Bosphorus data set to `./datasets/bosphorus/` such that the folder `bs000` lies in `./datasets/bosphorus/`
- Our own dataset has to be in the folder `./datasets/self_created/` such that e.g. the file `./datasets/self_created/dataset/RGB_0000.png` exists.
- `pip install -r requirements.txt`

## Structure
- `main.py` is the main entry point.
- `faceDetection.py` handles the face detection.
- `landmarkDetection.py` handles the landmark detection.
- To build the CNN for facial landmark detection use `buildModel.py`. Data augmentation is done in `augment.py`.
- `visualizaion.py` contains methods for visualization.
- The folder `sources` contains the code to read the Bosphorus dataset, read from the realsense camera and read our dataset.
- The folder `landmarkdetect_bulat` contains the advanced landmark detection method by Bulat et al.
