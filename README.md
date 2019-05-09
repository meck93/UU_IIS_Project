# UU_IIS_Project
Uppsala University - Intelligent Interactive System Project


## Getting started
- Copy the Bosphorus data set to `./datasets/bosphorus/` such that the folder `bs000` lies in `./datasets/bosphorus/`
- `pip install -r requirements.txt`

## Structure
- `main.py` is the main entry point
- `faceDetection.py` handles the face detection
- `landmarkDetection.py` handles the landmark detection
- To build the CNN for facial landmark detection use `buildModel.py`
- the folder `sources` contains the code to read the Bosphorus dataset, read from the realsense camera and read our dataset