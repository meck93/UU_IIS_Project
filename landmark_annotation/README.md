# UU_IIS_Project
Uppsala University - Intelligent Interactive System Project

## Facial Landmark Detection
To get the repository to work, run ```python setup.py``` first. 
- This will create a new `conda` environment called `IIS`. 
- Install all the necessary packages: 
  - `PyTorch > 1.0`
  - `face_alignment`

### Webcam
Run `python webcam.py` for interactive webcam mode -> visualizes the facial landmark on the webcam output.

### Pictures => Landmarks
Run `python landmark{2|3}D.py` for 2D or 3D facial landmark output for any input picture.
