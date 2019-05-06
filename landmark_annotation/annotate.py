import face_alignment
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

def annotate_images(pathname):
    """
    This method annotates the facial landmarks of all images in the folder 'pathname'
    and creates a matrix containing the landmarks one row per matrix.

    Inputs:
    - pathname: path to folder
    
    Returns:
    - df: pandas dataframe containing all landmarks
    - output: csv file
    """
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)

    # get the facial landmarks  
    landmarks = fa.get_landmarks_from_directory(pathname)

    # create dataframe to store landmark coordinates
    df = pd.DataFrame()

    # column names C1-C68 + key: label
    n_columns = len(list(landmarks.values())[0][0])
    columns = ["C{}".format(i+1) for i in range(n_columns)]
    columns.append("label")

    # initialize dictionary
    data = {key: list() for key in columns}

    for key, values in landmarks.items():
        # extract the picture name to use it as label
        label = key[key.rfind("\\")+1:key.rfind(".")]
        data['label'].append(label)

        for i in range(len(values[0])):
            # fix assingment
            data["C{}".format(i+1)].append(values[0][i])

    
    # create pandas dataframe out of it
    df = pd.DataFrame.from_dict(data)
    print(df.head())

    # write result to CSV
    filename = "{}/output.csv".format(pathname)
    df.to_csv(filename, sep=",", header=True, index=False)
        
    
if __name__ == "__main__":
    annotate_images(pathname="../datasets/self_created/test/input/")