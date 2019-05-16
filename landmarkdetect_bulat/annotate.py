import face_alignment
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")  # Adds higher directory to python modules path.

from constants import LANDMARK_MAPPING


def annotate_images(pathname, save_to_file=True, print_preview=True):
    """
    This method annotates the facial landmarks of all images in the folder 'pathname'
    and creates a matrix containing the landmarks one row per matrix.

    Inputs:
    - pathname: string - path to folder
    - save_to_file: bool - saving the result to a CSV file
    - print_preview: bool - print a preview of the dataset

    Returns:
    - df: pandas dataframe containing all landmarks (one row per image)
    """
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)

    # get the directory content
    dir_content = os.listdir(pathname)
    landmarks = {}

    for filename in dir_content:
        if filename[:3] == "RGB" and filename[-4:] == ".png": # only process files named: RBG[.*].png 
            landmarks_ = fa.get_landmarks_from_image("{}/{}".format(pathname, filename)) # get the facial landmarks
            landmarks[filename] = landmarks_ # use the filename as key

    # create dataframe to store landmark coordinates
    df = pd.DataFrame()

    # create the column names C1-C68 + key: filename
    n_columns = len(list(landmarks.values())[0][0])
    columns = ["{}".format(i+1) for i in range(n_columns)]
    columns.append("filename")

    # initialize dictionary
    data = {key: list() for key in columns}

    # transform the landmarks into a format useable by pandas
    for key, values in landmarks.items():
        # extract the picture name to use it as label
        label = key[:key.rfind(".")]
        data['filename'].append(label)

        for i in range(len(values[0])):
            # fix assingment
            data["{}".format(i+1)].append(values[0][i])

    # create pandas dataframe out of it
    df = pd.DataFrame.from_dict(data)

    # extract the relevant columns and drop the irrelevant
    relevant_columns = list(LANDMARK_MAPPING.values())
    relevant_columns = [col for col in relevant_columns if col is not None]
    not_relevant_columns = np.arange(1, 69, 1)
    not_relevant_columns = [str(col) for col in not_relevant_columns if col not in relevant_columns]
    df = df.drop(columns=not_relevant_columns)

    # rename the columns with the original names (key&value inverted)
    REVERSED_LANDMARK_MAPPING = {str(value): key for key, value in LANDMARK_MAPPING.items()}
    df = df.rename(columns=REVERSED_LANDMARK_MAPPING)

    if print_preview:
        print(df.head())

    if save_to_file:
        # write result to CSV
        filename = "./{}/landmarks_dataset.csv".format(pathname)
        df.to_csv(filename, sep=",", header=True, index=False)

        # write frame numbers to CSV
        filename = "./{}/frames_used.csv".format(pathname)
        frames = df['filename'].apply(lambda name: name[4:]) # remove the RGB_ part of the filename
        frames.to_csv(filename, sep=",", header=True, index=False)


if __name__ == "__main__":
    # TODO: make sure that the correct input folder is selected
    annotate_images(pathname="../datasets/self_created/test/")
