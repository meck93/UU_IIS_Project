import cv2
import pyrealsense2 as rs
import numpy as np
from record_dataset.record import create_camera_configuration
from face_recognition import crop_image
from record_dataset.source import RealSenseCam
import matplotlib.pyplot as plt

def main():
    source = RealSenseCam()

    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    image, depth = source.getFrame()
    im1 = ax1.imshow(image)
    im2 = ax2.imshow(image)
    im3 = ax3.imshow(np.zeros((128,128)), cmap='gray', vmin=0, vmax=255)
    sc1 = ax1.scatter([],[], s=5, c="white", alpha=1.0, edgecolor='black')

    plt.ion()

    while True:
        image, depth = source.getFrame()

        image2, face = crop_image(image, depth)

        landmark_set = fa.get_landmarks(image)
        if landmark_set:
            for landmarks in landmark_set:
                sc1.set_offsets(landmarks)

        if face is not None:
            x, y, w, h = face
            face_image = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (128, 128))
            im3.set_data(face_image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        im1.set_data(image_rgb)
        im2.set_data(image2_rgb)
        plt.pause(0.1)


if __name__ == "__main__":
    main()
