import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

def main(filename, plot_with_lines=True, save=True):
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)

    # read image
    input = io.imread(filename)

    # get the facial landmarks  
    landmarks = fa.get_landmarks(input)[-1]

    # plot the image and the facial landmarks
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(input)

    if plot_with_lines:
        ax.plot(landmarks[0:17,0],landmarks[0:17,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[17:22,0],landmarks[17:22,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[22:27,0],landmarks[22:27,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[27:31,0],landmarks[27:31,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[31:36,0],landmarks[31:36,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[36:42,0],landmarks[36:42,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[42:48,0],landmarks[42:48,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[48:60,0],landmarks[48:60,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
        ax.plot(landmarks[60:68,0],landmarks[60:68,1],marker='o',markersize=2,linestyle='-',color='w',lw=0.75)
    else:
        ax.scatter(landmarks[:,0],landmarks[:,1], s=5, c="white", alpha=1.0, edgecolor='black') 

    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    surf = ax.scatter(landmarks[:,0],landmarks[:,1],c="cyan", alpha=1.0, edgecolor='black')

    # set axis origin to be top left corner (0,0)
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    plt.show()

    if save:
        output_filename = filename[filename[:-5].rfind("/")+1:filename.rfind(".")]
        output_filename = "../datasets/self_created/test/{}Annotated.png".format(output_filename)
        fig.savefig(output_filename, format='png', quality=95)


if __name__ == "__main__":
    main(filename="../datasets/self_created/test/Lukas2.png", plot_with_lines=True, save=True)