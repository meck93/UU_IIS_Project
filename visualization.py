import matplotlib.pyplot as plt
import numpy as np

from constants import FACIAL_LANDMARKS


def initVisualization():
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    im1 = ax1.imshow(np.zeros((128,128)), cmap='gray', vmin=0, vmax=1)
    im2 = ax2.imshow(np.zeros((128,128)), cmap='gray', vmin=0, vmax=1)
    sc = ax1.scatter([],[], s=20, c="red", alpha=1.0, edgecolor="black")
    plt.ion() # interactive mode on
    return im1, im2, sc


def updateVisualization(x, y, vis):
    im1, im2, sc = vis
    im1.set_data(x[0, :, :])
    y = y.copy()
    y *= 128
    y = y.reshape((22, 2))
    sc.set_offsets(y)
    im2.set_data(x[1, :, :])
    plt.pause(0.1)


def visualize(x, y, plot_landmarks=True, annotate_landmarks=True):
    y = y.copy()
    # plot the RGB image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(x[0, :, :], cmap="gray")
    y *= 128
    y = y.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        plt.scatter(y[:, 0], y[:, 1], s=20, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y[i, 0], y[i, 1]), color="white", fontsize="small")

    plt.subplot(1, 2, 2)
    plt.title("Depth")
    plt.imshow(x[1, :, :], cmap="gray")
    plt.show()


def visualise_and_compare(x, y1, y2, title1, title2, plot_landmarks=True, annotate_landmarks=False):
    # plot the RGB image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(title1, fontsize=30)
    ax1.imshow(x[0, :, :], cmap="gray")
    ax1.tick_params(axis='both', which='major', labelsize=30)
    y1 *= 128
    y1 = y1.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax1.scatter(y1[:, 0], y1[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            ax1.annotate(label, (y1[i, 0], y1[i, 1]), color="white", fontsize="small")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(title2, fontsize=30)
    ax2.imshow(x[0, :, :], cmap="gray")
    ax2.tick_params(axis='both', which='major', labelsize=30)
    y2 *= 128
    y2 = y2.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax2.scatter(y2[:, 0], y2[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y2[i, 0], y2[i, 1]), color="white", fontsize="small")

    plt.show()
