import matplotlib.pyplot as plt
import numpy as np

from constants import FACIAL_LANDMARKS


def initVisualization(hasDepthData):
    ax1 = plt.subplot(1, 2, 1)    
    im1 = ax1.imshow(np.zeros((128,128)), cmap='gray', vmin=0, vmax=1)
    sc = ax1.scatter([], [], s=20, c="red", alpha=1.0, edgecolor="black")

    if hasDepthData:
        ax2 = plt.subplot(1, 2, 2)
        im2 = ax2.imshow(np.zeros((128,128)), cmap='gray', vmin=0, vmax=1)
        plt.ion() # interactive mode on
        return im1, im2, sc
    else:
        plt.ion()  # interactive mode on
        return im1, None, sc


def updateVisualization(x, y, vis, hasDepthData):
    im1, im2, sc = vis
    im1.set_data(x[0, :, :])
    y = y.copy()
    y *= 128
    y = y.reshape((22, 2))
    sc.set_offsets(y)
    
    if hasDepthData:
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


def visualize_prediction(x, y_pred, y_true, plot_landmarks=True, annotate_landmarks=False):
    # plot the RGB image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Predicted Landmarks", fontsize=30)
    ax1.imshow(x[0, :, :], cmap="gray")
    ax1.tick_params(axis='both', which='major', labelsize=30)
    y_pred *= 128
    y_pred = y_pred.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax1.scatter(y_pred[:, 0], y_pred[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            ax1.annotate(label, (y_pred[i, 0], y_true[i, 1]), color="white", fontsize="small")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("True Landmarks", fontsize=30)
    ax2.imshow(x[0, :, :], cmap="gray")
    ax2.tick_params(axis='both', which='major', labelsize=30)
    y_true *= 128
    y_true = y_true.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax2.scatter(y_true[:, 0], y_true[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y_true[i, 0], y_true[i, 1]), color="white", fontsize="small")

    plt.show()


def visualise_comparison(x, y_pred, y_true, plot_landmarks=True, annotate_landmarks=False):
    # plot the RGB image
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("With Depth Data", fontsize=30)
    ax1.imshow(x[0, :, :], cmap="gray")
    ax1.tick_params(axis='both', which='major', labelsize=30)
    y_pred *= 128
    y_pred = y_pred.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax1.scatter(y_pred[:, 0], y_pred[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            ax1.annotate(label, (y_pred[i, 0], y_true[i, 1]), color="white", fontsize="small")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Without Depth Data", fontsize=30)
    ax2.imshow(x[0, :, :], cmap="gray")
    ax2.tick_params(axis='both', which='major', labelsize=30)
    y_true *= 128
    y_true = y_true.reshape((22, 2))

    if plot_landmarks:  # plot facial landmarks as points
        ax2.scatter(y_true[:, 0], y_true[:, 1], s=90, c="red", alpha=1.0, edgecolor="black")

    if annotate_landmarks:  # annotate each landmark with its label
        for i, label in enumerate(FACIAL_LANDMARKS):
            plt.annotate(label, (y_true[i, 0], y_true[i, 1]), color="white", fontsize="small")

    plt.show()
