import cv2 
import numpy as np
import face_alignment

def main():
    image_store = None
    color = np.array((0, 0, 0))

    cv2.startWindowThread()
    cv2.namedWindow("WebCam-Facial-Landmarks")

    # run the 2D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', face_detector='sfd', flip_input=True)

    # camera caputre object
    cam = cv2.VideoCapture(0)

    while cv2.getWindowProperty('WebCam-Facial-Landmarks', 0) != -1:
        # raw image capture and preparation
        ret, frame = cam.read()
        if not ret:
            print("Could not get a frame from the camera, exiting.")
            break

        # try to get the landmarks for all faces
        landmark_set = fa.get_landmarks(frame)

        # if no face is detected, skip the annotating part
        if not landmark_set:
            continue

        for landmarks in landmark_set:
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them as red circles on the image
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # display the final image
        cv2.imshow("WebCam-Facial-Landmarks", frame)

        # close if the window is closed or escape is pressed
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
