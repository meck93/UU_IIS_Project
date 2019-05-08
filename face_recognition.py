import numpy as np

import cv2


# facial recognition code
# can use the coordinates x,y,x+w,y+h to segment the face
def crop_image(RGB_path, D_path, cv_lib_path, clipping_distance=100):
    img = cv2.imread(RGB_path)
    dpth = cv2.imread(D_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    background = np.zeros((480, 640, 3))
    image = np.asanyarray(img).astype(np.float32)
    image -= np.min(image[:])
    image /= np.max(image[:])

    # convert to grayscale (uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image -= np.min(gray_image[:])
    gray_image /= np.max(gray_image[:])
    gray_image_uint = gray_image * 255
    gray_image_uint = gray_image_uint.astype(np.uint8)

    # processing
    face_cascade = cv2.CascadeClassifier(cv_lib_path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)

    if len(faces) > 0:
        mask = np.zeros((image.shape[0], image.shape[1]))
        depth_image = np.asanyarray(dpth)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            person_mask = create_segmentation_mask(x, y, w, h, image, depth_image, clipping_distance)
            mask = np.logical_or(mask, person_mask)
    else:
        mask = np.ones((image.shape[0], image.shape[1]))

    # Postprocessing
    image = (mask[..., None] * image + (1 - mask[..., None]) * background)
    image = (image * 255).astype(np.uint8)
    return image


def create_segmentation_mask(x, y, w, h, image, depth_image, clipping_distance):
    mask = np.zeros((image.shape[0], image.shape[1]))
    print(depth_image.shape)
    average_depth = np.mean(depth_image[y:y+h, x:x+w])
    high = min(average_depth + clipping_distance, 2**16)+100
    low = max(average_depth - clipping_distance, 0)-100
    mask[np.logical_and(low <= depth_image, depth_image <= high)] = 1.0
    return mask

def main(rgb_file, depth_file, opencv_lib_path):
    # create a named window
    cv2.startWindowThread()
    cv2.namedWindow("Face-Recognition")

    # process the image
    image = crop_image(rgb_file, depth_file, opencv_lib_path)

    # show the processed image
    cv2.imshow("Face-Recognition", image)

    while cv2.getWindowProperty('Face-Recognition', 0) != -1:
        # close if the window is closed or escape is pressed
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # TODO: adjust input files to your local settings
    rgb_file = './datasets/self_created/dataset/rgb/RGB_0200.png'
    depth_file = './datasets/self_created/dataset/depth/D_0200.png'
    opencv_lib_path = "C:/Users/meck/.conda/envs/iis/Library/etc/haarcascades/"

    main(rgb_file, depth_file, opencv_lib_path)
