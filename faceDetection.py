import numpy as np
import cv2


# facial recognition code
# can use the coordinates x,y,x+w,y+h to segment the face
def crop_image(image, dpth, clipping_distance=100):
    image = image.copy()
    background = np.zeros((480, 640, 3))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # processing
    face_cascade = cv2.CascadeClassifier(cv2.__path__[0] + '/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) > 0:
        mask = np.zeros((image.shape[0], image.shape[1]))
        depth_image = np.asanyarray(dpth)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            person_mask = create_segmentation_mask(x, y, w, h, image, depth_image, clipping_distance)
            mask = np.logical_or(mask, person_mask)
    else:
        mask = np.ones((image.shape[0], image.shape[1]))
        faces = [None]

    # Postprocessing
    image = (mask[..., None] * image + (1 - mask[..., None]) * background)
    image = image.astype(np.uint8)
    return image, faces[0]


def create_segmentation_mask(x, y, w, h, image, depth_image, clipping_distance):
    mask = np.zeros((image.shape[0], image.shape[1]))
    average_depth = np.mean(depth_image[y:y+h, x:x+w])
    high = min(average_depth + clipping_distance, 2**16)+100
    low = max(average_depth - clipping_distance, 0)-100
    mask[np.logical_and(low <= depth_image, depth_image <= high)] = 1.0
    return mask


def main(rgb_file, depth_file):
    # create a named window
    cv2.startWindowThread()
    cv2.namedWindow("Face-Recognition")

    # process the image
    image = crop_image(rgb_file, depth_file)

    # show the processed image
    cv2.imshow("Face-Recognition", image)

    while cv2.getWindowProperty('Face-Recognition', 0) != -1:
        # close if the window is closed or escape is pressed
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break

    cv2.destroyAllWindows()


def detectFaces(image, depth, boundary=(-0.05, 0.15)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.__path__[0] + '/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    output = []
    im_h, im_w = gray_image.shape

    if faces is not None:
        for (x, y, w, h) in faces:
            b_x = int(round(boundary[0] * w))
            b_y = int(round(boundary[1] * h))
            x1 = min(max(x - b_x, 0), im_w)
            y1 = min(max(y - b_y, 0), im_h)
            x2 = min(max(x + w + b_x, 0), im_w)
            y2 = min(max(y + h + b_y, 0), im_h)
            d = depth[y1:y2, x1:x2]
            im = gray_image[y1:y2, x1:x2]

            d = cv2.resize(d, (128, 128))
            d = np.asarray(d, dtype='float32')
            # remove everything that is too far from the average
            avg = np.average(d)
            lower_b = avg-avg*0.25
            upper_b = avg+avg*0.25
            d_copy = d.copy()
            d[d_copy < lower_b] = np.nan
            d[d_copy > upper_b] = np.nan
            # scale to range 0-1
            d_min = np.nanmin(d)
            d_max = np.nanmax(d)
            d = (d-d_min) / (d_max-d_min)
            # invert to match bosphorus
            d = 1-d
            d = np.nan_to_num(d)
            # use a median filter to remove high frequent salt and pepper noise
            d = cv2.medianBlur(d, 5)
            # use a closing operation with a disk-shaped structuring element to close holes
            d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            # scale to range 0-1 again
            d_min = np.min(d)
            d_max = np.max(d)
            if float(d_max-d_min) == 0:
                continue
            d = (d-d_min) / (d_max-d_min)
            if np.isnan(np.sum(d)):
                continue

            im = cv2.resize(im, (128, 128))
            im = np.asarray(im, dtype="float")
            im /= 255.0

            x = [im, d]
            x = np.asarray(x, dtype='float')
            output.append(x)

    output = np.asarray(output)

    return output
