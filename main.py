import numpy as np
import cv2 as cv2

def crop_image(RGB_path, D_path, clipping_distance = 100):	
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

	# Processing
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray_image_uint, 1.3, 5)
	if len(faces) > 0:
	    mask = np.zeros((image.shape[0], image.shape[1]))
	    depth_image = np.asanyarray(dpth)
	    for (x, y, w, h) in faces:
	        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
	        person_mask = create_segmentation_mask(x, y, w, h, image,
	                                               depth_image, clipping_distance)
	        mask = np.logical_or(mask, person_mask)
	else:
	    mask = np.ones((image.shape[0], image.shape[1]))

	# Postprocessing
	image = (mask[..., None] * image + (1 - mask[..., None]) *
	         background)
	image = (image * 255).astype(np.uint8)

	return image


def create_segmentation_mask(x, y, w, h, image, depth_image, clipping_distance):
    mask = np.zeros((image.shape[0], image.shape[1]))
    average_depth = np.mean(depth_image[y:y+h, x:x+w])
    high = min(average_depth + clipping_distance, 2**16)+100
    low = max(average_depth - clipping_distance, 0)-100
    mask[np.logical_and(low <= depth_image, depth_image <= high)] = 1.0
    return mask

if __name__ == '__main__':
	image = crop_image('own_dataset/dataset/RGB_0200.png','own_dataset/dataset/D_0200.png')

	cv2.imshow('img',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()