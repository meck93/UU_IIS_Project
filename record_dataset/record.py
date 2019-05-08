import pyrealsense2 as rs
import numpy as np
import cv2


def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)

    return config


def record(foldername):

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    # configure the realsense camera
    pipeline = rs.pipeline()
    config = create_camera_configuration()
    frame_aligner = rs.align(rs.stream.color)
    clipping_distance_meters = 0.2

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = clipping_distance_meters / depth_scale
    i = 0
    try:
        while cv2.getWindowProperty('RealSense', 0) != -1:
            # get a raw image into the pipeline
            frames = pipeline.wait_for_frames()
            aligned_frames = frame_aligner.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Preprocessing
            # image to display (normalized float bgr)
            image = np.asanyarray(color_frame.get_data()).astype(np.float32)
            image -= np.min(image[:])
            image /= np.max(image[:])

            # Postprocessing
            image = (image * 255).astype(np.uint8)

            depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            
            # for depth visualization
            depth_modulo = depth % 255
            depth_modulo = depth_modulo.astype(np.uint8)

            # Display the final image
            cv2.imshow('RealSense', image)

            #cv2.imwrite(foldername + "RGB_{:04d}.png".format(i), image)
            #cv2.imwrite(foldername + "D_{:04d}.png".format(i), depth)

            i += 1
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    record("./dataset/")