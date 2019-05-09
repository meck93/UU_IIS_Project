import pyrealsense2 as rs
import numpy as np

def create_camera_configuration():
    image_size = (640, 480)
    config = rs.config()
    config.enable_stream(rs.stream.color, image_size[0], image_size[1],
                         rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, image_size[0], image_size[1],
                         rs.format.z16, 30)
    return config


class RealSenseCam():
    def __init__(self):
        # configure the realsense camera
        self.pipeline = rs.pipeline()
        config = create_camera_configuration()
        self.frame_aligner = rs.align(rs.stream.color)
        clipping_distance_meters = 0.2

        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance = clipping_distance_meters / depth_scale

    def getFrame(self):
        # get a raw image into the pipeline
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.frame_aligner.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Preprocessing
        # image to display (normalized float bgr)
        image = np.asanyarray(color_frame.get_data()).astype(np.float32)
        image -= np.min(image[:])
        image /= np.max(image[:])
        image = (image * 255).astype(np.uint8)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

        return image, depth
    
    def __del__(self):
        self.pipeline.stop()