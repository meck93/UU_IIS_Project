from face_recognition import main

if __name__ == "__main__":
	# TODO: adjust input files to your local settings
    rgb_file = './datasets/self_created/dataset/rgb/RGB_0200.png'
    depth_file = './datasets/self_created/dataset/depth/D_0200.png'
    opencv_lib_path = "C:/Users/meck/.conda/envs/iis/Library/etc/haarcascades/"

    main(rgb_file, depth_file, opencv_lib_path)
