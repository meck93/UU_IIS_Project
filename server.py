import socketserver

import numpy as np

from faceDetection import detectFaces
from landmarkDetection import LandmarkDetector
from sources.camera import RealSenseCam
from sources.dataset import Dataset
from visualization import initVisualization, updateVisualization, visualize


class MyTCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.main()

    def main(self):
        source = RealSenseCam()  # for camera use RealsenseCam()

        landmarkDetector = LandmarkDetector()

        vis = initVisualization()

        while True:
            image, depth = source.getFrame()

            faces = detectFaces(image, depth)

            landmarks = landmarkDetector.detectLandmarks(faces)

            for f, l in zip(faces, landmarks):
                updateVisualization(f, l, vis)

                # add depth information
                landmarks = l.copy().reshape((22,2))
                data = []
                for x, y in landmarks:
                    x_ = min(int(round(x*128)), 128)
                    y_ = min(int(round(y*128)), 128)
                    z = f[1,x_,y_]
                    data.append([x, y, z])
                data = np.asarray(data, dtype=np.float32)
                print(data)
                self.wfile.write(data.tostring()) # send message


if __name__ == "__main__":
    HOST, PORT = "192.168.43.156", 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        server.serve_forever()
