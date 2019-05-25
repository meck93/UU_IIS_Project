import socket
import time
import numpy as np

HOST, PORT = "localhost", 9999

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    sock.sendall(bytes("Hello World\n", "utf-8"))

    while True:
        received = sock.recv(1024)
        x = np.fromstring(received, dtype=np.float32)
        if len(x) != 44:
            time.sleep(0.1)
            continue
        x = x.reshape((22,2))
        print(x)
        # TODO
