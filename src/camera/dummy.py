import numpy as np
import time
from .interface import KinectInterface
import logging

class DummyCamera(KinectInterface):
    def __init__(self):
        logging.info("Initialized Dummy Camera")
        self.tilt = 0

    def open(self):
        logging.info("Dummy Camera Opened")

    def close(self):
        logging.info("Dummy Camera Closed")

    def get_frame(self):
        # Generate fake data
        # RGB: 640x480 noise
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Depth: Gradient
        x = np.linspace(0, 1, 640)
        y = np.linspace(0, 1, 480)
        xv, yv = np.meshgrid(x, y)
        depth = (xv * 1000 + 500).astype(np.float32) # 500mm to 1500mm

        time.sleep(0.03) # ~30fps
        return rgb, depth

    def set_tilt(self, angle):
        self.tilt = angle
        logging.info(f"Dummy Camera Tilt set to {angle}")
