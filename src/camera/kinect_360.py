import time
import numpy as np
from .interface import KinectInterface
import logging

class Kinect360(KinectInterface):
    def __init__(self):
        try:
            from . import kinect_driver as freenect
            self.freenect = freenect
        except ImportError:
            self.freenect = None
            logging.error("kinect_driver module not found or failed to load.")

    def open(self):
        if self.freenect is None:
            raise RuntimeError("freenect not available")
        # Initialize sync? It auto-initializes on first call usually.
        # But we can try to "reset" it by stopping any existing.
        self.freenect.sync_stop()

    def close(self):
        if self.freenect:
            self.freenect.sync_stop()

    def get_frame(self):
        if self.freenect is None:
            return None, None

        try:
            # sync_get_video returns (data, timestamp)
            rgb, _ = self.freenect.sync_get_video()

            if rgb is None:
                return None, None

            # sync_get_depth returns (data, timestamp)
            # Use registered depth for RGB-D alignment (mm)
            depth, _ = self.freenect.sync_get_depth(format=self.freenect.DEPTH_REGISTERED)

            if depth is None:
                return None, None

            return rgb, depth
        except Exception as e:
            logging.error(f"Error getting frame: {e}")
            return None, None

    def set_tilt(self, angle):
        if self.freenect is None:
            return

        # Open a transient context for motor control
        try:
            ctx = self.freenect.init()
            # Open device 0
            dev = self.freenect.open_device(ctx, 0)

            self.freenect.set_tilt_degs(dev, angle)
            # It takes some time for the motor to move
            time.sleep(1)

            self.freenect.close_device(dev)
            self.freenect.shutdown(ctx)
        except Exception as e:
            logging.error(f"Error setting tilt: {e}")

    def startup_sequence(self):
        if self.freenect is None:
            return

        logging.info("Starting Kinect 360 startup sequence...")
        self.set_tilt(30) # All the way up (approx)
        time.sleep(2)
        self.set_tilt(-30) # All the way down
        time.sleep(2)
        self.set_tilt(0) # Flat
        logging.info("Startup sequence complete.")
