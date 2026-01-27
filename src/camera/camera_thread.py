import threading
import time
import logging
import numpy as np

class CameraThread(threading.Thread):
    def __init__(self, camera):
        """
        Threaded wrapper for camera capture.
        Args:
            camera: An instance of KinectInterface (e.g. Kinect360, KinectOne, DummyCamera)
        """
        super().__init__()
        self.camera = camera
        self.latest_rgb = None
        self.latest_depth = None
        self.running = False
        self.lock = threading.Lock()
        self.daemon = True # Allow main program to exit even if this thread is running

    def run(self):
        logging.info("CameraThread: Starting capture loop...")
        self.running = True
        while self.running:
            try:
                # This call might block depending on driver implementation
                rgb, depth = self.camera.get_frame()

                if rgb is not None and depth is not None:
                    with self.lock:
                        # Store references. get_frame usually returns new arrays.
                        self.latest_rgb = rgb
                        self.latest_depth = depth
                else:
                    # Small sleep if no frame to prevent busy loop on errors
                    time.sleep(0.005)
            except Exception as e:
                logging.error(f"CameraThread: Error capturing frame: {e}")
                time.sleep(0.1)

        logging.info("CameraThread: Stopped.")

    def stop(self):
        self.running = False
        # We don't join here to allow non-blocking stop request,
        # but usually join is called by main.

    def get_latest_frame(self):
        """
        Returns the most recent frame.
        Returns:
            (rgb, depth): Tuple of numpy arrays or (None, None)
        """
        with self.lock:
            if self.latest_rgb is None:
                return None, None
            # Return references. Consumer should copy if modification is needed.
            return self.latest_rgb, self.latest_depth
