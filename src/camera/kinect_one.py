import numpy as np
from .interface import KinectInterface
import logging

class KinectOne(KinectInterface):
    def __init__(self):
        self.device = None
        self.fn = None
        self.listener = None
        try:
            from pylibfreenect2 import Freenect2, SyncMultiFrameListener
            from pylibfreenect2 import FrameType, Registration
            self.pylibfreenect2 = __import__('pylibfreenect2')
            self.Freenect2 = Freenect2
            self.SyncMultiFrameListener = SyncMultiFrameListener
            self.FrameType = FrameType
        except ImportError:
            self.pylibfreenect2 = None
            logging.error("pylibfreenect2 module not found. Please install pylibfreenect2.")

    def open(self):
        if self.pylibfreenect2 is None:
            raise RuntimeError("pylibfreenect2 not available")

        self.fn = self.Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            raise RuntimeError("No Kinect v2 devices found")

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial)

        self.listener = self.SyncMultiFrameListener(
            self.FrameType.Color | self.FrameType.Depth
        )

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        self.device.start()
        logging.info(f"Kinect v2 device {serial} opened.")

    def close(self):
        if self.device:
            self.device.stop()
            self.device.close()

    def get_frame(self):
        if self.device is None:
            return None, None

        frames = self.listener.waitForNewFrame()

        color_frame = frames["color"]
        depth_frame = frames["depth"]

        # Color is BGRA usually, 1920x1080
        rgb = color_frame.asarray() # This might be RGBA or BGRA, usually BGRA.
        # Remove alpha channel and convert to RGB if needed.
        # Assuming we want RGB.
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3] # Drop Alpha

        # Depth is 512x424 float32 (mm)
        depth = depth_frame.asarray()

        # Important: You must release the frames!
        self.listener.release(frames)

        return rgb, depth

    def set_tilt(self, angle):
        logging.warning("Kinect v2 does not support motorized tilt control.")
