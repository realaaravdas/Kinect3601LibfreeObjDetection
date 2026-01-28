import numpy as np
from .interface import KinectInterface
import logging

class KinectOne(KinectInterface):
    def __init__(self):
        self.device = None
        self.fn = None
        self.listener = None
        self.registration = None
        self.undistorted = None
        self.registered = None

        try:
            from pylibfreenect2 import Freenect2, SyncMultiFrameListener
            from pylibfreenect2 import FrameType, Registration
            self.pylibfreenect2 = __import__('pylibfreenect2')
            self.Freenect2 = Freenect2
            self.SyncMultiFrameListener = SyncMultiFrameListener
            self.FrameType = FrameType
            self.Registration = Registration
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
            self.FrameType.Color | self.FrameType.Ir | self.FrameType.Depth
        )

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        self.device.start()

        # Init Registration
        # We need the IrCameraParams and ColorCameraParams from device
        ir_params = self.device.getIrCameraParams()
        color_params = self.device.getColorCameraParams()

        self.registration = self.Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())

        # Prepare Frame Containers for Registration
        # Undistorted depth frame: 512x424 float32
        # Registered color frame: 512x424 RGB (mapped to depth)
        # OR Registered depth frame: 1920x1080 (mapped to color)

        # We prefer to align Depth to Color (1920x1080) for high res,
        # OR Color to Depth (512x424) for speed.
        # Given "efficiency" request, let's use 512x424 (BigDepth).
        # Actually standard libfreenect2 registration maps depth to color (registered) OR color to depth (bigdepth).
        # Let's align RGB to Depth resolution (512x424) -> 'registered' in apply means something else?

        # From pylibfreenect2 docs/examples:
        # registration.apply(color, depth, undistorted, registered, bigdepth=None, color_depth_map=None)
        # 'registered' is Depth-to-Color mapping (1920x1082) ??
        # No, 'registered' is "Color aligned to Depth geometry" -> 512x424 RGB.
        # This is ideal for processing speed!

        from pylibfreenect2 import Frame
        self.undistorted = Frame(512, 424, 4) # float32
        self.registered = Frame(512, 424, 4) # 4 bytes per pixel (BGRA)

        logging.info(f"Kinect v2 device {serial} opened.")

    def close(self):
        if self.device:
            self.device.stop()
            self.device.close()

    def get_frame(self):
        if self.device is None:
            return None, None

        frames = self.listener.waitForNewFrame()

        color = frames["color"]
        depth = frames["depth"]

        # Apply Registration
        # This maps color on top of depth (512x424)
        self.registration.apply(color, depth, self.undistorted, self.registered)

        # Get RGB from registered (it is BGRA)
        # Note: data is a pointer/buffer

        # Copy to numpy
        # Registered frame is 512x424, 4 bytes (BGRA)
        img_registered = self.registered.asarray(dtype=np.uint8)

        # Convert BGRA to RGB
        # Slicing is faster than cvtColor
        rgb = img_registered[:, :, :3][:, :, ::-1].copy() # BGRA -> BGR -> RGB

        # Get Depth (undistorted)
        # 512x424, float32, mm
        depth_map = self.undistorted.asarray(dtype=np.float32).copy()

        # Release
        self.listener.release(frames)

        return rgb, depth_map

    def set_tilt(self, angle):
        logging.warning("Kinect v2 does not support motorized tilt control.")
