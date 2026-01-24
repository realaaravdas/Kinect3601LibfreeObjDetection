from abc import ABC, abstractmethod
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KinectDevice(ABC):
    """Abstract base class for Kinect devices."""

    def __init__(self):
        self.tilt_angle = 0.0

    @abstractmethod
    def get_frames(self):
        """
        Returns the latest RGB and Depth frames.

        Returns:
            tuple: (rgb_frame, depth_frame) where:
                rgb_frame is a numpy array (H, W, 3)
                depth_frame is a numpy array (H, W) or (H, W, 1) in mm or meters
        """
        pass

    @abstractmethod
    def set_tilt(self, angle):
        """
        Sets the tilt angle of the device (if supported).

        Args:
            angle (float): Angle in degrees.
        """
        pass

    @abstractmethod
    def get_tilt(self):
        """
        Gets the current tilt angle.

        Returns:
            float: Angle in degrees.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops the device."""
        pass

    def startup_sequence(self):
        """
        Performs the startup sequence: Tilt Up -> Tilt Down -> Center.
        Only applicable for devices with motors (Kinect 360).
        """
        logger.info("Starting tilt sequence...")
        self.set_tilt(30)
        time.sleep(3) # Give time to move
        self.set_tilt(-30)
        time.sleep(4) # Full range move
        self.set_tilt(0)
        time.sleep(2)
        logger.info("Tilt sequence complete. Centered at 0 degrees.")


class Freenect1Device(KinectDevice):
    def __init__(self):
        super().__init__()
        try:
            import freenect
            self.freenect = freenect
        except ImportError:
            logger.error("freenect module not found. Please install libfreenect.")
            raise

        self.ctx = self.freenect.init()
        self.dev = self.freenect.open_device(self.ctx, 0)

        if not self.dev:
            raise RuntimeError("Could not open Kinect 360 device")

        self.freenect.set_led(self.dev, self.freenect.LED_GREEN)
        self.freenect.set_depth_mode(self.dev, self.freenect.RESOLUTION_MEDIUM, self.freenect.DEPTH_REGISTERED)
        self.freenect.set_video_mode(self.dev, self.freenect.RESOLUTION_MEDIUM, self.freenect.VIDEO_RGB)

    def get_frames(self):
        rgb, _ = self.freenect.sync_get_video()
        depth, _ = self.freenect.sync_get_depth()
        return rgb, depth

    def set_tilt(self, angle):
        self.freenect.set_tilt_degs(self.dev, angle)
        self.tilt_angle = angle

    def get_tilt(self):
        return self.tilt_angle

    def stop(self):
        self.freenect.close_device(self.dev)
        self.freenect.shutdown(self.ctx)


class Freenect2Device(KinectDevice):
    def __init__(self):
        super().__init__()
        try:
            from pylibfreenect2 import Freenect2, SyncMultiFrameListener
            from pylibfreenect2 import FrameType, Registration, Frame
            self.pylibfreenect2 = locals()
        except ImportError:
            logger.error("pylibfreenect2 module not found.")
            raise

        self.fn2 = self.pylibfreenect2.Freenect2()
        self.num_devices = self.fn2.enumerateDevices()
        if self.num_devices == 0:
            raise RuntimeError("No Kinect v2 devices found")

        self.serial = self.fn2.getDefaultDeviceSerialNumber()
        self.dev = self.fn2.openDevice(self.serial)

        self.listener = self.pylibfreenect2.SyncMultiFrameListener(
            self.pylibfreenect2.FrameType.Color | self.pylibfreenect2.FrameType.Ir | self.pylibfreenect2.FrameType.Depth
        )
        self.dev.setColorFrameListener(self.listener)
        self.dev.setIrAndDepthFrameListener(self.listener)

        self.dev.start()

        self.registration = self.pylibfreenect2.Registration(self.dev.getIrCameraParams(), self.dev.getColorCameraParams())

    def get_frames(self):
        frames = self.listener.waitForNewFrame()
        color = frames["color"]
        depth = frames["depth"]

        undistorted = self.pylibfreenect2.Frame(512, 424, 4)
        registered = self.pylibfreenect2.Frame(512, 424, 4)

        self.registration.apply(color, depth, undistorted, registered)

        rgb_data = registered.asarray(dtype=np.uint8) # 512x424x4 (BGRA)
        depth_data = undistorted.asarray(dtype=np.float32) # 512x424

        # Convert BGRA to RGB
        rgb_data = rgb_data[:, :, :3][:, :, ::-1]

        self.listener.release(frames)

        return rgb_data, depth_data

    def set_tilt(self, angle):
        logger.warning("Kinect v2 does not support tilt control.")
        pass

    def get_tilt(self):
        return 0.0

    def stop(self):
        self.dev.stop()
        self.dev.close()

class MockKinectDevice(KinectDevice):
    def __init__(self):
        super().__init__()
        logger.info("MockKinectDevice initialized.")
        self.tilt_angle = 0.0

    def get_frames(self):
        h, w = 480, 640
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Add noise
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        rgb = cv2.add(rgb, noise) if 'cv2' in globals() else rgb

        # Make a yellow ball in the center
        try:
            import cv2
            # RGB Yellow is (255, 255, 0)
            cv2.circle(rgb, (320, 240), 50, (255, 255, 0), -1)
        except ImportError:
            rgb[190:290, 270:370] = [255, 255, 0] # Manual yellow square

        depth = np.indices((h, w))[1].astype(np.float32) * 10 # Gradient in mm

        return rgb, depth

    def set_tilt(self, angle):
        logger.info(f"Mock Tilt set to {angle}")
        self.tilt_angle = angle

    def get_tilt(self):
        return self.tilt_angle

    def stop(self):
        logger.info("Mock Device stopped.")
