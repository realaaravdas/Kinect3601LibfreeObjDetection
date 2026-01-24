from abc import ABC, abstractmethod
import numpy as np

class KinectInterface(ABC):
    """
    Abstract base class for Kinect devices.
    """

    @abstractmethod
    def open(self):
        """
        Open the connection to the device.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the connection to the device.
        """
        pass

    @abstractmethod
    def get_frame(self):
        """
        Get the latest RGB and Depth frames.

        Returns:
            tuple: (rgb_frame, depth_frame)
                   rgb_frame: numpy array (H, W, 3)
                   depth_frame: numpy array (H, W) in millimeters (or device specific unit)
        """
        pass

    @abstractmethod
    def set_tilt(self, angle):
        """
        Set the tilt angle of the camera.

        Args:
            angle (float): Angle in degrees.
        """
        pass
