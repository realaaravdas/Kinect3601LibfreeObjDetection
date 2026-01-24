import numpy as np

class DepthProcessor:
    def __init__(self, fov_h=57.0, fov_v=43.0):
        self.fov_h = fov_h
        self.fov_v = fov_v

    def get_distance_and_angle(self, depth_frame, bbox):
        """
        Calculate distance and angle to the object in the bounding box.

        Args:
            depth_frame: Depth image (unit depends on camera, usually mm).
            bbox: (x1, y1, x2, y2) bounding box.

        Returns:
            distance (float): Mean distance in meters.
            angle (float): Angle in radians relative to center (negative left, positive right).
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to image bounds
        h, w = depth_frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x1 >= x2 or y1 >= y2:
            return None, None

        roi = depth_frame[y1:y2, x1:x2]

        # Filter out 0s (invalid depth)
        valid_pixels = roi[roi > 0]

        if valid_pixels.size == 0:
            return None, None

        # Use median to be robust against outliers
        dist_mm = np.median(valid_pixels)
        dist_m = dist_mm / 1000.0

        # Calculate angle
        center_x = (x1 + x2) / 2.0
        # Offset from image center
        offset_x = center_x - (w / 2.0)

        # angle per pixel
        deg_per_pixel = self.fov_h / w
        angle_deg = offset_x * deg_per_pixel
        angle_rad = np.radians(angle_deg)

        return dist_m, angle_rad
