import numpy as np
import cv2
import logging

class VisualOdometry:
    def __init__(self, fov_h=57.0):
        self.fov_h = fov_h
        self.prev_profile = None
        self.search_range = 60 # Max pixel shift to check

    def update(self, depth_frame):
        """
        Estimate rotation change based on depth frame.

        Args:
            depth_frame: Depth image (H, W).

        Returns:
            delta_yaw (float): Rotation change in radians.
        """
        h, w = depth_frame.shape

        # 1. Preprocess: Extract central band
        # Use 20% of image height
        band_h = int(h * 0.2)
        y_start = (h - band_h) // 2
        band = depth_frame[y_start:y_start+band_h, :]

        # 2. Compute Column Profile
        # Convert to float and set 0 to NaN
        band_float = band.astype(np.float32)
        band_float[band_float == 0] = np.nan

        # Calculate mean of columns ignoring NaNs
        with np.errstate(invalid='ignore'):
            profile = np.nanmean(band_float, axis=0)

        # Fill remaining NaNs (empty columns) with nearest valid or 0
        # Simple forward fill then backward fill
        mask = np.isnan(profile)
        if np.all(mask):
            # No valid data in band
            return 0.0

        # Basic imputation: fill with 0 or max range?
        # Let's fill with 0, but during matching 0s might be an issue if using SQDIFF.
        # Better: Linear interpolation.
        x = np.arange(w)
        profile[mask] = np.interp(x[mask], x[~mask], profile[~mask])

        # 3. Match with previous profile
        if self.prev_profile is None:
            self.prev_profile = profile
            return 0.0

        # We assume small rotation between frames.
        # Template: Center part of PREVIOUS profile
        # Source: Full CURRENT profile

        margin = self.search_range
        if w <= 2 * margin:
            # Image too small for this margin
            margin = w // 4

        template = self.prev_profile[margin:-margin]

        # Reshape for matchTemplate (1, N)
        templ_img = template.reshape(1, -1).astype(np.float32)
        curr_img = profile.reshape(1, -1).astype(np.float32)

        # Normalized Cross Correlation might be more robust to depth noise/scale
        res = cv2.matchTemplate(curr_img, templ_img, cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # For TM_CCORR_NORMED, we want MAX value
        best_match_idx = max_loc[0]

        # Calculate shift
        # If perfectly aligned, best_match_idx should be 'margin'
        shift = best_match_idx - margin

        # Logic from thought process:
        # If robot rotates Left (+ Yaw), scene moves Right (shift > 0).
        # So Yaw is positive when shift is positive.

        deg_per_pixel = self.fov_h / w
        delta_deg = shift * deg_per_pixel

        # Update previous profile
        self.prev_profile = profile

        return np.radians(delta_deg)

    def reset(self):
        self.prev_profile = None
