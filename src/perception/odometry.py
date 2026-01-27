import numpy as np
import cv2
import logging

class VisualOdometry:
    def __init__(self, fov_h=57.0):
        # We assume standard Kinect intrinsics for 640x480
        # Kinect v1 (Xbox 360) intrinsics
        self.W = 640
        self.H = 480
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Feature Detector (ORB is fast and efficient)
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous Frame State
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None

        # Global Pose (Camera to World)
        # R is rotation matrix (3x3), t is translation vector (3x1)
        self.R_acc = np.eye(3)
        self.t_acc = np.zeros((3, 1))

        # Initializing logger
        self.logger = logging.getLogger("VisualOdometry")

    def update(self, rgb, depth):
        """
        Estimate camera motion.
        Args:
            rgb: Current RGB image (H, W, 3)
            depth: Current Depth image (H, W) in mm (aligned/registered).

        Returns:
            pose: (x, y, z, yaw) in global frame (Camera Start Frame)
        """
        if rgb is None or depth is None:
            return self.get_pose_vector()

        # 1. Feature Detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_kp is None or des is None or len(kp) < 10:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth
            return self.get_pose_vector()

        if self.prev_des is None:
             self.prev_kp = kp
             self.prev_des = des
             self.prev_depth = depth
             return self.get_pose_vector()

        # 2. Match Features
        matches = self.matcher.match(self.prev_des, des)
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep top matches
        matches = matches[:200]

        if len(matches) < 10:
            return self.get_pose_vector()

        # 3. Retrieve 3D points for Previous Features
        pts_3d = []
        pts_2d = []

        for m in matches:
            idx_prev = m.queryIdx
            idx_curr = m.trainIdx

            # Get 2D point in prev
            u_prev, v_prev = self.prev_kp[idx_prev].pt
            u_prev, v_prev = int(u_prev), int(v_prev)

            # Check bounds
            if 0 <= u_prev < self.W and 0 <= v_prev < self.H:
                # Assuming depth is uint16 mm
                d = self.prev_depth[v_prev, u_prev]

                # Check valid depth (0 is invalid in Kinect)
                if d > 0:
                    z = float(d) / 1000.0 # Convert mm to meters

                    x = (u_prev - self.cx) * z / self.fx
                    y = (v_prev - self.cy) * z / self.fy

                    pts_3d.append([x, y, z])
                    pts_2d.append(kp[idx_curr].pt)

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # 4. Solve PnP (Finds pose of Previous Points in Current Frame)
        if len(pts_3d) < 6:
            # Not enough points
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth
            return self.get_pose_vector()

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera_matrix, None,
            iterationsCount=100, reprojectionError=8.0, confidence=0.99
        )

        if success:
            # P_curr = R * P_prev + t
            # Transform from Prev to Curr
            R, _ = cv2.Rodrigues(rvec)

            # We want to update Global Pose: C_new = C_old * M
            # Where M is motion from Prev to Curr.
            # M = inverse of (R, t) because (R, t) transforms points P_prev -> P_curr
            # which is equivalent to World moving relative to Camera.
            # Camera movement is inverse.

            R_inv = R.T
            t_inv = -R_inv @ tvec

            # Update Global Accumulation
            # C_global_new = C_global_prev * T_local
            self.t_acc = self.t_acc + self.R_acc @ t_inv
            self.R_acc = self.R_acc @ R_inv

        # Update Previous
        self.prev_kp = kp
        self.prev_des = des
        self.prev_depth = depth

        return self.get_pose_vector()

    def get_pose_vector(self):
        """
        Returns (x, y, z, yaw) in meters and radians.
        Coordinates are relative to the initial camera frame:
        x: Right
        y: Down
        z: Forward
        yaw: Rotation around Y axis
        """
        x = self.t_acc[0, 0]
        y = self.t_acc[1, 0]
        z = self.t_acc[2, 0]

        # Extract yaw from R_acc
        # Z-axis direction vector
        forward = self.R_acc[:, 2]
        # Project on X-Z plane
        yaw = np.arctan2(forward[0], forward[2])

        return x, y, z, yaw

    def reset(self):
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None
        self.R_acc = np.eye(3)
        self.t_acc = np.zeros((3, 1))
