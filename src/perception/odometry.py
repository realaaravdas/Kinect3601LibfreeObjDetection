import numpy as np
import cv2
import logging

class VisualOdometry:
    def __init__(self, fov_h=57.0, width=640, height=480):
        # Intrinsics
        self.W = width
        self.H = height
        self.fov_h = np.radians(fov_h)

        # Approximate intrinsics based on FOV
        # tan(fov/2) = (W/2) / fx
        # fx = (W/2) / tan(fov/2)

        self.fx = (self.W / 2.0) / np.tan(self.fov_h / 2.0)
        self.fy = self.fx # Assume square pixels usually
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Keyframe State
        self.kf_kp = None
        self.kf_des = None
        self.kf_depth = None
        self.kf_pose = np.eye(4) # Pose of KF in World

        # Current Pose (in World)
        # 4x4 Homogeneous Matrix
        self.current_pose = np.eye(4)

        # Thresholds for new Keyframe
        self.min_trans = 0.1 # meters
        self.min_rot = np.radians(5) # degrees

    def update(self, rgb, depth):
        """
        Update VO.
        Returns:
            (x, y, z, yaw) relative to Start.
        """
        if rgb is None or depth is None:
            return self.get_pose_vector()

        # Ensure dimensions match (in case camera resolution changes dynamically?)
        h, w = depth.shape
        if w != self.W or h != self.H:
             # Just update intrinsics on the fly if needed
             self.W = w
             self.H = h
             self.fx = (self.W / 2.0) / np.tan(self.fov_h / 2.0)
             self.fy = self.fx
             self.cx = self.W / 2.0
             self.cy = self.H / 2.0
             self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ], dtype=np.float32)


        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        # Init First Keyframe
        if self.kf_kp is None:
            self.kf_kp = kp
            self.kf_des = des
            self.kf_depth = depth
            self.kf_pose = np.eye(4)
            self.current_pose = np.eye(4)
            return self.get_pose_vector()

        if len(kp) < 10 or des is None:
            return self.get_pose_vector()

        # Match against KEYFRAME
        matches = self.matcher.match(self.kf_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:300]

        if len(matches) < 10:
            return self.get_pose_vector()

        pts_3d_kf = []
        pts_2d_curr = []

        for m in matches:
            idx_kf = m.queryIdx
            idx_curr = m.trainIdx

            u_kf, v_kf = self.kf_kp[idx_kf].pt
            u_kf, v_kf = int(u_kf), int(v_kf)

            if 0 <= u_kf < self.W and 0 <= v_kf < self.H:
                d = self.kf_depth[v_kf, u_kf]
                if d > 0:
                    z = float(d) / 1000.0
                    x = (u_kf - self.cx) * z / self.fx
                    y = (v_kf - self.cy) * z / self.fy

                    pts_3d_kf.append([x, y, z])
                    pts_2d_curr.append(kp[idx_curr].pt)

        pts_3d_kf = np.array(pts_3d_kf, dtype=np.float32)
        pts_2d_curr = np.array(pts_2d_curr, dtype=np.float32)

        if len(pts_3d_kf) < 6:
            return self.get_pose_vector()

        # Solve PnP: Find Pose of KF points in Current Frame
        # T_curr_kf (Transform from KF to Curr)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_kf, pts_2d_curr, self.camera_matrix, None,
            iterationsCount=100, reprojectionError=6.0, confidence=0.99
        )

        if success:
            R_rel, _ = cv2.Rodrigues(rvec)
            t_rel = tvec

            # T_curr_kf = [R_rel | t_rel]
            # Point_curr = R_rel * Point_kf + t_rel

            # We want Point_world = T_world_curr * Point_curr
            # We know Point_world = T_world_kf * Point_kf
            # So T_world_kf = T_world_curr * T_curr_kf
            # => T_world_curr = T_world_kf * inv(T_curr_kf)

            T_curr_kf = np.eye(4)
            T_curr_kf[:3, :3] = R_rel
            T_curr_kf[:3, 3] = t_rel.flatten()

            T_kf_curr = np.linalg.inv(T_curr_kf)

            self.current_pose = self.kf_pose @ T_kf_curr

            # Check for Keyframe Update
            # Calculate relative motion since KF
            # Just look at T_kf_curr
            delta_trans = np.linalg.norm(T_kf_curr[:3, 3])
            # Rotation angle from trace
            trace = np.trace(T_kf_curr[:3, :3])
            # trace = 1 + 2cos(theta) -> cos(theta) = (tr - 1)/2
            val = (trace - 1) / 2
            val = np.clip(val, -1.0, 1.0)
            delta_rot = np.arccos(val)

            if delta_trans > self.min_trans or delta_rot > self.min_rot:
                # Update Keyframe
                self.kf_kp = kp
                self.kf_des = des
                self.kf_depth = depth
                self.kf_pose = self.current_pose.copy()

        return self.get_pose_vector()

    def get_pose_vector(self):
        """
        Return (x, y, z, yaw) in World Frame.
        (Using standard robotics coordinate convention from camera frame is handled in MapManager mostly,
         but here we return the raw Camera World Pose)
        Camera World Frame:
        X: Right, Y: Down, Z: Forward (relative to Start)
        """
        x = self.current_pose[0, 3]
        y = self.current_pose[1, 3]
        z = self.current_pose[2, 3]

        # Yaw is rotation around Y-axis (since Y is down, and we move on X-Z plane)
        # Vector Z_local projected on X-Z plane
        # Z_local in World = R_world_curr * [0,0,1]^T = Col 2 of R
        forward = self.current_pose[:3, 2]
        yaw = np.arctan2(forward[0], forward[2])

        return x, y, z, yaw

    def reset(self):
        self.kf_kp = None
        self.current_pose = np.eye(4)
