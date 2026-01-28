import numpy as np

class CameraProjector:
    def __init__(self, fov_h=57.0, fov_v=43.0, cam_height=1.0, tilt_angle=0.0):
        """
        Handles 2D -> 3D projection and Frustum checks.
        Args:
            fov_h: Horizontal FOV in degrees.
            fov_v: Vertical FOV in degrees.
            cam_height: Height of camera from ground in meters.
            tilt_angle: Tilt angle in degrees (positive up, negative down).
        """
        self.fov_h = np.radians(fov_h)
        self.fov_v = np.radians(fov_v)
        self.height = cam_height
        self.tilt = np.radians(tilt_angle)

        # Max reliable depth for mapping
        self.max_depth = 8.0 # meters

    def update_config(self, height, tilt):
        self.height = height
        self.tilt = np.radians(tilt)

    def pixel_to_world(self, u, v, depth_m, W, H, robot_pose):
        """
        Project pixel to World Frame (Ground Plane).
        Args:
            u, v: Pixel coordinates.
            depth_m: Depth in meters.
            W, H: Image dimensions.
            robot_pose: (x, y, theta) of robot.
        Returns:
            (gx, gy, gz): Point in Global Map Frame.
                          gx, gy are ground coordinates.
                          gz is height above ground (should be ~0 for objects on floor).
        """
        # 1. Camera Intrinsics (approximated from FOV)
        fx = W / (2 * np.tan(self.fov_h / 2))
        fy = H / (2 * np.tan(self.fov_v / 2))
        cx = W / 2
        cy = H / 2

        # 2. Project to Camera Frame (Right, Down, Forward)
        # z_c is forward
        z_c = depth_m
        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy

        point_cam = np.array([x_c, y_c, z_c])

        # 3. Rotate for Tilt (Pitch around X-axis)
        # Camera is tilted. If Tilt > 0 (Up), Z_cam points up.
        # We need to transform from Tilted Frame to Level Frame.
        # Rot_tilt = Rotation of -Tilt around X.
        # R_x(-t)
        c_t = np.cos(-self.tilt)
        s_t = np.sin(-self.tilt)
        Rx = np.array([
            [1, 0, 0],
            [0, c_t, -s_t],
            [0, s_t, c_t]
        ])

        point_level = Rx @ point_cam

        # 4. Translate for Height (Camera is at +h in Y-down? No, usually +h in Z-up).
        # Let's define Level Frame: X-Right, Y-Down, Z-Forward (Standard CV).
        # Ground is at Y = height. (Since Y is down).
        # Or standard Robotics: X-Forward, Y-Left, Z-Up.
        # Let's stick to Map Frame definition:
        # Robot Frame: X-Forward, Y-Left, Z-Up.
        # Camera is mounted at (0, 0, h) in Robot Frame.
        # Camera Optical Axis aligns with X-Forward (if tilt=0).
        # But Camera Optical Frame is: X-Right, Y-Down, Z-Forward.
        # Transform Camera Optical -> Robot Frame:
        # Cam Z (Forward) -> Robot X (Forward)
        # Cam X (Right)   -> Robot Y (Left) * -1 (Right is -Y)
        # Cam Y (Down)    -> Robot Z (Up)   * -1 (Down is -Z)

        # Re-doing step 3+4 in Robot Frame context.

        # Point in Optical Frame: P_opt = [x_c, y_c, z_c]

        # Apply Tilt (Pitch up = Rotate camera around -Y_robot / +X_opt).
        # Actually easier to think:
        # P_robot = R_opt_to_rob @ R_tilt @ P_opt + Offset

        # Standard Optical to Robot (No Tilt):
        # X_rob = Z_opt
        # Y_rob = -X_opt
        # Z_rob = -Y_opt
        R_o2r = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

        # Tilt correction:
        # If camera is tilted UP by theta:
        # The optical vector needs to be pitched DOWN to match robot horizon?
        # No, if Cam is tilted UP, a point in center of image (0,0,z) is actually pointing UP in robot frame.
        # So we rotate P_opt by Pitch=Theta around X_opt (Right).
        # Then convert to Robot.

        c = np.cos(self.tilt)
        s = np.sin(self.tilt)
        # Rotation around X-axis (Right)
        R_tilt = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        P_tilted = R_tilt @ point_cam

        # Convert to Robot Frame (Centered at Ground Projection)
        P_robot = R_o2r @ P_tilted
        P_robot[2] += self.height # Add height to Z

        # 5. Transform to World Map Frame
        # Robot Pose: (rx, ry, rtheta)
        # World Frame: X-East, Y-North (or whatever).
        # Standard 2D rotation.

        rx, ry, rtheta = robot_pose
        cr = np.cos(rtheta)
        sr = np.sin(rtheta)

        R_map = np.array([
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1]
        ])

        P_world = R_map @ P_robot
        P_world[0] += rx
        P_world[1] += ry

        return P_world[0], P_world[1], P_world[2]

    def get_frustum_polygon(self, robot_pose, max_dist=None):
        """
        Get 2D polygon of camera FOV on ground.
        Args:
            robot_pose: (x, y, theta)
        Returns:
            points: List of (x, y) tuples.
        """
        if max_dist is None:
            max_dist = self.max_depth

        rx, ry, theta = robot_pose

        # FOV half angle
        alpha = self.fov_h / 2.0

        # Left Ray
        lx = rx + max_dist * np.cos(theta + alpha)
        ly = ry + max_dist * np.sin(theta + alpha)

        # Right Ray
        rx_pt = rx + max_dist * np.cos(theta - alpha)
        ry_pt = ry + max_dist * np.sin(theta - alpha)

        return [(rx, ry), (lx, ly), (rx_pt, ry_pt)]

    def is_in_frustum(self, obj_x, obj_y, robot_pose):
        """
        Check if object is within FOV and range.
        """
        rx, ry, theta = robot_pose

        dx = obj_x - rx
        dy = obj_y - ry
        dist = np.sqrt(dx*dx + dy*dy)

        if dist > self.max_depth:
            return False

        # Check angle
        # Angle to object
        obj_angle = np.arctan2(dy, dx)

        # Diff
        diff = obj_angle - theta
        # Normalize
        diff = (diff + np.pi) % (2*np.pi) - np.pi

        if abs(diff) < (self.fov_h / 2.0):
            return True

        return False
