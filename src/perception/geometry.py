import math

class CameraProjector:
    def __init__(self, fov_h=57.0, fov_v=43.0, cam_height=1.0, tilt_angle=0.0):
        self.fov_h_rad = math.radians(fov_h)
        self.fov_v_rad = math.radians(fov_v)
        self.cam_height = cam_height
        self.tilt_rad = math.radians(tilt_angle)
        self.max_depth = 8.0

    def update_config(self, height, tilt):
        self.cam_height = height
        self.tilt_rad = math.radians(tilt)

    def pixel_to_world(self, u, v, depth_m, W, H, robot_pose):
        # 1. Intrinsics
        # Avoid division by zero if fov is 0 (unlikely but safe to check?)
        # Standard usage assumes non-zero FOV.
        fx = W / (2.0 * math.tan(self.fov_h_rad / 2.0))
        fy = H / (2.0 * math.tan(self.fov_v_rad / 2.0))
        cx = W / 2.0
        cy = H / 2.0

        # 2. Camera Frame (x_c, y_c, z_c)
        z_c = depth_m
        x_c = (u - cx) * z_c / fx
        y_c = (v - cy) * z_c / fy

        # 3. Tilt Rotation (Around X-axis of Opt Frame)
        c = math.cos(self.tilt_rad)
        s = math.sin(self.tilt_rad)

        # P_tilted = R_tilt * P_cam
        # Rot around X:
        # [1  0  0]
        # [0  c -s]
        # [0  s  c]
        x_t = x_c
        y_t = y_c * c - z_c * s
        z_t = y_c * s + z_c * c

        # 4. To Robot Frame
        # X_rob = Z_opt_tilted
        # Y_rob = -X_opt_tilted
        # Z_rob = -Y_opt_tilted + height
        x_r = z_t
        y_r = -x_t
        z_r = -y_t + self.cam_height

        # 5. To World Frame
        # Robot Pose (rx, ry, rtheta)
        # Rot around Z:
        # [cr -sr  0]
        # [sr  cr  0]
        # [ 0   0  1]
        rx, ry, rtheta = robot_pose
        cr = math.cos(rtheta)
        sr = math.sin(rtheta)

        x_w = x_r * cr - y_r * sr + rx
        y_w = x_r * sr + y_r * cr + ry
        z_w = z_r

        return x_w, y_w, z_w

    def is_in_frustum(self, obj_x, obj_y, robot_pose):
        rx, ry, rtheta = robot_pose
        dx = obj_x - rx
        dy = obj_y - ry
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > self.max_depth:
            return False

        obj_angle = math.atan2(dy, dx)
        diff = obj_angle - rtheta

        # Normalize -PI to PI
        while diff > math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi

        return abs(diff) < (self.fov_h_rad / 2.0)

    def get_frustum_polygon(self, robot_pose):
        rx, ry, rtheta = robot_pose
        alpha = self.fov_h_rad / 2.0

        p1 = (rx, ry)

        lx = rx + self.max_depth * math.cos(rtheta + alpha)
        ly = ry + self.max_depth * math.sin(rtheta + alpha)
        p2 = (lx, ly)

        rx_pt = rx + self.max_depth * math.cos(rtheta - alpha)
        ry_pt = ry + self.max_depth * math.sin(rtheta - alpha)
        p3 = (rx_pt, ry_pt)

        return [p1, p2, p3]
