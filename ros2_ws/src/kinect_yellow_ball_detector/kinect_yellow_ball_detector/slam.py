import numpy as np
import cv2
import logging

class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None
        self.focal_length = 525.0
        self.cx = 320.0
        self.cy = 240.0
        self.logger = logging.getLogger(__name__)

    def reset(self):
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None

    def process_frame(self, rgb, depth):
        """
        Estimates the rotation change (yaw) since the last frame.
        Args:
            rgb: RGB image
            depth: Depth image (aligned to RGB, values in mm or m)
        Returns:
            float: yaw change in degrees
        """
        kp, des = self.orb.detectAndCompute(rgb, None)

        if self.prev_kp is None or des is None or len(kp) < 10:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth
            return 0.0

        if self.prev_des is None:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth
            return 0.0

        matches = self.bf.match(self.prev_des, des)
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Take top matches
        good_matches = matches[:100]

        src_pts = []
        dst_pts = []

        for m in good_matches:
            p1 = self.prev_kp[m.queryIdx].pt
            p2 = kp[m.trainIdx].pt

            u1, v1 = int(p1[0]), int(p1[1])
            u2, v2 = int(p2[0]), int(p2[1])

            # Check bounds and valid depth
            if (0 <= v1 < depth.shape[0] and 0 <= u1 < depth.shape[1] and
                0 <= v2 < depth.shape[0] and 0 <= u2 < depth.shape[1]):

                d1 = float(self.prev_depth[v1, u1])
                d2 = float(depth[v2, u2])

                # Filter bad depth
                if d1 > 0 and d2 > 0 and not np.isnan(d1) and not np.isnan(d2):
                    P1 = np.array([
                        (u1 - self.cx) * d1 / self.focal_length,
                        (v1 - self.cy) * d1 / self.focal_length,
                        d1
                    ])
                    P2 = np.array([
                        (u2 - self.cx) * d2 / self.focal_length,
                        (v2 - self.cy) * d2 / self.focal_length,
                        d2
                    ])
                    src_pts.append(P1)
                    dst_pts.append(P2)

        # Update previous frame
        self.prev_kp = kp
        self.prev_des = des
        self.prev_depth = depth

        if len(src_pts) < 10:
            return 0.0

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        # Estimate 3D affine transform
        # We really only want rotation, but affine is general enough
        ret, M, inliers = cv2.estimateAffine3D(src_pts, dst_pts)

        if not ret:
            return 0.0

        R = M[:3, :3]

        # Extract yaw (rotation around Y axis in camera frame? Usually Y is down, Z is forward, X is right)
        # Wait, usually camera coords: X right, Y down, Z forward.
        # So "horizontal rotation" is rotation around Y axis.
        # Rotation matrix around Y:
        # [ cos  0  sin]
        # [ 0    1  0  ]
        # [-sin  0  cos]

        # But here we have general rotation. We want the component corresponding to yaw.
        # Let's convert R to Euler angles.
        # Assuming R * P1 = P2.
        # If camera rotates by +theta around Y (pans right), the world points move by -theta.
        # So we are estimating the motion of points relative to camera.
        # Camera rotation = Inverse of Point rotation.

        # Let's extract euler angles from R.
        # sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        # singular = sy < 1e-6
        # if not singular:
        #     x = atan2(R[2,1] , R[2,2])
        #     y = atan2(-R[2,0], sy)
        #     z = atan2(R[1,0], R[0,0])
        # else:
        #     x = atan2(-R[1,2], R[1,1])
        #     y = atan2(-R[2,0], sy)
        #     z = 0

        # However, for Y-axis rotation (pan), we care about the angle that affects X and Z.
        # Yaw = atan2(R[0, 2], R[2, 2]) ??

        # Actually, let's just use the rotation vector.
        rvec, _ = cv2.Rodrigues(R)
        # rvec is [rx, ry, rz].
        # In camera frame:
        # ry is rotation around Y axis (down).
        # We return -degrees(ry[0]).
        # Why negative? If points move left, camera moved right.

        return -np.degrees(rvec[1][0])


class Mapper:
    def __init__(self):
        self.balls = [] # List of {'id': int, 'x': float, 'z': float, 'class': str}
        self.robot_pose = {'x': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.focal_length = 525.0
        self.cx = 320.0
        self.ball_id_counter = 0
        self.match_threshold = 500.0 # mm

    def update_robot_pose(self, yaw_change_deg):
        self.robot_pose['yaw'] += yaw_change_deg
        # Normalize to -180..180
        self.robot_pose['yaw'] = (self.robot_pose['yaw'] + 180) % 360 - 180

    def process_detections(self, detections, depth_frame):
        # detections: list of dicts from detector
        # depth_frame: numpy array

        current_balls = []

        for det in detections:
            box = det['box'] # x1, y1, x2, y2
            cx_box = int((box[0] + box[2]) / 2)
            cy_box = int((box[1] + box[3]) / 2)

            # Get depth at center
            if 0 <= cy_box < depth_frame.shape[0] and 0 <= cx_box < depth_frame.shape[1]:
                d = depth_frame[cy_box, cx_box]

                # If d is 0 (invalid), try to find valid depth in neighborhood
                if d == 0:
                    roi = depth_frame[max(0, cy_box-5):min(depth_frame.shape[0], cy_box+5),
                                      max(0, cx_box-5):min(depth_frame.shape[1], cx_box+5)]
                    valid = roi[roi > 0]
                    if len(valid) > 0:
                        d = np.median(valid)
                    else:
                        continue

                z_c = float(d)
                x_c = (cx_box - self.cx) * z_c / self.focal_length

                # Transform to world frame
                theta = np.radians(self.robot_pose['yaw'])

                # World X (Right), World Z (Forward)
                # w_x = x_c * cos(theta) + z_c * sin(theta) + robot_x
                # w_z = -x_c * sin(theta) + z_c * cos(theta) + robot_z

                w_x = x_c * np.cos(theta) + z_c * np.sin(theta) + self.robot_pose['x']
                w_z = -x_c * np.sin(theta) + z_c * np.cos(theta) + self.robot_pose['z']

                current_balls.append({'x': w_x, 'z': w_z, 'class': det['class_name']})

        # Merge with map
        for ball in current_balls:
            matched = False
            for map_ball in self.balls:
                dist = np.sqrt((ball['x'] - map_ball['x'])**2 + (ball['z'] - map_ball['z'])**2)
                if dist < self.match_threshold:
                    # Update position (moving average)
                    map_ball['x'] = 0.8 * map_ball['x'] + 0.2 * ball['x']
                    map_ball['z'] = 0.8 * map_ball['z'] + 0.2 * ball['z']
                    matched = True
                    break

            if not matched:
                self.balls.append({
                    'id': self.ball_id_counter,
                    'x': ball['x'],
                    'z': ball['z'],
                    'class': ball['class']
                })
                self.ball_id_counter += 1

    def get_map(self):
        return self.balls
