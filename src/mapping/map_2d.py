import numpy as np
import logging

class MapManager:
    def __init__(self):
        # Robot Pose in Map Frame: x (forward/North), y (left/West), theta (radians, +Left/CCW)
        # Initial pose is (0, 0, 0)
        self.robot_pose = np.array([0.0, 0.0, 0.0])

        # Objects: List of {'id', 'class_id', 'x', 'y', 'confidence', 'hits'}
        self.objects = []
        self.next_id = 0
        self.merge_threshold = 0.5 # meters

    def update_pose(self, cam_pose):
        """
        Update robot pose based on Visual Odometry (Camera Frame).
        Args:
            cam_pose: (x, y, z, yaw) in Camera Frame (Initial).
                      X (Right), Y (Down), Z (Forward). Yaw +Right around Y-down.
        """
        if cam_pose is None:
            return

        x_cam, y_cam, z_cam, yaw_cam = cam_pose

        # Coordinate Transform to Map Frame (X Forward, Y Left, Theta +Left)
        # We align Map Initial Frame with Camera Initial Frame
        map_x = z_cam
        map_y = -x_cam
        map_theta = -yaw_cam

        self.robot_pose = np.array([map_x, map_y, map_theta])

        # Normalize angle to [-pi, pi]
        self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

    def add_observation(self, class_id, dist, angle, confidence):
        """
        Add a new object observation.
        Args:
            class_id (int): Class ID of the object.
            dist (float): Distance to object in meters.
            angle (float): Angle to object relative to camera center (radians).
                          Positive is Right (based on DepthProcessor).
        """
        # Robot global angle (Theta +Left)
        theta = self.robot_pose[2]

        # Object global angle
        # Since 'angle' is +Right, we subtract it from Theta (+Left)
        # to get angle in Map Frame relative to X-axis.
        # Example: Robot at 0 deg (facing X). Object at +30 deg Right.
        # Object Global Angle = 0 - 30 = -30 deg. Correct.
        obj_theta = theta - angle

        gx = self.robot_pose[0] + dist * np.cos(obj_theta)
        gy = self.robot_pose[1] + dist * np.sin(obj_theta)

        # Check for existing objects to merge
        found = False
        for obj in self.objects:
            if obj['class_id'] == class_id:
                d = np.sqrt((obj['x'] - gx)**2 + (obj['y'] - gy)**2)
                if d < self.merge_threshold:
                    n = obj['hits']
                    # Weighted average
                    obj['x'] = (obj['x'] * n + gx) / (n + 1)
                    obj['y'] = (obj['y'] * n + gy) / (n + 1)
                    obj['hits'] += 1
                    obj['confidence'] = max(obj['confidence'], confidence)
                    found = True
                    break

        if not found:
            self.objects.append({
                'id': self.next_id,
                'class_id': class_id,
                'x': gx,
                'y': gy,
                'confidence': confidence,
                'hits': 1
            })
            self.next_id += 1

    def get_objects(self):
        return self.objects

    def get_pose(self):
        return self.robot_pose
