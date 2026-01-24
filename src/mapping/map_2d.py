import numpy as np
import logging

class MapManager:
    def __init__(self):
        # Robot Pose: x, y, theta (radians)
        self.robot_pose = np.array([0.0, 0.0, 0.0])

        # Objects: List of {'id', 'class_id', 'x', 'y', 'confidence', 'hits'}
        self.objects = []
        self.next_id = 0
        self.merge_threshold = 0.5 # meters

    def update_pose(self, delta_yaw):
        """
        Update robot orientation based on visual odometry.
        Args:
            delta_yaw (float): Change in yaw in radians.
        """
        self.robot_pose[2] += delta_yaw
        # Normalize angle to [-pi, pi]
        self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

    def add_observation(self, class_id, dist, angle, confidence):
        """
        Add a new object observation.
        Args:
            class_id (int): Class ID of the object.
            dist (float): Distance to object in meters.
            angle (float): Angle to object relative to camera center (radians).
            confidence (float): Detection confidence.
        """
        # Calculate global position
        # Robot theta + object relative angle
        theta = self.robot_pose[2] + angle

        # Map coordinates
        # x is forward? Let's assume standard robotics:
        # x forward, y left.
        # But if robot starts at (0,0) facing 0 (East/Right in plots usually).
        # Let's assume standard 2D cartesian.

        gx = self.robot_pose[0] + dist * np.cos(theta)
        gy = self.robot_pose[1] + dist * np.sin(theta)

        # Check for existing objects to merge
        # We only merge if class_id matches and distance is small
        found = False
        for obj in self.objects:
            if obj['class_id'] == class_id:
                d = np.sqrt((obj['x'] - gx)**2 + (obj['y'] - gy)**2)
                if d < self.merge_threshold:
                    # Update existing using weighted average?
                    # For simplicity, simple moving average or just replace.
                    # Using moving average helps reduce noise.
                    n = obj['hits']
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
            logging.info(f"New object detected at ({gx:.2f}, {gy:.2f})")

    def get_objects(self):
        return self.objects

    def get_pose(self):
        return self.robot_pose
