import numpy as np
import logging
from src.perception.geometry import CameraProjector

class MapManager:
    def __init__(self, projector=None):
        # Robot Pose: x, y, theta
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.objects = []
        self.next_id = 0
        self.merge_threshold = 1.0 # meters (increased for robustness)
        self.projector = projector if projector else CameraProjector()

    def update_pose(self, cam_pose):
        """
        Update robot pose from VO.
        """
        if cam_pose is None:
            return

        x_cam, y_cam, z_cam, yaw_cam = cam_pose
        # Camera Frame (Start): X-Right, Y-Down, Z-Forward
        # Map Frame: X-Forward, Y-Left
        map_x = z_cam
        map_y = -x_cam
        map_theta = -yaw_cam

        self.robot_pose = np.array([map_x, map_y, map_theta])
        self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

    def update_map(self, detections, depth_frame, W, H):
        """
        Process detections and update map.
        Args:
            detections: List of dicts {'xyxy', 'cls', 'conf', 'id'}
            depth_frame: Depth image (mm)
            W, H: Dimensions
        """
        if depth_frame is None:
            return

        # 1. Decay Confidence of Unseen Objects
        # Identify which objects are in Frustum but NOT matched
        in_view_indices = []
        frustum = self.projector.get_frustum_polygon(self.robot_pose)

        for i, obj in enumerate(self.objects):
            if self.projector.is_in_frustum(obj['x'], obj['y'], self.robot_pose):
                in_view_indices.append(i)
                obj['health'] -= 2 # Decay faster

        # 2. Process New Detections
        if detections:
            for det in detections:
                box = det['xyxy']
                cls = det['cls']
                conf = det['conf']
                track_id = det.get('id', -1)

                # Get Depth
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cx = np.clip(cx, 0, W-1)
                cy = np.clip(cy, 0, H-1)

                # ROI depth
                roi = depth_frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                valid = roi[roi > 0]
                if valid.size == 0:
                    continue

                d_mm = np.median(valid)
                d_m = d_mm / 1000.0

                if d_m > self.projector.max_depth or d_m < 0.3:
                    continue

                # Project to World
                wx, wy, wz = self.projector.pixel_to_world(cx, cy, d_m, W, H, self.robot_pose)

                # Update/Merge
                matched = False

                # Try to match by Track ID first
                if track_id != -1:
                    for idx in in_view_indices:
                        obj = self.objects[idx]
                        if obj.get('track_id') == track_id:
                            self._update_object(obj, wx, wy, wz, conf)
                            matched = True
                            break

                # If no ID match, spatial match
                if not matched:
                    for idx in in_view_indices:
                        obj = self.objects[idx]
                        dist = np.sqrt((obj['x'] - wx)**2 + (obj['y'] - wy)**2)

                        if dist < self.merge_threshold and obj['class_id'] == cls:
                            self._update_object(obj, wx, wy, wz, conf)
                            # Associate ID if new
                            if track_id != -1:
                                obj['track_id'] = track_id
                            matched = True
                            break

                if not matched:
                    # Create new object
                    self.objects.append({
                        'id': self.next_id,
                        'track_id': track_id,
                        'class_id': cls,
                        'x': wx,
                        'y': wy,
                        'z': wz,
                        'confidence': conf,
                        'health': 50 # Start health
                    })
                    self.next_id += 1

        # 3. Cleanup Dead Objects
        self.objects = [o for o in self.objects if o['health'] > 0]

    def _update_object(self, obj, wx, wy, wz, conf):
        # Weighted Smooth Update (Exponential Moving Average)
        # Low alpha = smoother but more lag
        alpha = 0.2
        obj['x'] = (1-alpha)*obj['x'] + alpha*wx
        obj['y'] = (1-alpha)*obj['y'] + alpha*wy
        obj['z'] = (1-alpha)*obj.get('z', 0) + alpha*wz
        obj['confidence'] = max(obj['confidence'], conf)
        obj['health'] = min(obj['health'] + 5, 100) # Heal

    def get_objects(self):
        return self.objects

    def get_pose(self):
        return self.robot_pose

    def get_frustum(self):
        return self.projector.get_frustum_polygon(self.robot_pose)
