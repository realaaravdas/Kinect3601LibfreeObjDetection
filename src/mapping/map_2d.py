import numpy as np
import math
from src.perception.geometry import CameraProjector

class MapManager:
    def __init__(self, projector=None):
        if projector is None:
            self.projector = CameraProjector()
        else:
            self.projector = projector

        self.objects = [] # List of dicts
        self.next_id = 0
        self.merge_threshold = 1.0
        self.robot_pose = (0.0, 0.0, 0.0) # x, y, theta

    def update_pose(self, cam_pose):
        if cam_pose is None:
            return

        x_cam, y_cam, z_cam, yaw_cam = cam_pose

        # Camera Frame (Start): X-Right, Y-Down, Z-Forward
        # Map Frame: X-Forward, Y-Left

        map_x = z_cam
        map_y = -x_cam
        map_theta = -yaw_cam

        # Normalize theta to [-pi, pi)
        map_theta = (map_theta + math.pi) % (2 * math.pi) - math.pi

        self.robot_pose = (map_x, map_y, map_theta)

    def update_map(self, detections, depth_frame, W, H):
        if depth_frame is None:
            return

        # 1. Identify objects in frustum and decay
        in_view_indices = []
        for i, obj in enumerate(self.objects):
            if self.projector.is_in_frustum(obj['x'], obj['y'], self.robot_pose):
                in_view_indices.append(i)
                obj['health'] -= 1

        if not detections:
            # Just clean up
            self.objects = [o for o in self.objects if o['health'] > 0]
            return

        # Prepare Depth
        # Assume meters for calculations.
        if depth_frame.dtype == np.uint16:
            depth_m = depth_frame.astype(np.float32) / 1000.0
        else:
            # If already float, assume it is in meters
            depth_m = depth_frame

        # 2. Process Detections
        stride = 2

        for det in detections:
            # det is dict: {'xyxy': [...], 'cls': int, 'conf': float}
            # Handle list or numpy array for xyxy
            bbox = det['xyxy']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cls_id = int(det['cls'])
            conf = float(det['conf'])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cx = max(0, min(cx, W-1))
            cy = max(0, min(cy, H-1))

            # Limit ROI
            rx1 = max(0, x1)
            ry1 = max(0, y1)
            rx2 = min(W, x2)
            ry2 = min(H, y2)

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            # Extract ROI and Stride
            roi = depth_m[ry1:ry2:stride, rx1:rx2:stride]
            valid_depths = roi[roi > 0]

            if valid_depths.size == 0:
                continue

            d_m = np.median(valid_depths)

            if d_m > self.projector.max_depth or d_m < 0.3:
                continue

            # Project
            px, py, pz = self.projector.pixel_to_world(cx, cy, d_m, W, H, self.robot_pose)

            # Match
            matched = False
            for idx in in_view_indices:
                obj = self.objects[idx]
                dist = math.sqrt((obj['x'] - px)**2 + (obj['y'] - py)**2)

                if dist < self.merge_threshold and obj['class_id'] == cls_id:
                    # Match
                    alpha = 0.3
                    obj['x'] = (1.0 - alpha) * obj['x'] + alpha * px
                    obj['y'] = (1.0 - alpha) * obj['y'] + alpha * py
                    obj['z'] = (1.0 - alpha) * obj['z'] + alpha * pz
                    obj['confidence'] = max(obj['confidence'], conf)
                    obj['health'] = min(obj['health'] + 10, 100)
                    matched = True
                    break

            if not matched:
                new_obj = {
                    'id': self.next_id,
                    'class_id': cls_id,
                    'x': px,
                    'y': py,
                    'z': pz,
                    'confidence': conf,
                    'health': 50
                }
                self.next_id += 1
                self.objects.append(new_obj)

        # 3. Cleanup
        self.objects = [o for o in self.objects if o['health'] > 0]

    def get_objects(self):
        return self.objects

    def get_pose(self):
        # Return as numpy array
        return np.array(self.robot_pose)

    def get_frustum(self):
        return self.projector.get_frustum_polygon(self.robot_pose)
