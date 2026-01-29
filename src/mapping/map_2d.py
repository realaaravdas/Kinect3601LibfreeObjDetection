import numpy as np
import ctypes
import logging
from src.perception.geometry import CameraProjector

class MapManager:
    def __init__(self, projector=None):
        if projector is None:
            self.projector = CameraProjector()
        else:
            self.projector = projector

        self.lib = self.projector.lib

        # Init C++ Map
        self.lib.Map_new.argtypes = [ctypes.c_void_p]
        self.lib.Map_new.restype = ctypes.c_void_p

        self.obj = self.lib.Map_new(self.projector.obj)

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
             self.lib.Map_delete(self.obj)

    def update_pose(self, cam_pose):
        if cam_pose is None:
            return

        x_cam, y_cam, z_cam, yaw_cam = cam_pose

        self.lib.Map_update_pose.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.Map_update_pose(self.obj, x_cam, y_cam, z_cam, yaw_cam)

    def update_map(self, detections, depth_frame, W, H):
        if depth_frame is None:
            return

        # Prepare Detections Array
        # [x1, y1, x2, y2, cls, conf]
        num_dets = len(detections) if detections else 0
        if num_dets > 0:
            det_arr = (ctypes.c_double * (num_dets * 6))()
            for i, det in enumerate(detections):
                det_arr[i*6 + 0] = det['xyxy'][0]
                det_arr[i*6 + 1] = det['xyxy'][1]
                det_arr[i*6 + 2] = det['xyxy'][2]
                det_arr[i*6 + 3] = det['xyxy'][3]
                det_arr[i*6 + 4] = det['cls']
                det_arr[i*6 + 5] = det['conf']
        else:
             det_arr = None

        # Prepare Depth Array
        # Ensure it is contiguous and correct type (uint16)
        if depth_frame.dtype != np.uint16:
            if depth_frame.dtype == np.float32:
                 # Assume meters -> mm
                 depth_uint16 = (depth_frame * 1000).astype(np.uint16)
            else:
                 depth_uint16 = depth_frame.astype(np.uint16)
        else:
             depth_uint16 = depth_frame

        if not depth_uint16.flags['C_CONTIGUOUS']:
            depth_uint16 = np.ascontiguousarray(depth_uint16)

        depth_ptr = depth_uint16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        self.lib.Map_update_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_ushort), ctypes.c_int, ctypes.c_int]

        self.lib.Map_update_map(self.obj, det_arr, num_dets, depth_ptr, W, H)

    def get_objects(self):
        # Fetch objects from C++
        self.lib.Map_get_objects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.lib.Map_get_objects.restype = ctypes.c_int

        max_objs = 100
        buffer = (ctypes.c_double * (max_objs * 7))()

        count = self.lib.Map_get_objects(self.obj, buffer, max_objs)

        objects = []
        for i in range(count):
            obj = {
                'id': int(buffer[i*7 + 0]),
                'class_id': int(buffer[i*7 + 1]),
                'x': buffer[i*7 + 2],
                'y': buffer[i*7 + 3],
                'z': buffer[i*7 + 4],
                'confidence': buffer[i*7 + 5],
                'health': int(buffer[i*7 + 6])
            }
            objects.append(obj)
        return objects

    def get_pose(self):
        # Fetch pose from C++
        self.lib.Map_get_pose.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        pose_arr = (ctypes.c_double * 3)()
        self.lib.Map_get_pose(self.obj, pose_arr)
        return np.array([pose_arr[0], pose_arr[1], pose_arr[2]])

    def get_frustum(self):
        return self.projector.get_frustum_polygon(self.get_pose())
