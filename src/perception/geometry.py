import ctypes
import numpy as np
import os
import logging

class CameraProjector:
    def __init__(self, fov_h=57.0, fov_v=43.0, cam_height=1.0, tilt_angle=0.0):
        self.lib = self._load_library()

        # Init C++ Object
        self.lib.Projector_new.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.Projector_new.restype = ctypes.c_void_p

        self.obj = self.lib.Projector_new(fov_h, fov_v, cam_height, tilt_angle)

        self.fov_h_deg = fov_h
        self.height = cam_height
        self.tilt = tilt_angle
        # Max reliable depth for mapping (mirrors C++ default)
        self.max_depth = 8.0

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
             self.lib.Projector_delete(self.obj)

    def _load_library(self):
        # Path relative to this file: ../../src/cpp/build/libperception.so
        # This file is in src/perception/geometry.py
        lib_path = os.path.join(os.path.dirname(__file__), '../../src/cpp/build/libperception.so')
        lib_path = os.path.abspath(lib_path)
        if not os.path.exists(lib_path):
             # Fallback or error?
             logging.error(f"C++ Library not found at {lib_path}")
        return ctypes.CDLL(lib_path)

    def update_config(self, height, tilt):
        self.lib.Projector_update_config.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self.lib.Projector_update_config(self.obj, height, tilt)
        self.height = height
        self.tilt = tilt

    def pixel_to_world(self, u, v, depth_m, W, H, robot_pose):
        self.lib.Projector_pixel_to_world.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                      ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        rx, ry, rtheta = robot_pose
        ox = ctypes.c_double()
        oy = ctypes.c_double()
        oz = ctypes.c_double()

        self.lib.Projector_pixel_to_world(self.obj, u, v, depth_m, W, H, rx, ry, rtheta,
                                          ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))

        return ox.value, oy.value, oz.value

    def is_in_frustum(self, obj_x, obj_y, robot_pose):
        self.lib.Projector_is_in_frustum.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                                     ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.Projector_is_in_frustum.restype = ctypes.c_bool

        rx, ry, rtheta = robot_pose
        return self.lib.Projector_is_in_frustum(self.obj, obj_x, obj_y, rx, ry, rtheta)

    def get_frustum_polygon(self, robot_pose):
        self.lib.Projector_get_frustum.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                   ctypes.POINTER(ctypes.c_double)]

        rx, ry, rtheta = robot_pose
        # 3 points * 2 coords = 6 doubles
        out_arr = (ctypes.c_double * 6)()
        self.lib.Projector_get_frustum(self.obj, rx, ry, rtheta, out_arr)

        return [(out_arr[0], out_arr[1]), (out_arr[2], out_arr[3]), (out_arr[4], out_arr[5])]
