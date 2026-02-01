import ctypes
import ctypes.util
import numpy as np
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Constants matching libfreenect.h

# LED Options
LED_OFF = 0
LED_GREEN = 1
LED_RED = 2
LED_YELLOW = 3
LED_BLINK_GREEN = 4
LED_BLINK_RED_YELLOW = 6

# Video Formats
VIDEO_RGB = 0
VIDEO_BAYER = 1
VIDEO_IR_8BIT = 2
VIDEO_IR_10BIT = 3
VIDEO_IR_10BIT_PACKED = 4
VIDEO_YUV_RGB = 5
VIDEO_YUV_RAW = 6

# Depth Formats
DEPTH_11BIT = 0
DEPTH_10BIT = 1
DEPTH_11BIT_PACKED = 2
DEPTH_10BIT_PACKED = 3
DEPTH_REGISTERED = 4
DEPTH_MM = 5

# Resolutions
RESOLUTION_LOW = 0    # 320x240
RESOLUTION_MEDIUM = 1 # 640x480
RESOLUTION_HIGH = 2   # 1280x1024

# C Types
class freenect_context(ctypes.Structure):
    pass

class freenect_device(ctypes.Structure):
    pass

# Library Loading
_lib = None
_lib_sync = None

def _load_libraries():
    global _lib, _lib_sync

    # Potential paths for libfreenect.so
    lib_paths = [
        '/usr/local/lib/libfreenect.so',
        '/usr/lib/libfreenect.so',
        'libfreenect.so'
    ]

    for path in lib_paths:
        try:
            _lib = ctypes.CDLL(path)
            logger.info(f"Loaded libfreenect from {path}")
            break
        except OSError:
            pass

    if _lib is None:
        # Try finding via ldconfig/standard paths
        path = ctypes.util.find_library('freenect')
        if path:
            try:
                _lib = ctypes.CDLL(path)
                logger.info(f"Loaded libfreenect from {path}")
            except OSError:
                pass

    if _lib is None:
        logger.error("Could not load libfreenect.so. Ensure it is installed.")
        return

    # Potential paths for libfreenect_sync.so
    sync_paths = [
        '/usr/local/lib/libfreenect_sync.so',
        '/usr/lib/libfreenect_sync.so',
        'libfreenect_sync.so'
    ]

    for path in sync_paths:
        try:
            _lib_sync = ctypes.CDLL(path)
            logger.info(f"Loaded libfreenect_sync from {path}")
            break
        except OSError:
            pass

    if _lib_sync is None:
        # Try finding via ldconfig/standard paths
        path = ctypes.util.find_library('freenect_sync')
        if path:
            try:
                _lib_sync = ctypes.CDLL(path)
                logger.info(f"Loaded libfreenect_sync from {path}")
            except OSError:
                pass

    if _lib_sync is None:
        logger.warning("Could not load libfreenect_sync.so. Sync functions will not work.")

    # Define function signatures for _lib

    # int freenect_init(freenect_context **ctx, freenect_usb_context *usb_ctx);
    _lib.freenect_init.argtypes = [ctypes.POINTER(ctypes.POINTER(freenect_context)), ctypes.c_void_p]
    _lib.freenect_init.restype = ctypes.c_int

    # int freenect_shutdown(freenect_context *ctx);
    _lib.freenect_shutdown.argtypes = [ctypes.POINTER(freenect_context)]
    _lib.freenect_shutdown.restype = ctypes.c_int

    # int freenect_open_device(freenect_context *ctx, freenect_device **dev, int index);
    _lib.freenect_open_device.argtypes = [ctypes.POINTER(freenect_context), ctypes.POINTER(ctypes.POINTER(freenect_device)), ctypes.c_int]
    _lib.freenect_open_device.restype = ctypes.c_int

    # int freenect_close_device(freenect_device *dev);
    _lib.freenect_close_device.argtypes = [ctypes.POINTER(freenect_device)]
    _lib.freenect_close_device.restype = ctypes.c_int

    # int freenect_set_tilt_degs(freenect_device *dev, double angle);
    _lib.freenect_set_tilt_degs.argtypes = [ctypes.POINTER(freenect_device), ctypes.c_double]
    _lib.freenect_set_tilt_degs.restype = ctypes.c_int

    # int freenect_set_led(freenect_device *dev, freenect_led_options option);
    _lib.freenect_set_led.argtypes = [ctypes.POINTER(freenect_device), ctypes.c_int]
    _lib.freenect_set_led.restype = ctypes.c_int

    # Define function signatures for _lib_sync
    if _lib_sync:
        # int freenect_sync_get_video(void **video, uint32_t *timestamp, int index, freenect_video_format fmt);
        _lib_sync.freenect_sync_get_video.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int]
        _lib_sync.freenect_sync_get_video.restype = ctypes.c_int

        # int freenect_sync_get_depth(void **depth, uint32_t *timestamp, int index, freenect_depth_format fmt);
        _lib_sync.freenect_sync_get_depth.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int]
        _lib_sync.freenect_sync_get_depth.restype = ctypes.c_int

# Load libs on module import
_load_libraries()

# Wrapper Functions

def init():
    if _lib is None:
        raise RuntimeError("libfreenect not loaded")
    ctx = ctypes.POINTER(freenect_context)()
    res = _lib.freenect_init(ctypes.byref(ctx), None)
    if res < 0:
        raise RuntimeError(f"freenect_init failed with code {res}")
    return ctx

def shutdown(ctx):
    if _lib is None: return
    _lib.freenect_shutdown(ctx)

def open_device(ctx, index):
    if _lib is None:
        raise RuntimeError("libfreenect not loaded")
    dev = ctypes.POINTER(freenect_device)()
    res = _lib.freenect_open_device(ctx, ctypes.byref(dev), index)
    if res < 0:
        raise RuntimeError(f"freenect_open_device failed with code {res}")
    return dev

def close_device(dev):
    if _lib is None: return
    _lib.freenect_close_device(dev)

def set_tilt_degs(dev, angle):
    if _lib is None: return
    res = _lib.freenect_set_tilt_degs(dev, float(angle))
    if res < 0:
        logger.error(f"freenect_set_tilt_degs failed with code {res}")

def set_led(dev, option):
    if _lib is None: return
    res = _lib.freenect_set_led(dev, int(option))
    if res < 0:
        logger.error(f"freenect_set_led failed with code {res}")

def sync_get_video(index=0, format=VIDEO_RGB):
    if _lib_sync is None:
        raise RuntimeError("libfreenect_sync not loaded")

    video_ptr = ctypes.c_void_p()
    timestamp = ctypes.c_uint32()

    res = _lib_sync.freenect_sync_get_video(ctypes.byref(video_ptr), ctypes.byref(timestamp), index, format)
    if res != 0:
        raise RuntimeError("freenect_sync_get_video failed")

    # Determine size and shape
    if format == VIDEO_RGB:
        # 640x480 RGB
        width, height = 640, 480
        channels = 3
        size = width * height * channels
        array_type = ctypes.c_uint8 * size

        # Cast void pointer to array pointer
        buffer_ptr = ctypes.cast(video_ptr, ctypes.POINTER(array_type))
        # Create numpy array from buffer
        # Note: np.frombuffer or np.ctypeslib.as_array
        # as_array creates a view, frombuffer creates a copy usually unless using proper buffer interface
        # Since the buffer is owned by freenect_sync and valid until next call,
        # using a copy is safer if we want to process it while next frame is grabbed.
        # However, for speed, a view is better, but we must be careful.
        # Original freenect.sync_get_video() returns a COPY usually?
        # Let's use np.ctypeslib.as_array which creates a numpy array sharing memory.
        # But wait, freenect_sync docs say: "The returned buffer is valid until this function is called again"
        # So if we hold onto this array and call sync_get_video again, the data changes.
        # We should probably return a copy to be safe, like standard python wrappers usually do.

        arr = np.ctypeslib.as_array(buffer_ptr.contents)
        arr = arr.reshape((480, 640, 3)).copy() # Ensure we have a copy and correct shape

    elif format == VIDEO_IR_8BIT:
        width, height = 640, 488 # IR is 640x488? Header says MEDIUM is 640x488 for IR
        # Wait, libfreenect.h says: "FREENECT_RESOLUTION_MEDIUM is 640x488 for the IR camera."
        # Standard usage often crops or assumes 640x480.
        # Let's assume 640x480 for now or check size.
        # Actually, let's just implement RGB as primary requirement.
        # If the user uses IR, they might need adjustments.
        # For now, I'll stick to RGB logic as primary.

        # If someone asks for IR, we can try generic size
        size = 640 * 488
        array_type = ctypes.c_uint8 * size
        buffer_ptr = ctypes.cast(video_ptr, ctypes.POINTER(array_type))
        arr = np.ctypeslib.as_array(buffer_ptr.contents).reshape((488, 640)).copy()

    else:
        # Fallback or error
        raise NotImplementedError(f"Video format {format} not fully implemented yet in kinect_driver.py")

    return arr, timestamp.value

def sync_get_depth(index=0, format=DEPTH_11BIT):
    if _lib_sync is None:
        raise RuntimeError("libfreenect_sync not loaded")

    depth_ptr = ctypes.c_void_p()
    timestamp = ctypes.c_uint32()

    res = _lib_sync.freenect_sync_get_depth(ctypes.byref(depth_ptr), ctypes.byref(timestamp), index, format)
    if res != 0:
        raise RuntimeError("freenect_sync_get_depth failed")

    # Depth is usually 640x480
    width, height = 640, 480

    if format in [DEPTH_11BIT, DEPTH_10BIT, DEPTH_11BIT_PACKED, DEPTH_10BIT_PACKED, DEPTH_MM, DEPTH_REGISTERED]:
        # uint16 items
        size = width * height
        array_type = ctypes.c_uint16 * size
        buffer_ptr = ctypes.cast(depth_ptr, ctypes.POINTER(array_type))
        arr = np.ctypeslib.as_array(buffer_ptr.contents).reshape((480, 640)).copy()
    else:
         raise NotImplementedError(f"Depth format {format} not fully implemented yet in kinect_driver.py")

    return arr, timestamp.value