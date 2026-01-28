import argparse
import time
import logging
import sys
import numpy as np
import threading
import multiprocessing
import cv2

# Adjust path if needed
from src.camera import Kinect360, KinectOne, DummyCamera
from src.camera.camera_thread import CameraThread
from src.perception.geometry import CameraProjector
from src.perception.odometry import VisualOdometry
from src.perception.detection_process import DetectionProcess
from src.mapping import MapManager
from src.ui import Display, CLI

# Needed for multiprocessing in some environments
def setup_multiprocessing():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def main():
    setup_multiprocessing()

    parser = argparse.ArgumentParser(description="Kinect Object Detector & Mapper")
    parser.add_argument("--camera", choices=["360", "one", "dummy"], default="dummy", help="Camera type")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--height", type=float, default=1.0, help="Camera height in meters")
    parser.add_argument("--tilt", type=float, default=0.0, help="Camera tilt in degrees (positive up)")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Init Camera Hardware
    logging.info(f"Initializing {args.camera} camera...")

    # Default intrinsics placeholders (will be updated when frame is received)
    width, height = 640, 480
    fov_h = 57.0
    fov_v = 43.0

    if args.camera == "360":
        cam_hw = Kinect360()
        fov_h = 57.0
        fov_v = 43.0
        width, height = 640, 480
    elif args.camera == "one":
        cam_hw = KinectOne()
        fov_h = 70.6
        fov_v = 60.0
        width, height = 512, 424 # As per kinect_one.py Registration choice
    else:
        cam_hw = DummyCamera()
        fov_h = 60.0
        width, height = 640, 480

    try:
        cam_hw.open()
    except Exception as e:
        logging.critical(f"Failed to open camera: {e}")
        return

    # Startup Sequence (360)
    if args.camera == "360":
        cam_hw.startup_sequence()
        # If user specified tilt via CLI, apply it now?
        # But startup sequence resets to 0.
        if args.tilt != 0:
            cam_hw.set_tilt(args.tilt)

    # 2. Start Camera Thread
    cam_thread = CameraThread(cam_hw)
    cam_thread.start()

    # Wait for camera
    logging.info("Waiting for camera stream...")
    frame_ready = False
    for _ in range(50):
        rgb, _ = cam_thread.get_latest_frame()
        if rgb is not None:
            frame_ready = True
            # Update width/height based on actual frame
            height, width, _ = rgb.shape
            break
        time.sleep(0.1)

    if not frame_ready:
        logging.error("Camera failed to provide frames.")
        cam_thread.stop()
        return

    logging.info(f"Camera started. Resolution: {width}x{height}")

    # 3. Init Perception & Mapping

    # Projector
    projector = CameraProjector(fov_h=fov_h, fov_v=fov_v, cam_height=args.height, tilt_angle=args.tilt)

    # Odometry (Initialize with correct resolution)
    odom = VisualOdometry(fov_h=fov_h, width=width, height=height)

    # Mapping
    mapper = MapManager(projector=projector)

    # Detection (Multiprocessing)
    frame_queue = multiprocessing.Queue(maxsize=2)
    result_queue = multiprocessing.Queue()

    det_process = DetectionProcess(args.model, frame_queue, result_queue)
    det_process.start()

    # 4. Init UI
    display = Display()
    cli = CLI()

    logging.info("Starting main loop...")

    try:
        while True:
            # 1. Get Frame
            rgb, depth = cam_thread.get_latest_frame()
            if rgb is None:
                time.sleep(0.001)
                continue

            # 2. Update Configuration (if changed via CLI)
            # (CLI tilt changes handled below, need to update Projector)

            # 3. Visual Odometry
            pose = odom.update(rgb, depth)
            mapper.update_pose(pose)

            # 4. Object Detection (Async)
            # Try push frame
            try:
                # Only push if queue is empty to avoid lag?
                # Or use maxsize=2 and put_nowait
                if not frame_queue.full():
                    frame_queue.put_nowait(rgb)
            except queue.Full:
                pass

            # Get Results
            detections = []
            try:
                while True:
                    detections = result_queue.get_nowait()
            except queue.Empty:
                pass

            # 5. Map Updates
            # MapManager now handles projection, persistence, etc.
            if depth is not None:
                H, W = depth.shape
                mapper.update_map(detections, depth, W, H)

            # 6. Display
            key = display.show(rgb, detections, mapper.get_objects(), mapper.get_pose(), mapper.get_frustum())
            if key == ord('q'):
                break

            # 7. CLI
            cmd = cli.get_command()
            if cmd:
                if cmd in ["q", "quit", "exit"]:
                    break
                if cmd.startswith("tilt ") and args.camera == "360":
                    try:
                        angle = float(cmd.split()[1])
                        cam_hw.set_tilt(angle)
                        projector.update_config(projector.height, angle) # Sync projector
                        logging.info(f"Tilt set to {angle}")
                    except ValueError:
                        logging.error("Invalid tilt angle")
                elif cmd == "reset":
                    odom.reset()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        logging.info("Shutting down...")
        cam_thread.stop()

        # Kill Detection Process
        det_process.terminate()
        det_process.join()

        cli.stop()
        display.close()

if __name__ == "__main__":
    main()
