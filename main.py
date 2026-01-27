import argparse
import time
import logging
import sys
import numpy as np
import threading
import cv2

# Adjust path if needed, but since main.py is in root, src should be importable
from src.camera import Kinect360, KinectOne, DummyCamera
from src.camera.camera_thread import CameraThread
from src.perception import ObjectDetector, DepthProcessor, VisualOdometry
from src.perception.detection_thread import DetectionThread
from src.mapping import MapManager
from src.ui import Display, CLI

def main():
    parser = argparse.ArgumentParser(description="Kinect Object Detector & Mapper")
    parser.add_argument("--camera", choices=["360", "one", "dummy"], default="dummy", help="Camera type")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Init Camera Hardware
    logging.info(f"Initializing {args.camera} camera...")
    if args.camera == "360":
        cam_hw = Kinect360()
        fov = 57.0
    elif args.camera == "one":
        cam_hw = KinectOne()
        fov = 70.6
    else:
        cam_hw = DummyCamera()
        fov = 60.0

    try:
        cam_hw.open()
    except Exception as e:
        logging.critical(f"Failed to open camera: {e}")
        return

    # Startup Sequence (360)
    if args.camera == "360":
        cam_hw.startup_sequence()

    # 2. Start Camera Thread
    # This decouples camera capture (30fps) from processing
    cam_thread = CameraThread(cam_hw)
    cam_thread.start()

    # Wait for camera to warm up and provide first frame
    logging.info("Waiting for camera stream...")
    frame_ready = False
    for _ in range(50): # Wait up to 5 seconds
        if cam_thread.get_latest_frame()[0] is not None:
            frame_ready = True
            break
        time.sleep(0.1)

    if not frame_ready:
        logging.error("Camera failed to provide frames. Exiting.")
        cam_thread.stop()
        return

    # 3. Init Perception & Mapping
    # Detector (YOLO) - Heavy, put in thread
    detector = ObjectDetector(args.model)
    det_thread = DetectionThread(detector)
    det_thread.start()

    # Depth Processor & Odometry
    depth_proc = DepthProcessor(fov_h=fov)
    odom = VisualOdometry(fov_h=fov)
    mapper = MapManager()

    # 4. Init UI
    display = Display()
    cli = CLI()

    logging.info("Starting main loop. Press 'q' in console or GUI window to quit.")

    try:
        while True:
            # 1. Get Latest Frame (Instant)
            rgb, depth = cam_thread.get_latest_frame()
            if rgb is None:
                time.sleep(0.001)
                continue

            # 2. Visual Odometry (Fast, C++ backed)
            # Update robot pose based on visual features
            pose = odom.update(rgb, depth)
            mapper.update_pose(pose)

            # 3. Object Detection (Async)
            # Send current frame to detection thread
            det_thread.process_frame(rgb)

            # Get latest available results
            results = det_thread.get_latest_results()

            # 4. Map Updates using Detection Results
            if results and results.boxes:
                for box in results.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item()) if box.cls.numel() > 0 else 0
                    conf = float(box.conf[0].item()) if box.conf.numel() > 0 else 0.0

                    # Estimate Distance using CURRENT depth
                    dist, angle = depth_proc.get_distance_and_angle(depth, coords)

                    if dist is not None:
                        mapper.add_observation(cls, dist, angle, conf)

            # 5. Display
            # Show the RGB frame (current) and overlay *latest* detections.
            key = display.show(rgb, results, mapper.get_objects(), mapper.get_pose())
            if key == ord('q'):
                break

            # 6. CLI
            cmd = cli.get_command()
            if cmd:
                if cmd in ["q", "quit", "exit"]:
                    break
                if cmd.startswith("tilt ") and args.camera == "360":
                    try:
                        angle = float(cmd.split()[1])
                        cam_hw.set_tilt(angle)
                        logging.info(f"Tilt set to {angle}")
                    except ValueError:
                        logging.error("Invalid tilt angle")
                elif cmd == "reset":
                    odom.reset()
                    logging.info("Odometry reset")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("Shutting down...")
        cam_thread.stop()
        det_thread.stop()
        cli.stop()
        display.close()
        # cam_hw.close()

if __name__ == "__main__":
    main()
