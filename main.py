import argparse
import time
import logging
import sys
import numpy as np

# Adjust path if needed, but since main.py is in root, src should be importable
from src.camera import Kinect360, KinectOne, DummyCamera
from src.perception import ObjectDetector, DepthProcessor, VisualOdometry
from src.mapping import MapManager
from src.ui import Display, CLI

def main():
    parser = argparse.ArgumentParser(description="Kinect Object Detector & Mapper")
    parser.add_argument("--camera", choices=["360", "one", "dummy"], default="dummy", help="Camera type")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Init Camera
    logging.info(f"Initializing {args.camera} camera...")
    if args.camera == "360":
        cam = Kinect360()
        fov = 57.0
    elif args.camera == "one":
        cam = KinectOne()
        fov = 70.6
    else:
        cam = DummyCamera()
        fov = 60.0

    try:
        cam.open()
    except Exception as e:
        logging.critical(f"Failed to open camera: {e}")
        return

    # Startup Sequence (360)
    if args.camera == "360":
        cam.startup_sequence()

    # 2. Init Perception
    # Check if model exists or let ultralytics download it
    detector = ObjectDetector(args.model)
    depth_proc = DepthProcessor(fov_h=fov)
    odom = VisualOdometry(fov_h=fov)

    # 3. Init Mapping
    mapper = MapManager()

    # 4. Init UI
    display = Display()
    cli = CLI()

    logging.info("Starting main loop. Press 'q' in console or GUI window to quit.")
    try:
        while True:
            # Check CLI
            cmd = cli.get_command()
            if cmd:
                if cmd in ["q", "quit", "exit"]:
                    break
                if cmd.startswith("tilt ") and args.camera == "360":
                    try:
                        angle = float(cmd.split()[1])
                        cam.set_tilt(angle)
                        logging.info(f"Tilt set to {angle}")
                    except ValueError:
                        logging.error("Invalid tilt angle")
                elif cmd == "reset":
                    odom.reset()
                    logging.info("Odometry reset")

            # 1. Get Frame
            rgb, depth = cam.get_frame()
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            # 2. Odometry
            delta_yaw = odom.update(depth)
            mapper.update_pose(delta_yaw)

            # 3. Detection
            results = detector.detect(rgb) # Returns a Results object

            # 4. Map Updates
            # Iterate through detections
            if results and results.boxes:
                for box in results.boxes:
                    # box.xyxy is tensor
                    coords = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2

                    # Handle cls and conf
                    if box.cls.numel() > 0:
                        cls = int(box.cls[0].item())
                    else:
                        continue

                    if box.conf.numel() > 0:
                        conf = float(box.conf[0].item())
                    else:
                        conf = 0.0

                    # Estimate Distance
                    dist, angle = depth_proc.get_distance_and_angle(depth, coords)

                    if dist is not None:
                        # Log specific classes if needed (e.g., yellow ball)
                        # For now, map everything
                        mapper.add_observation(cls, dist, angle, conf)

            # 5. Display
            key = display.show(rgb, results, mapper.get_objects(), mapper.get_pose())
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("Shutting down...")
        cli.stop()
        display.close()
        cam.close()

if __name__ == "__main__":
    main()
