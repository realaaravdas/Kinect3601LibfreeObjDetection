import sys
import os
import time
import logging
import numpy as np
import threading
from unittest.mock import MagicMock

# Mock ultralytics before importing src
sys.modules["ultralytics"] = MagicMock()

# Add root to path
sys.path.append(os.getcwd())

from src.camera.dummy import DummyCamera
from src.camera.camera_thread import CameraThread
from src.perception.detection_thread import DetectionThread
from src.perception.odometry import VisualOdometry

# Mock detector to avoid loading heavy YOLO model in test
class MockDetector:
    def detect(self, frame):
        # Return fake results object
        class Box:
            def __init__(self):
                import torch # mimic torch tensor behavior if needed or just numpy
                # The code uses .cpu().numpy()
                # So we need an object that has .cpu().numpy()
                pass

        # Simpler: Create a mock class structure matching usage
        class MockTensor:
            def __init__(self, data):
                self.data = data
            def cpu(self):
                return self
            def numpy(self):
                return np.array(self.data)
            def numel(self):
                return len(self.data) if isinstance(self.data, list) else 1
            def item(self):
                return self.data[0] if isinstance(self.data, list) else self.data

        class MockBox:
            def __init__(self):
                self.xyxy = [MockTensor([100, 100, 200, 200])]
                self.cls = MockTensor([0])
                self.conf = MockTensor([0.9])

        class MockResults:
            def __init__(self):
                self.boxes = [MockBox()]

        return MockResults()

def test_system():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting System Test...")

    # 1. Camera
    cam = DummyCamera()
    cam_thread = CameraThread(cam)
    cam_thread.start()

    # Wait for frame
    time.sleep(1)
    rgb, depth = cam_thread.get_latest_frame()
    assert rgb is not None, "Camera thread failed to produce RGB"
    assert depth is not None, "Camera thread failed to produce Depth"
    logging.info("Camera Thread verified.")

    # 2. Odometry
    odom = VisualOdometry()
    # Run a few updates
    for i in range(5):
        rgb, depth = cam_thread.get_latest_frame()
        if rgb is not None:
            pose = odom.update(rgb, depth)
            logging.info(f"Odom Update {i}: Pose={pose}")
            assert len(pose) == 4, "Pose should be (x, y, z, yaw)"
        time.sleep(0.1)
    logging.info("Odometry verified (ran without crash).")

    # 3. Detection
    # Use mock detector
    detector = MockDetector()
    det_thread = DetectionThread(detector)
    det_thread.start()

    # Process a few frames
    det_thread.process_frame(rgb)
    time.sleep(0.5)
    results = det_thread.get_latest_results()

    assert results is not None, "Detection thread failed to produce results"
    # Verify content
    box = results.boxes[0]
    coords = box.xyxy[0].cpu().numpy()
    assert coords[0] == 100
    logging.info("Detection Thread verified.")

    # Cleanup
    cam_thread.stop()
    det_thread.stop()
    logging.info("Test Complete.")

if __name__ == "__main__":
    test_system()
