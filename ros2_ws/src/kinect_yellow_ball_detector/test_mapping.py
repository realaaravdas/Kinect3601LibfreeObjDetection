from kinect_yellow_ball_detector.slam import Mapper
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Mapper...")
    mapper = Mapper()

    # Simulate a detection
    # Box at center (320, 240)
    # Depth 1000mm (1 meter)
    detections = [{
        'box': [300, 220, 340, 260],
        'confidence': 0.9,
        'class_id': 0,
        'class_name': 'ball'
    }]

    depth_frame = np.zeros((480, 640), dtype=np.float32)
    depth_frame[240, 320] = 1000.0

    print("Processing detection at Yaw 0...")
    mapper.process_detections(detections, depth_frame)

    map_data = mapper.get_map()
    print(f"Map: {map_data}")

    # Expect ball at (0, 1000) roughly
    if len(map_data) == 1 and abs(map_data[0]['z'] - 1000) < 10:
        print("PASS: Ball detected correctly at Yaw 0.")
    else:
        print("FAIL: Ball position incorrect.")

    # Rotate robot 90 degrees right
    print("Rotating 90 degrees...")
    mapper.update_robot_pose(90)

    # Detection is now at left of image (approx) if ball stayed still relative to world
    # But let's simulate a NEW ball detection while rotated.
    # If I am facing right (X+), and I see a ball in front (Z_local +),
    # it should be at World X+, World Z 0.

    detections_2 = [{
        'box': [300, 220, 340, 260],
        'confidence': 0.9,
        'class_id': 0,
        'class_name': 'ball'
    }]
    depth_frame[240, 320] = 2000.0 # 2 meters

    mapper.process_detections(detections_2, depth_frame)
    map_data = mapper.get_map()
    print(f"Map: {map_data}")

    # Expect 2nd ball at roughly (2000, 0)
    found = False
    for ball in map_data:
        if abs(ball['x'] - 2000) < 100 and abs(ball['z']) < 100:
            found = True
            break

    if found:
        print("PASS: Ball detected correctly at Yaw 90.")
    else:
        print("FAIL: Ball position incorrect at Yaw 90.")

if __name__ == "__main__":
    main()
