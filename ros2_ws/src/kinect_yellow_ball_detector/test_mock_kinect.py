from kinect_yellow_ball_detector.kinect_wrapper import MockKinectDevice
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Mock Device...")
    kinect = MockKinectDevice()

    print("Running Startup Sequence...")
    kinect.startup_sequence()

    print("Getting frames...")
    rgb, depth = kinect.get_frames()
    print(f"RGB Shape: {rgb.shape}")
    print(f"Depth Shape: {depth.shape}")

    print("Stopping device...")
    kinect.stop()
    print("Test passed.")

if __name__ == "__main__":
    main()
