# Kinect Yellow Ball Detector

This is a ROS2 package for detecting yellow balls using an Xbox 360 or Xbox One Kinect, mapping their locations, and estimating robot rotation.

## Features
- **Object Detection**: Uses YOLOv8 to detect yellow balls (or any object if generic model used).
- **Mapping**: Creates a 2D map of detected objects using depth data and robot orientation.
- **Visual Odometry**: Estimates horizontal rotation (yaw) using ORB feature tracking and depth data.
- **Hardware Support**:
  - Xbox 360 Kinect (`libfreenect`) - Includes motor control (tilt up/down/center at startup).
  - Xbox One Kinect (`libfreenect2`).
- **GUI**: PyQt5-based interface showing RGB feed, Depth feed, and the live 2D map.

## Installation

### 1. System Dependencies
This package requires ROS2 (Humble or later recommended) and Kinect libraries.

**Ubuntu (Host or Jetson):**
```bash
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install libfreenect-dev freenect-bin # For Xbox 360 Kinect
# For Xbox One Kinect, follow libfreenect2 installation instructions from their repo.
sudo apt install python3-pip
```

### 2. Python Dependencies
```bash
pip3 install ultralytics opencv-python numpy PyQt5
# For Xbox 360
pip3 install freenect
# For Xbox One
# pip3 install pylibfreenect2 # Requires building from source usually
```

### 3. Build the Package
```bash
mkdir -p ros2_ws/src
cd ros2_ws/src
# Clone this repo here or copy the folder 'kinect_yellow_ball_detector'
cd ..
colcon build --packages-select kinect_yellow_ball_detector
source install/setup.bash
```

## Usage

### Connecting the Kinect
- **Xbox 360**: Plug in the USB and Power. Ensure the motor is working.
- **Xbox One**: Plug in USB 3.0 and Power.

### Running the System
To launch the driver and the GUI:

**For Xbox 360 Kinect:**
```bash
ros2 launch kinect_yellow_ball_detector kinect_system.launch.py device_type:=freenect1
```
The camera will perform a startup sequence: Tilt Up -> Tilt Down -> Center (0 deg).

**For Xbox One Kinect:**
```bash
ros2 launch kinect_yellow_ball_detector kinect_system.launch.py device_type:=freenect2
```

**For Simulation (No Hardware):**
```bash
ros2 launch kinect_yellow_ball_detector kinect_system.launch.py device_type:=mock
```

### Using a Custom Model
If you have a custom YOLOv8 model (e.g., `yellow_ball.pt`):
```bash
ros2 launch kinect_yellow_ball_detector kinect_system.launch.py device_type:=freenect1 model_path:=/path/to/yellow_ball.pt
```

## GUI Manual
- **RGB Feed**: Shows the camera view with bounding boxes around detected balls.
- **Depth Feed**: Shows the depth map (lighter is closer).
- **Map**: Shows a 2D top-down view.
  - **Blue Dot**: Robot position (starts at 0,0).
  - **Red Line**: Robot orientation.
  - **Yellow Dots**: Detected balls.

## CLI / Controls
Currently, the system is autonomous.
- **Xbox 360 Tilt**: The tilt is calibrated at startup.
- **Mapping**: The map updates automatically as you move the robot (rotate).

## Architecture
- `driver_node`: Handles hardware interface, detection, SLAM, and publishing.
- `visualization_node`: Subscribes to topics and runs the PyQt5 GUI.
- `kinect_wrapper`: Abstraction for different hardware.
- `slam.py`: Contains `VisualOdometry` and `Mapper` classes.

## Notes for Jetson Deployment
- Ensure CUDA is available for YOLOv8 to offload processing to GPU. `ultralytics` will use GPU automatically if installed with CUDA support.
- If using `libfreenect2`, ensure OpenCL/CUDA packet processing is enabled for performance.
