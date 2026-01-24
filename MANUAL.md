# User Manual

## Running the Program

Run the main script from the terminal:

```bash
python3 main.py --camera [360|one|dummy] --model [path/to/yolo.pt]
```

### Arguments
- `--camera`: Choose the camera type.
    - `360`: Xbox 360 Kinect (requires `freenect`).
    - `one`: Xbox One Kinect (requires `libfreenect2`).
    - `dummy`: Simulation mode (random noise).
- `--model`: Path to the YOLOv8 model file (e.g., `yolov8n.pt`). Defaults to `yolov8n.pt`.

## Interface

### GUI Windows
1. **Feed**: Shows the RGB camera stream with bounding boxes around detected objects.
2. **Map**: Shows a top-down 2D map.
   - **Red Dot/Arrow**: The robot's estimated position and heading.
   - **Yellow Dots**: Detected objects (e.g., yellow balls).
   - **Grid**: 1-meter grid lines.

### CLI Commands
Type commands into the terminal window where the program is running.

- `q`, `quit`, `exit`: Stop the program.
- `tilt <angle>`: (Xbox 360 only) Set the camera tilt angle (degrees).
    - Example: `tilt 15` (up), `tilt -10` (down), `tilt 0` (flat).
- `reset`: Reset the visual odometry.

## Operation Logic

1. **Startup**:
   - If using Kinect 360, the camera will perform a calibration sequence: Tilt Up -> Tilt Down -> Center.
2. **Mapping**:
   - The robot starts at (0,0) facing East (0 degrees).
   - As the camera rotates, the software estimates the rotation angle using depth sensor data ("Visual Odometry").
   - Detected objects are placed on the map relative to the robot's position.
   - Duplicate detections (close to existing ones) are merged to refine position.

## Troubleshooting

- **No Camera Found**: Ensure permissions are set (udev rules) and drivers are installed.
- **GUI Error**: If running over SSH or WSL, ensure X11 forwarding or a display server is active.
- **Lag**: Detection is heavy. On Jetson Nano, ensure you are using a CUDA-enabled PyTorch build.
