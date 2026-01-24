# Installation Instructions

This project is designed to run on Linux (Ubuntu, WSL, Jetson Nano).

## Prerequisites

- Python 3.8 or higher
- System libraries for Kinect devices (`libfreenect` or `libfreenect2`)

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 2. Install Kinect Drivers

### For Xbox 360 Kinect (`libfreenect`)

On Ubuntu/Debian/Jetson:
```bash
sudo apt-get install libfreenect-dev freenect
```
You may also need the python bindings. If they are not installed by apt:
```bash
sudo apt-get install python3-freenect
```
Or install via pip (experimental):
```bash
pip install freenect
```

### For Xbox One Kinect (`libfreenect2`)

**Ubuntu / Jetson:**
Follow the instructions at [OpenKinect/libfreenect2](https://github.com/OpenKinect/libfreenect2).

1. Install build tools and dependencies:
   ```bash
   sudo apt-get install build-essential cmake pkg-config libusb-1.0-0-dev \
      libturbojpeg0-dev libglfw3-dev
   ```

2. Build `libfreenect2`:
   ```bash
   git clone https://github.com/OpenKinect/libfreenect2.git
   cd libfreenect2
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
   make
   make install
   ```

3. Install Python bindings (`pylibfreenect2`):
   ```bash
   pip install pylibfreenect2
   ```
   You might need to set `LIBFREENECT2_INSTALL_PREFIX` if installed in a custom location.

## 3. Jetson Nano Specifics (Legacy)

The Jetson Nano requires specific PyTorch versions compatible with JetPack.
Do **not** simply run `pip install torch`.
Follow the NVIDIA forums to install PyTorch for Jetson.
Typically:
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
```
(Adjust for your Python version and JetPack version).

`ultralytics` (YOLOv8) depends on PyTorch. Ensure the system PyTorch is detected.

## 4. WSL Setup

WSL2 does not natively support USB devices (needed for Kinect) without `usbipd-win`.
1. Install `usbipd-win` on Windows.
2. Bind the Kinect USB device to WSL.
3. Verify access with `lsusb` inside WSL.
4. Note: GUI windows (OpenCV) require an X Server (like VcXsrv) or WSLg (Windows 11).

## 5. Verify Installation

Run the dummy camera mode to check software dependencies:
```bash
python3 main.py --camera dummy
```
