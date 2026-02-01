#!/bin/bash

# Script to install rknn-toolkit-lite2 on Orange Pi 5
# This script assumes you are running on the device or a compatible aarch64 environment.

set -e

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: System architecture is $ARCH. rknn-toolkit-lite2 is designed for aarch64 (Orange Pi 5)."
    read -p "Continue anyway? (y/N) " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        exit 1
    fi
fi

# Detect Python Version
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -ne 3 ]; then
    echo "Error: Python 3 is required."
    exit 1
fi

PY_TAG="cp${PY_MAJOR}${PY_MINOR}"
echo "Detected Python version: ${PY_MAJOR}.${PY_MINOR} ($PY_TAG)"

# Define RKNN Toolkit Lite 2 Version
# You can check available versions at: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
VERSION="1.6.0"

WHEEL_NAME="rknn_toolkit_lite2-${VERSION}-${PY_TAG}-${PY_TAG}-linux_aarch64.whl"
DOWNLOAD_URL="https://github.com/rockchip-linux/rknn-toolkit2/raw/master/rknn-toolkit-lite2/packages/${WHEEL_NAME}"

echo "Downloading ${WHEEL_NAME}..."
echo "URL: ${DOWNLOAD_URL}"

if wget -q --show-progress "${DOWNLOAD_URL}"; then
    echo "Download successful."
else
    echo "Download failed. This might be because:"
    echo "1. The version ${VERSION} is not available for Python ${PY_MAJOR}.${PY_MINOR}."
    echo "2. No internet connection."
    echo "Please check https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages for available versions."
    exit 1
fi

echo "Installing ${WHEEL_NAME}..."
pip install "${WHEEL_NAME}"
pip install numpy opencv-python

echo "Cleaning up..."
rm "${WHEEL_NAME}"

echo "Installation complete! You can now use .rknn models."
