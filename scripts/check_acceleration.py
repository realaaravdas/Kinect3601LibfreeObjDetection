from ultralytics import YOLO
import logging
import sys

logging.basicConfig(level=logging.INFO)

def check():
    try:
        model = YOLO('yolov8n.pt')
        logging.info("Exporting to NCNN...")
        # Export
        # This might take a while and require 'ultralytics[export]' dependencies
        path = model.export(format='ncnn')
        logging.info(f"NCNN Export successful: {path}")
    except Exception as e:
        logging.error(f"NCNN Export failed: {e}")

    try:
        model = YOLO('yolov8n.pt')
        logging.info("Exporting to ONNX...")
        path = model.export(format='onnx', opset=12)
        logging.info(f"ONNX Export successful: {path}")
    except Exception as e:
        logging.error(f"ONNX Export failed: {e}")

if __name__ == "__main__":
    check()
