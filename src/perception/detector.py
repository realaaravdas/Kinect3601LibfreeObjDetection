from ultralytics import YOLO
import cv2
import logging

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        logging.info(f"Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = None

    def detect(self, image):
        """
        Detect objects in the image.
        Args:
            image: RGB numpy array.
        Returns:
            results: List of detection results (boxes, classes, confidences).
        """
        if self.model is None:
            return []

        # Run inference
        results = self.model(image, verbose=False)
        return results[0] # Return the first result (single image)
