from ultralytics import YOLO
import numpy as np
import logging

class ObjectDetector:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger(__name__)
        if model_path:
             self.logger.info(f"Loading custom model: {model_path}")
             self.model = YOLO(model_path)
             self.is_custom = True
        else:
             self.logger.info("Loading generic YOLOv8n model")
             self.model = YOLO("yolov8n.pt")
             self.is_custom = False

    def detect(self, image):
        """
        Detects objects in the image.
        Args:
            image: numpy array (H, W, 3)
        Returns:
            list of dicts: [{'box': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'class_name': str}]
        """
        results = self.model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]

                # Filter if generic model
                if not self.is_custom:
                    if class_name != 'sports ball':
                        continue

                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name
                })
        return detections
