import logging
import cv2
import numpy as np

# Try imports
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

class BaseDetector:
    def detect(self, image):
        """
        Detect objects in the image.
        Args:
            image: RGB numpy array (H, W, 3).
        Returns:
            list of dicts: [{'xyxy': [x1, y1, x2, y2], 'cls': int, 'conf': float}, ...]
        """
        raise NotImplementedError

class UltralyticsDetector(BaseDetector):
    def __init__(self, model_path="yolov8n.pt"):
        if not ULTRALYTICS_AVAILABLE:
            logging.error("Ultralytics not installed. Cannot load .pt model.")
            self.model = None
            return

        logging.info(f"Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = None

    def detect(self, image):
        if self.model is None:
            return []

        # Run inference
        results = self.model(image, verbose=False)
        result = results[0]

        boxes_data = []
        if result.boxes:
            for box in result.boxes:
                boxes_data.append({
                    'xyxy': box.xyxy[0].cpu().numpy().tolist(),
                    'cls': int(box.cls[0].item()) if box.cls.numel() > 0 else 0,
                    'conf': float(box.conf[0].item()) if box.conf.numel() > 0 else 0.0
                })
        return boxes_data

class RKNNDetector(BaseDetector):
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.25, iou_thres=0.45):
        if not RKNN_AVAILABLE:
            logging.error("rknnlite not installed. Cannot load .rknn model.")
            logging.error("Please run 'bash scripts/install_rknn_opi5.sh' to install it for Orange Pi 5 NPU support.")
            self.rknn = None
            return

        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        logging.info(f"Loading RKNN model from {model_path}...")
        self.rknn = RKNNLite()

        # Load RKNN model
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logging.error("Load RKNN model failed")
            self.rknn = None
            return

        # Init runtime environment
        ret = self.rknn.init_runtime()
        if ret != 0:
            logging.error("Init runtime environment failed")
            self.rknn = None
            return

        logging.info("RKNN model loaded and initialized.")

    def detect(self, image):
        if self.rknn is None:
            return []

        # Preprocess
        input_img, ratio, (dw, dh) = self.letterbox(image)

        # Inference
        # RKNNLite inference returns a list of outputs
        outputs = self.rknn.inference(inputs=[input_img])

        # Postprocess
        detections = self.postprocess(outputs, ratio, (dw, dh))

        return detections

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # RKNN input usually expects RGB?
        # OpenCV is BGR. If model expects RGB, we convert.
        # Assuming model expects RGB (YOLOv8 default).
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Wait, the input `image` to detect() is stated as RGB in docstrings of previous code.
        # If it's already RGB, we are good.

        # Expand dims for batch: (1, H, W, 3)
        img = np.expand_dims(img, axis=0)

        return img, ratio, (dw, dh)

    def postprocess(self, outputs, ratio, pad):
        """
        Args:
            outputs: list of numpy arrays from RKNN inference.
            ratio: (rx, ry) scaling factor.
            pad: (dw, dh) padding.
        """
        # YOLOv8 Output processing
        # Expected output shape: (1, 84, 8400) or similar. 84 = 4 box + 80 classes.

        output = outputs[0] # Assuming single output head for v8 export

        # Handle shapes
        # If (1, 84, N) -> transpose to (1, N, 84)
        if output.shape[1] == 84:
            output = output.transpose(0, 2, 1)

        # Now shape is (1, N, 84)
        # Squeeze batch
        pred = output[0] # (N, 84)

        # Split boxes and scores
        boxes = pred[:, :4] # cx, cy, w, h
        scores = pred[:, 4:] # classes

        # Get max score and class
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_thres
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert boxes from cx, cy, w, h to x1, y1, x2, y2
        # Note: These are in the resized image coordinates (with padding)
        xc = boxes[:, 0]
        yc = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        # Prepare for NMS
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h] usually?
        # No, it expects [x_left, y_top, width, height] for NMSBoxes.
        # But we want output as xyxy.

        # Let's use NMSBoxes
        boxes_nms = np.stack([x1, y1, w, h], axis=1).tolist()
        scores_nms = confidences.tolist()

        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.conf_thres, self.iou_thres)

        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                # Get the box in resized coords
                bx1, by1, bw, bh = boxes_nms[i]
                bx2 = bx1 + bw
                by2 = by1 + bh

                # Scale back to original image
                # (x - dw) / ratio
                # pad is (dw, dh)
                # ratio is (rx, ry) or scalar r

                rx, ry = ratio
                dw, dh = pad

                orig_x1 = (bx1 - dw) / rx
                orig_y1 = (by1 - dh) / ry
                orig_x2 = (bx2 - dw) / rx
                orig_y2 = (by2 - dh) / ry

                results.append({
                    'xyxy': [orig_x1, orig_y1, orig_x2, orig_y2],
                    'cls': int(class_ids[i]),
                    'conf': float(confidences[i])
                })

        return results

def get_detector(model_path):
    if model_path.endswith('.rknn'):
        return RKNNDetector(model_path)
    return UltralyticsDetector(model_path)
