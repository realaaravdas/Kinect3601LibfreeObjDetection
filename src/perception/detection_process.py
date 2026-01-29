import multiprocessing
import time
import logging
import queue
import cv2

class DetectionProcess(multiprocessing.Process):
    def __init__(self, model_path, input_queue, result_queue):
        """
        Process-based Object Detection to bypass GIL.
        Args:
            model_path: Path to YOLO model.
            input_queue: Queue for incoming RGB frames.
            result_queue: Queue for outgoing results.
        """
        super().__init__()
        self.model_path = model_path
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.daemon = True # Kill when main process exits

    def run(self):
        # Re-configure logging for this process
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("DetectionProcess: Initializing...")

        # Import locally to avoid issues in parent process
        try:
            from src.perception.detector import ObjectDetector
            detector = ObjectDetector(self.model_path)
        except Exception as e:
            logging.error(f"DetectionProcess: Failed to load model: {e}")
            return

        logging.info("DetectionProcess: Started loop.")

        while True:
            try:
                # Get frame (blocking with timeout to save CPU)
                # We only want the LATEST frame.
                try:
                    frame = self.input_queue.get(timeout=0.01)
                    # If queue has more, drain it to get the very last one
                    while True:
                        try:
                            frame = self.input_queue.get_nowait()
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if frame is None:
                    continue

                # Run Detection
                results = detector.detect(frame)

                # We cannot put the entire Results object if it contains Tensors/complex types that don't pickle well?
                # Ultralytics Results are pickleable usually, but let's be safe and extract data.
                # Actually, main.py expects Results object or similar.
                # Let's try sending the object. If it fails, we strip it down.
                # However, Results object holds ref to original image. We might not want to send that back?
                # Ideally we just send boxes, classes, confs.

                # To be safe and efficient:
                boxes_data = []
                if results and results.boxes:
                    for box in results.boxes:
                        boxes_data.append({
                            'xyxy': box.xyxy[0].cpu().numpy(),
                            'cls': int(box.cls[0].item()) if box.cls.numel() > 0 else 0,
                            'conf': float(box.conf[0].item()) if box.conf.numel() > 0 else 0.0
                        })

                # Put in result queue
                # Clear old results? No, consumer handles that.
                self.result_queue.put(boxes_data)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"DetectionProcess: Error: {e}")
                time.sleep(0.1)

        logging.info("DetectionProcess: Stopped.")
