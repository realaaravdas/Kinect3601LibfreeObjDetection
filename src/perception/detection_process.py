import multiprocessing
import time
import logging
import queue
import os

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

        try:
            from ultralytics import YOLO

            # Check if optimized model exists
            ncnn_path = self.model_path.replace(".pt", "_ncnn_model")
            if os.path.isdir(ncnn_path):
                logging.info(f"Loading NCNN optimized model: {ncnn_path}")
                detector = YOLO(ncnn_path, task='detect')
            else:
                logging.info(f"Loading standard model: {self.model_path}")
                detector = YOLO(self.model_path)

        except Exception as e:
            logging.error(f"DetectionProcess: Failed to load model: {e}")
            return

        logging.info("DetectionProcess: Started loop.")

        while True:
            try:
                # Get frame (non-blocking)
                frame = None
                try:
                    # Skip to latest frame if multiple
                    while True:
                         frame = self.input_queue.get_nowait()
                except queue.Empty:
                    pass

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Run Tracking
                # persist=True is needed for ID tracking
                # tracker="bytetrack.yaml" is standard
                # verbose=False reduces log spam
                # stream=True ?? No, we process one by one here.
                results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                boxes_data = []
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        # Extract ID if available
                        obj_id = int(box.id[0].item()) if box.id is not None else -1

                        boxes_data.append({
                            'xyxy': box.xyxy[0].cpu().numpy(),
                            'cls': int(box.cls[0].item()) if box.cls.numel() > 0 else 0,
                            'conf': float(box.conf[0].item()) if box.conf.numel() > 0 else 0.0,
                            'id': obj_id
                        })

                self.result_queue.put(boxes_data)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"DetectionProcess: Error: {e}")
                time.sleep(0.1)

        logging.info("DetectionProcess: Stopped.")
