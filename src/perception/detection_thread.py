import threading
import time
import logging

class DetectionThread(threading.Thread):
    def __init__(self, detector):
        """
        Threaded wrapper for object detection.
        Args:
            detector: Instance of ObjectDetector
        """
        super().__init__()
        self.detector = detector
        self.latest_results = None
        self.running = False
        self.input_frame = None
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()
        self.daemon = True

    def process_frame(self, frame):
        """
        Submit a frame for processing. This is non-blocking.
        If the thread is busy, it effectively overwrites the pending frame
        if one was waiting, ensuring we always process the freshest data.
        """
        with self.lock:
            # We store a reference. The frame should not be modified by the caller
            # while detection is running, but usually detection reads.
            self.input_frame = frame
            self.new_frame_event.set()

    def run(self):
        logging.info("DetectionThread: Starting...")
        self.running = True
        while self.running:
            # Wait for a new frame
            if not self.new_frame_event.wait(timeout=0.2):
                continue

            # Reset event and get frame
            with self.lock:
                frame = self.input_frame
                self.new_frame_event.clear()

            if frame is None:
                continue

            try:
                # Run detection (compute intensive)
                # Ensure we are not processing a frame that might be modified elsewhere?
                # YOLO usually copies/converts to tensor.
                results = self.detector.detect(frame)

                with self.lock:
                    self.latest_results = results
            except Exception as e:
                logging.error(f"DetectionThread: Error: {e}")

        logging.info("DetectionThread: Stopped.")

    def stop(self):
        self.running = False
        self.new_frame_event.set() # Wake up thread to exit loop

    def get_latest_results(self):
        """
        Returns the most recent detection results.
        """
        with self.lock:
            return self.latest_results
