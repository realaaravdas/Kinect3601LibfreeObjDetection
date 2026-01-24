import threading
import queue
import sys

class CLI:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        print("------------------------------------------------")
        print("Command Interface")
        print("Available commands:")
        print("  q, quit   : Exit the program")
        print("  tilt <deg>: Set tilt angle (Kinect 360 only)")
        print("------------------------------------------------")

        while self.running:
            try:
                line = input()
                if line.strip():
                    self.command_queue.put(line.strip())
            except EOFError:
                self.running = False
                break
            except Exception:
                break

    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
