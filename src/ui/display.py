import cv2
import numpy as np

class Display:
    def __init__(self, title="Kinect Detector"):
        self.title = title
        self.map_size = 600
        self.map_scale = 50 # pixels per meter (1m = 50px)

    def show(self, rgb, detections, objects, robot_pose):
        """
        Update and show windows.
        Args:
            rgb: RGB image
            detections: Ultralytics Results object (or None)
            objects: List of map objects
            robot_pose: (x, y, theta)
        Returns:
            key: Key code from waitKey
        """
        # 1. Camera Feed
        if rgb is not None:
            if detections:
                # detections is a ultralytics.engine.results.Results object
                # It has a plot() method that returns numpy array (BGR)
                out_img = detections.plot()
                # If RGB was passed, plot() usually uses the original image stored in results.
                # However, results.orig_img might be the one we passed.
                # Just to be safe, plot() handles it.
            else:
                out_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow(f"{self.title} - Feed", out_img)

        # 2. Map
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # Center of map
        cx, cy = self.map_size // 2, self.map_size // 2

        # Draw Grid (every 1m)
        grid_color = (50, 50, 50)
        for i in range(0, self.map_size, self.map_scale):
            cv2.line(map_img, (i, 0), (i, self.map_size), grid_color, 1)
            cv2.line(map_img, (0, i), (self.map_size, i), grid_color, 1)

        # Draw Robot
        # Invert Y for visualization (up is positive Y in map, up is negative Y in image)
        rx = int(cx + robot_pose[0] * self.map_scale)
        ry = int(cy - robot_pose[1] * self.map_scale)

        if 0 <= rx < self.map_size and 0 <= ry < self.map_size:
            cv2.circle(map_img, (rx, ry), 8, (0, 0, 255), -1) # Red Robot
            # Direction indicator
            # Robot theta is standard math angle (CCW from East)
            # We want to draw a line. In image coords:
            # dx = cos(theta)
            # dy = -sin(theta) (because y is flipped)
            end_x = int(rx + 20 * np.cos(robot_pose[2]))
            end_y = int(ry - 20 * np.sin(robot_pose[2]))
            cv2.line(map_img, (rx, ry), (end_x, end_y), (0, 0, 255), 2)

        # Draw Objects
        for obj in objects:
            ox = int(cx + obj['x'] * self.map_scale)
            oy = int(cy - obj['y'] * self.map_scale)

            if 0 <= ox < self.map_size and 0 <= oy < self.map_size:
                # Yellow for balls
                color = (0, 255, 255)
                cv2.circle(map_img, (ox, oy), 6, color, -1)
                # Text
                label = f"{obj['class_id']}"
                cv2.putText(map_img, label, (ox+8, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(f"{self.title} - Map", map_img)

        return cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
