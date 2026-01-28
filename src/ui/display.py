import cv2
import numpy as np

class Display:
    def __init__(self, title="Kinect System"):
        self.title = title
        self.W = 800
        self.H = 800
        self.scale = 50 # px per meter
        # Center the Robot at (W/2, H*0.8)
        self.cx = self.W // 2
        self.cy = int(self.H * 0.8)

    def world_to_screen(self, wx, wy, robot_pose):
        """
        Transform World(wx, wy) to Screen(sx, sy) relative to Robot.
        Robot is always at (cx, cy) pointing UP.
        """
        rx, ry, rtheta = robot_pose

        # Translate to Robot
        dx = wx - rx
        dy = wy - ry

        # Rotate by -rtheta + 90deg (since Robot Theta is CCW from East,
        # and we want Robot Forward (Theta) to be UP (North/Screen -Y)).
        # Screen Up is -Y.
        # Angle relative to robot heading:
        # rel_angle = atan2(dy, dx) - rtheta
        # x_rel = dist * cos(rel_angle) -> Right on screen
        # y_rel = dist * sin(rel_angle) -> Up on screen (negative)

        # Rotation Matrix to align X-axis with Robot Heading
        c = np.cos(-rtheta)
        s = np.sin(-rtheta)

        # Rotate so X is Forward
        # x_rob = dx * cos + dy * sin (Forward)
        # y_rob = -dx * sin + dy * cos (Left)

        x_rob = dx * c - dy * s
        y_rob = dx * s + dy * c

        # Screen Coords:
        # Screen X (Right) = -y_rob (since y_rob is Left)
        # Screen Y (Down) = -x_rob (since x_rob is Forward)

        sx = int(self.cx - y_rob * self.scale)
        sy = int(self.cy - x_rob * self.scale)

        return sx, sy

    def show(self, rgb, detections, objects, robot_pose, frustum_poly):
        # 1. Feed
        if rgb is not None:
             # Draw detections on Feed
            img_feed = rgb.copy()
            # Convert RGB to BGR for OpenCV
            img_feed = cv2.cvtColor(img_feed, cv2.COLOR_RGB2BGR)

            # Since detections are async, they might not match frame exactly,
            # but we can try drawing them if we have them.
            # Detections here are usually passed as list of dicts now
            if detections:
                for det in detections:
                     x1, y1, x2, y2 = map(int, det['xyxy'])
                     cv2.rectangle(img_feed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     label = f"{det['cls']} {det['conf']:.2f}"
                     cv2.putText(img_feed, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(f"{self.title} - Feed", img_feed)

        # 2. Map
        # White Background
        map_img = np.ones((self.H, self.W, 3), dtype=np.uint8) * 255

        # Draw Grid
        grid_color = (220, 220, 220)
        # Draw local grid relative to robot
        # It's easier to just draw lines at fixed screen intervals? No, moving grid.
        # Skip grid for now or make simple crosshair.
        cv2.line(map_img, (0, self.cy), (self.W, self.cy), grid_color, 1)
        cv2.line(map_img, (self.cx, 0), (self.cx, self.H), grid_color, 1)

        # Draw Frustum
        if frustum_poly:
            pts = []
            for pt in frustum_poly:
                sx, sy = self.world_to_screen(pt[0], pt[1], robot_pose)
                pts.append([sx, sy])
            pts = np.array(pts, np.int32)
            cv2.polylines(map_img, [pts], True, (200, 200, 200), 2)

        # Draw Robot (Always at Center)
        cv2.circle(map_img, (self.cx, self.cy), 10, (0, 0, 255), -1)

        # Draw Objects
        rx, ry, _ = robot_pose

        for obj in objects:
            sx, sy = self.world_to_screen(obj['x'], obj['y'], robot_pose)

            # Draw if on screen
            if 0 <= sx < self.W and 0 <= sy < self.H:
                # Color based on class
                color = (255, 0, 0) # Blue
                cv2.circle(map_img, (sx, sy), 8, color, -1)

                # Distance
                dist = np.sqrt((obj['x'] - rx)**2 + (obj['y'] - ry)**2)

                # Text Box
                label = f"ID:{obj['id']} D:{dist:.1f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Box bg
                cv2.rectangle(map_img, (sx - 10, sy - 25 - th), (sx - 10 + tw + 4, sy - 20), (255, 255, 255), -1)
                cv2.rectangle(map_img, (sx - 10, sy - 25 - th), (sx - 10 + tw + 4, sy - 20), (0, 0, 0), 1)

                cv2.putText(map_img, label, (sx - 8, sy - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(f"{self.title} - Map", map_img)
        return cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
