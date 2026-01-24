from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QPen, QColor
import numpy as np
import json
import cv2

class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.balls = []
        self.robot_pose = {'x': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.scale = 0.1 # pixels per mm (1000mm = 100px)
        self.center_x = 400
        self.center_y = 500

    def update_data(self, balls, robot_pose):
        self.balls = balls
        self.robot_pose = robot_pose
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), Qt.white)

        # Draw grid
        painter.setPen(QPen(Qt.gray, 1, Qt.DotLine))
        for x in range(0, self.width(), 50):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 50):
            painter.drawLine(0, y, self.width(), y)

        # Draw Robot
        rx = self.robot_pose['x'] * self.scale + self.center_x
        ry = self.center_y - self.robot_pose['z'] * self.scale

        painter.setBrush(QBrush(Qt.blue))
        painter.drawEllipse(int(rx)-10, int(ry)-10, 20, 20)

        # Draw direction
        theta = np.radians(self.robot_pose['yaw'])

        dx = 30 * np.sin(theta)
        dy = -30 * np.cos(theta)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawLine(int(rx), int(ry), int(rx+dx), int(ry+dy))

        # Draw balls
        painter.setBrush(QBrush(Qt.yellow))
        painter.setPen(QPen(Qt.black))

        for ball in self.balls:
            bx = ball['x'] * self.scale + self.center_x
            bz = self.center_y - ball['z'] * self.scale
            painter.drawEllipse(int(bx)-8, int(bz)-8, 16, 16)
            painter.drawText(int(bx)+10, int(bz), f"ID:{ball['id']}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Yellow Ball Detector")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Video Splitter
        video_splitter = QSplitter(Qt.Vertical)
        self.rgb_label = QLabel("RGB Feed")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setStyleSheet("border: 1px solid black;")

        self.depth_label = QLabel("Depth Feed")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setStyleSheet("border: 1px solid black;")

        video_splitter.addWidget(self.rgb_label)
        video_splitter.addWidget(self.depth_label)

        # Map
        self.map_widget = MapWidget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(video_splitter)
        splitter.addWidget(self.map_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def update_rgb(self, image):
        if image is None: return
        h, w, ch = image.shape
        bytes_per_line = ch * w
        # Image is already RGB, so no need to swap
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        self.rgb_label.setPixmap(p.scaled(self.rgb_label.width(), self.rgb_label.height(), Qt.KeepAspectRatio))

    def update_depth(self, image):
        if image is None: return

        # Normalize for display
        if image.dtype == np.float32:
            disp_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            # handle NaN
            disp_img = np.nan_to_num(disp_img, nan=0).astype(np.uint8)
        else:
            disp_img = (image / 256).astype(np.uint8)

        h, w = disp_img.shape
        bytes_per_line = w
        convert_to_Qt_format = QImage(disp_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        p = QPixmap.fromImage(convert_to_Qt_format)
        self.depth_label.setPixmap(p.scaled(self.depth_label.width(), self.depth_label.height(), Qt.KeepAspectRatio))

    def update_map(self, map_json):
        try:
            data = json.loads(map_json)
            self.map_widget.update_data(data['map'], data['robot_pose'])
        except Exception as e:
            pass
