import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import threading
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QObject
from kinect_yellow_ball_detector.gui import MainWindow

class GuiUpdater(QObject):
    rgb_signal = pyqtSignal(object)
    depth_signal = pyqtSignal(object)
    map_signal = pyqtSignal(str)

class VisualizationNode(Node):
    def __init__(self, updater):
        super().__init__('visualization_node')
        self.updater = updater
        self.bridge = CvBridge()

        self.create_subscription(Image, 'kinect/rgb', self.rgb_callback, 10)
        self.create_subscription(Image, 'kinect/depth', self.depth_callback, 10)
        self.create_subscription(String, 'kinect/map', self.map_callback, 10)

    def rgb_callback(self, msg):
        try:
            # Receive as RGB (or convert to RGB if it was BGR, but we know it is rgb8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.updater.rgb_signal.emit(cv_image)
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            # Depth could be 32FC1 or mono16
            if msg.encoding == '32FC1':
                cv_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'mono16')
            self.updater.depth_signal.emit(cv_image)
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def map_callback(self, msg):
        self.updater.map_signal.emit(msg.data)

def ros_spin(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)
    updater = GuiUpdater()
    window = MainWindow()

    updater.rgb_signal.connect(window.update_rgb)
    updater.depth_signal.connect(window.update_depth)
    updater.map_signal.connect(window.update_map)

    node = VisualizationNode(updater)

    thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    thread.start()

    window.show()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
