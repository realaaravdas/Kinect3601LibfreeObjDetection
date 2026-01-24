import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from kinect_yellow_ball_detector.kinect_wrapper import Freenect1Device, Freenect2Device, MockKinectDevice
from kinect_yellow_ball_detector.detector import ObjectDetector
from kinect_yellow_ball_detector.slam import VisualOdometry, Mapper

class KinectDriverNode(Node):
    def __init__(self):
        super().__init__('kinect_driver_node')

        self.declare_parameter('device_type', 'mock') # freenect1, freenect2, mock
        self.declare_parameter('model_path', '')

        device_type = self.get_parameter('device_type').value
        model_path = self.get_parameter('model_path').value

        self.get_logger().info(f"Initializing Kinect Driver with type: {device_type}")

        if device_type == 'freenect1':
            self.kinect = Freenect1Device()
            # Run startup sequence
            self.kinect.startup_sequence()
        elif device_type == 'freenect2':
            self.kinect = Freenect2Device()
        else:
            self.kinect = MockKinectDevice()
            self.kinect.startup_sequence()

        self.detector = ObjectDetector(model_path if model_path else None)
        self.vo = VisualOdometry()
        self.mapper = Mapper()

        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(Image, 'kinect/rgb', 10)
        self.depth_pub = self.create_publisher(Image, 'kinect/depth', 10)
        self.map_pub = self.create_publisher(String, 'kinect/map', 10)

        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz

    def timer_callback(self):
        rgb, depth = self.kinect.get_frames()

        # Visual Odometry
        yaw_change = self.vo.process_frame(rgb, depth)
        self.mapper.update_robot_pose(yaw_change)

        # Detection
        detections = self.detector.detect(rgb)

        # Mapping
        self.mapper.process_detections(detections, depth)

        # Visualize detections on RGB
        rgb_vis = rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(rgb_vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish
        try:
            # We are using RGB images throughout, so publish as rgb8
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb_vis, 'rgb8'))
            # Depth can be float32 or uint16. Mock is float32.
            if depth.dtype == np.float32:
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth, '32FC1'))
            else:
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth, 'mono16')) # or whatever

            map_data = self.mapper.get_map()
            robot_pose = self.mapper.robot_pose

            msg = {
                'map': map_data,
                'robot_pose': robot_pose
            }

            # Helper to convert numpy types
            def default(o):
                if isinstance(o, np.float32) or isinstance(o, np.float64): return float(o)
                if isinstance(o, np.int32) or isinstance(o, np.int64): return int(o)
                raise TypeError

            self.map_pub.publish(String(data=json.dumps(msg, default=default)))

        except Exception as e:
            self.get_logger().error(f"Publishing error: {e}")

    def destroy_node(self):
        self.kinect.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = KinectDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
