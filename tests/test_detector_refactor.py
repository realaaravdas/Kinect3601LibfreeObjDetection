import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might be missing
sys.modules['rknnlite'] = MagicMock()
sys.modules['rknnlite.api'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock cv2 functions used
sys.modules['cv2'].resize = MagicMock(side_effect=lambda img, size, interpolation: np.zeros((size[1], size[0], 3), dtype=np.uint8))
sys.modules['cv2'].copyMakeBorder = MagicMock(side_effect=lambda img, t, b, l, r, type, value: np.zeros((img.shape[0]+t+b, img.shape[1]+l+r, 3), dtype=np.uint8))
sys.modules['cv2'].dnn = MagicMock()
sys.modules['cv2'].dnn.NMSBoxes = MagicMock(return_value=np.array([0])) # Mock NMS returning first box

# Now import detector
from src.perception.detector import get_detector, RKNNDetector, UltralyticsDetector

class TestDetector(unittest.TestCase):
    def setUp(self):
        pass

    @patch('src.perception.detector.YOLO')
    def test_ultralytics_detector(self, mock_yolo):
        # Setup mock
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock result
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu().numpy().tolist.return_value = [10, 20, 30, 40]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 1
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.9

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        # Test
        detector = UltralyticsDetector("model.pt")
        res = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['xyxy'], [10, 20, 30, 40])
        self.assertEqual(res[0]['cls'], 1)
        self.assertEqual(res[0]['conf'], 0.9)

    @patch('src.perception.detector.RKNNLite')
    def test_rknn_detector(self, mock_rknn_lite):
        # Setup mock
        mock_rknn = MagicMock()
        mock_rknn_lite.return_value = mock_rknn
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0

        # Mock inference output: (1, 84, 8400)
        # We'll create a smaller one for testing: (1, 84, 10)
        # 4 box + 80 classes
        output = np.zeros((1, 84, 10), dtype=np.float32)

        # Set one anchor to be a hit
        # Box: cx=320, cy=320, w=100, h=100 -> x1=270, y1=270, x2=370, y2=370 (in 640x640)
        output[0, 0, 0] = 320 # cx
        output[0, 1, 0] = 320 # cy
        output[0, 2, 0] = 100 # w
        output[0, 3, 0] = 100 # h

        # Score for class 5
        output[0, 4+5, 0] = 0.9 # High score

        mock_rknn.inference.return_value = [output]

        # Test
        detector = RKNNDetector("model.rknn")

        # Mock input image 640x640 so ratio is 1
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        res = detector.detect(img)

        # Check if detect was called
        mock_rknn.inference.assert_called_once()

        # Check result
        self.assertEqual(len(res), 1)
        # Since ratio is 1 and pad is 0, output should match
        # Expected: x1=270, y1=270, x2=370, y2=370
        # cv2.dnn.NMSBoxes is mocked to return index 0

        self.assertEqual(res[0]['cls'], 5)
        self.assertAlmostEqual(res[0]['conf'], 0.9)
        self.assertAlmostEqual(res[0]['xyxy'][0], 270.0)

    def test_get_detector(self):
        with patch('src.perception.detector.RKNNDetector') as mock_rknn:
            get_detector("model.rknn")
            mock_rknn.assert_called_once()

        with patch('src.perception.detector.UltralyticsDetector') as mock_ultra:
            get_detector("model.pt")
            mock_ultra.assert_called_once()

if __name__ == '__main__':
    unittest.main()
