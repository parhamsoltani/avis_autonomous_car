import cv2
import numpy as np
import os
import onnxruntime as ort

class SignDetectorONNX:
    def __init__(self, show_visualization=False):
        self.show_visualization = show_visualization

        # Load ONNX model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'models', 'sign_detection.onnx')

        if os.path.exists(model_path):
            self.model = ort.InferenceSession(model_path)
            print("Loaded sign detection ONNX model")
            self.initialized = True
        else:
            print("Warning: sign_detection.onnx not found")
            self.model = None
            self.initialized = False

        # Classes from reference project
        self.classes = ['Proceed Forward', 'Proceed Left', 'Proceed Right',
                       'Stop', 'traffic light']

        # Detection parameters
        self.CONF_THRES = 0.5
        self.NMS_THRES = 0.6

        # Mapping to action
        self.action_map = {
            'Proceed Forward': 'forward',
            'Proceed Left': 'left',
            'Proceed Right': 'right',
            'Stop': 'stop',
            'traffic light': 'traffic_light'
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def iou(self, b1, b2):
        """Calculate intersection over union"""
        x1, y1, x2, y2 = b1
        X1, Y1, X2, Y2 = b2
        inter = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))
        area1 = (x2-x1) * (y2-y1)
        area2 = (X2-X1) * (Y2-Y1)
        return inter / (area1 + area2 - inter + 1e-6)

    def detect(self, frame):
        """
        Detect signs in frame using ONNX model
        Returns: List of detections with format [x1, y1, x2, y2, label, confidence]
        """
        if not self.initialized or frame is None:
            return []

        orig_h, orig_w = frame.shape[:2]

        # Preprocess input (384x384 as in reference)
        inp = cv2.resize(frame, (384, 384)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]  # (1, 3, 384, 384)

        try:
            # Run inference
            outputs = self.model.run(None, {"images": inp})
            out0 = outputs[0]          # (1, 9, 3024)
            out0 = out0.squeeze(0)     # (9, 3024)
            out0 = out0.transpose(1, 0) # (3024, 9)

            boxes = out0[:, :4]        # (3024, 4)
            scores = out0[:, 4:]       # (3024, 5)

            detections = []
            for i, box in enumerate(boxes):
                probs = scores[i]
                cid = probs.argmax()
                conf = probs[cid]

                if conf < self.CONF_THRES:
                    continue

                xc, yc, w, h = box
                x1 = (xc - w/2) / 384 * orig_w
                y1 = (yc - h/2) / 384 * orig_h
                x2 = (xc + w/2) / 384 * orig_w
                y2 = (yc + h/2) / 384 * orig_h

                detections.append([x1, y1, x2, y2, self.classes[cid], conf])

            # Apply NMS
            detections.sort(key=lambda x: x[5], reverse=True)
            final_dets = []
            while detections:
                best = detections.pop(0)
                final_dets.append(best)
                detections = [d for d in detections if self.iou(d[:4], best[:4]) < self.NMS_THRES]

            if self.show_visualization:
                self.visualize(frame, final_dets)

            return final_dets

        except Exception as e:
            print(f"Error in sign detection: {e}")
            return []

    def get_primary_detection(self, detections):
        """
        Get the most relevant detection based on confidence and position
        Returns: (action, confidence, bbox) or (None, 0, None)
        """
        if not detections:
            return None, 0, None

        # Filter high confidence detections
        high_conf = [d for d in detections if d[5] > 0.8]

        if not high_conf:
            return None, 0, None

        # Get the detection with highest confidence
        best = high_conf[0]
        action = self.action_map.get(best[4], None)

        return action, best[5], best[:4]

    def visualize(self, frame, detections):
        """Draw detections on frame"""
        for x1, y1, x2, y2, label, conf in detections:
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                       (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Sign Detection", frame)