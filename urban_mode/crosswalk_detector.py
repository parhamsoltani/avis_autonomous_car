import cv2
import numpy as np
import os
from collections import deque

class CrosswalkDetector:
    def __init__(self, use_yolo=True, show_visualization=False):
        """
        Initialize crosswalk detector
        Args:
            use_yolo: If True, use YOLO model. If False, use classical CV approach
            show_visualization: Show detection visualization
        """
        self.show_visualization = show_visualization
        self.use_yolo = use_yolo
        self.detection_history = deque(maxlen=5)
        
        if self.use_yolo and self._check_yolo_files():
            self._init_yolo()
        else:
            self.use_yolo = False
            print("Using classical crosswalk detection")
    
    def _check_yolo_files(self):
        """Check if YOLO files exist"""
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        weights_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_crosswalk', 'yolov4-tiny_best.weights')
        config_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_crosswalk', 'yolov4-tiny.cfg')
        
        return os.path.exists(weights_path) and os.path.exists(config_path)
    
    
    def _init_yolo(self):
        """Initialize YOLO model for crosswalk detection"""
        try:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            weights_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_crosswalk', 'yolov4-tiny_best.weights')
            config_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_crosswalk', 'yolov4-tiny.cfg')
            
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Try CUDA first, fall back to CPU if not available
            cuda_available = False
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                # Test CUDA with a dummy forward pass
                test_blob = cv2.dnn.blobFromImage(np.zeros((320, 320, 3), dtype=np.uint8), 1/255.0, (320, 320))
                self.net.setInput(test_blob)
                self.net.forward()
                cuda_available = True
                print("Using CUDA for crosswalk detection")
            except Exception as e:
                print(f"CUDA failed for crosswalk detection: {e}")
                cuda_available = False
            
            if not cuda_available:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU for crosswalk detection")
            
            self.classes = ["crosswalk"]
            self.confidence_threshold = 0.5
            print("YOLO crosswalk detector initialized")
            
        except Exception as e:
            print(f"Failed to initialize YOLO: {e}")
            self.use_yolo = False
    
    def detect_classical(self, frame):
        """Classical computer vision approach for crosswalk detection"""
        height, width = frame.shape[:2]
        
        # Focus on bottom half of frame where crosswalks appear
        roi = frame[height//2:, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to get white regions
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(thresh, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=width//8, maxLineGap=30)
        
        if lines is None:
            return False, 0.0
        
        # Analyze lines for crosswalk pattern
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Check if line is horizontal (within 15 degrees of horizontal)
            if angle < 15 or angle > 165:
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > width // 6:  # Filter short lines
                    horizontal_lines.append(line[0])
        
        # Check for crosswalk pattern (multiple parallel horizontal lines)
        if len(horizontal_lines) >= 3:
            # Sort lines by y-coordinate
            horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
            
            # Check for consistent spacing
            if len(horizontal_lines) >= 3:
                spacings = []
                for i in range(len(horizontal_lines) - 1):
                    y1 = (horizontal_lines[i][1] + horizontal_lines[i][3]) / 2
                    y2 = (horizontal_lines[i+1][1] + horizontal_lines[i+1][3]) / 2
                    spacings.append(abs(y2 - y1))
                
                # Check if spacings are consistent (crosswalk pattern)
                if spacings:
                    avg_spacing = np.mean(spacings)
                    std_spacing = np.std(spacings)
                    
                    # Low standard deviation means consistent spacing
                    if avg_spacing > 0 and (std_spacing / avg_spacing) < 0.5:
                        confidence = min(1.0, len(horizontal_lines) / 5.0)
                        
                        if self.show_visualization:
                            vis_frame = frame.copy()
                            for line in horizontal_lines:
                                x1, y1, x2, y2 = line
                                cv2.line(vis_frame, (x1, y1 + height//2), 
                                        (x2, y2 + height//2), (0, 255, 0), 2)
                            cv2.putText(vis_frame, f"Crosswalk: {confidence:.2f}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
                            cv2.imshow("Crosswalk Detection", vis_frame)
                        
                        return True, confidence
        
        return False, 0.0
    
    def detect_yolo(self, frame):
        """YOLO-based crosswalk detection"""
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        detected = False
        max_confidence = 0.0
        best_box = None
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                confidence = scores[0]  # Only one class (crosswalk)
                
                if confidence > self.confidence_threshold:
                    detected = True
                    if confidence > max_confidence:
                        max_confidence = confidence
                        
                        # Get bounding box
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x, y = int(center_x - w/2), int(center_y - h/2)
                        best_box = (x, y, w, h)
        
        if self.show_visualization and detected and best_box:
            vis_frame = frame.copy()
            x, y, w, h = best_box
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Crosswalk: {max_confidence:.2f}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Crosswalk Detection", vis_frame)
        
        return detected, max_confidence
    
    def detect(self, frame):
        """
        Main detection method
        Returns: (is_crosswalk, confidence)
        """
        if frame is None or frame.size == 0:
            return False, 0.0
        
        # Use appropriate detection method
        if self.use_yolo:
            detected, confidence = self.detect_yolo(frame)
        else:
            detected, confidence = self.detect_classical(frame)
        
        # Add to history for stability
        self.detection_history.append(detected)
        
        # Require consistent detection over multiple frames
        if len(self.detection_history) >= 3:
            detection_count = sum(self.detection_history)
            if detection_count >= 3:  # At least 3 out of 5 frames
                return True, confidence
        
        return False, 0.0
    
    def detect_simple(self, frame):
        """Simple detection for quick checks (returns boolean)"""
        detected, _ = self.detect(frame)
        return detected