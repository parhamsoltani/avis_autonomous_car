import cv2
import numpy as np
import os
from collections import deque

class CrosswalkDetector:
    def __init__(self, use_yolo=True, show_visualization=False):
        """Initialize crosswalk detector with enhanced horizontal line detection"""
        self.show_visualization = show_visualization
        self.use_yolo = use_yolo
        self.detection_history = deque(maxlen=5)
        
        if self.use_yolo and self._check_yolo_files():
            self._init_yolo()
        else:
            self.use_yolo = False
            print("Using enhanced classical crosswalk detection with horizontal lines")
    
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
            
            # Try CUDA first, fall back to CPU
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for crosswalk detection")
            except:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU for crosswalk detection")
            
            self.classes = ["crosswalk"]
            self.confidence_threshold = 0.5
            print("YOLO crosswalk detector initialized")
            
        except Exception as e:
            print(f"Failed to initialize YOLO: {e}")
            self.use_yolo = False
    
    def detect_horizontal_lines(self, mask):
        """Enhanced horizontal line detection for crosswalks"""
        # Focus on the region where crosswalks appear
        roi = mask[160:180, 96:160]
        
        try:
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=10,
                                   minLineLength=8, maxLineGap=4)
            
            if lines is None:
                return False
            
            # Check for horizontal lines
            horizontal_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle of line
                if x2 - x1 != 0:
                    slope = abs((y2 - y1) / (x2 - x1))
                    # Line is horizontal if slope is close to 0
                    if slope < 0.2:
                        horizontal_count += 1
            
            # Crosswalk detected if we have multiple horizontal lines
            return horizontal_count >= 1
            
        except Exception as e:
            return False
    
    def detect_classical_enhanced(self, frame):
        """Enhanced classical computer vision approach for crosswalk detection"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create white mask for detecting white lines
        white_mask = cv2.inRange(frame, np.array([240, 240, 240]), np.array([255, 255, 255]))
        
        # Apply ROI - focus on bottom half where crosswalks appear
        white_mask[:height//2, :] = 0
        
        # Check for horizontal lines
        if self.detect_horizontal_lines(white_mask):
            confidence = 0.9
            
            if self.show_visualization:
                vis_frame = frame.copy()
                cv2.putText(vis_frame, f"Crosswalk (H-Lines): {confidence:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                cv2.rectangle(vis_frame, (96, 160), (160, 180), (0, 255, 0), 2)
                cv2.imshow("Crosswalk Detection", vis_frame)
            
            return True, confidence
        
        # Fallback: Traditional line pattern detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Focus on bottom half
        roi = thresh[height//2:, :]
        
        # Edge detection
        edges = cv2.Canny(roi, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=width//8, maxLineGap=30)
        
        if lines is None:
            return False, 0.0
        
        # Analyze lines for crosswalk pattern
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Check if line is horizontal
            if angle < 15 or angle > 165:
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > width // 6:
                    horizontal_lines.append(line[0])
        
        # Check for multiple parallel horizontal lines (crosswalk pattern)
        if len(horizontal_lines) >= 3:
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
        """Main detection method - combines YOLO and horizontal line detection"""
        if frame is None or frame.size == 0:
            return False, 0.0
        
        # Try both methods
        detected_yolo = False
        confidence_yolo = 0.0
        
        if self.use_yolo:
            detected_yolo, confidence_yolo = self.detect_yolo(frame)
        
        # Always check for horizontal lines as well
        detected_classical, confidence_classical = self.detect_classical_enhanced(frame)
        
        # Combine results - if either method detects, consider it detected
        detected = detected_yolo or detected_classical
        confidence = max(confidence_yolo, confidence_classical)
        
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