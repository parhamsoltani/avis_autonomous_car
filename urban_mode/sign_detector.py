import cv2
import numpy as np
import os

class TrafficSignDetector:
    def __init__(self, show_visualization=False):
        self.show_visualization = show_visualization
        self.initialized = False
        
        # Sign mapping for simulator
        self.class_mapping = {
            "deadend": 0,
            "No Entry": 1,
            "Stop Sign": 2,
            "Straight Ahead Only": 3,
            "turn-left": 4,
            "turn-right": 5
        }
        
        # Reverse mapping for display
        self.class_names = {v: k for k, v in self.class_mapping.items()}
        
        if self._check_yolo_files():
            self._init_yolo()
    
    def _check_yolo_files(self):
        """Check if YOLO files exist"""
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        weights_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'yolov4-tiny_best.weights')
        config_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'yolov4-tiny.cfg')
        names_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'obj.names')
        
        return all(os.path.exists(p) for p in [weights_path, config_path, names_path])
    
    def _init_yolo(self):
        """Initialize YOLO model for sign detection"""
        try:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            weights_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'yolov4-tiny_best.weights')
            config_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'yolov4-tiny.cfg')
            names_path = os.path.join(base_path, 'yolov4_tiny_traffic_sign_final', 'obj.names')
            
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Try CUDA first, fall back to CPU if not available
            cuda_available = False
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                # Test CUDA with a dummy forward pass
                test_blob = cv2.dnn.blobFromImage(np.zeros((416, 416, 3), dtype=np.uint8), 1/255.0, (416, 416))
                self.net.setInput(test_blob)
                self.net.forward()
                cuda_available = True
                print("Using CUDA for sign detection")
            except Exception as e:
                print(f"CUDA failed for sign detection: {e}")
                cuda_available = False
            
            if not cuda_available:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU for sign detection")
            
            # Load class names
            with open(names_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
            
            self.confidence_threshold = 0.5
            self.initialized = True
            print("YOLO sign detector initialized")
            
        except Exception as e:
            print(f"Failed to initialize YOLO for signs: {e}")
            self.initialized = False

            
    
    def determine_arrow_direction(self, cropped_img):
        """
        Determine arrow direction in sign
        Returns: "turn-left" or "turn-right"
        """
        if cropped_img is None or cropped_img.size == 0:
            return "turn-right"
        
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get arrow shape
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Split image in half
        height, width = thresh.shape
        if width == 0:
            return "turn-right"
        
        mid = width // 2
        left_half = thresh[:, :mid]
        right_half = thresh[:, mid:]
        
        # Count white pixels in each half
        left_pixels = cv2.countNonZero(left_half)
        right_pixels = cv2.countNonZero(right_half)
        
        # More pixels on left means arrow points right, and vice versa
        direction = "turn-right" if left_pixels > right_pixels else "turn-left"
        
        if self.show_visualization:
            debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.line(debug_img, (mid, 0), (mid, height), (0, 255, 0), 1)
            cv2.putText(debug_img, f"L: {left_pixels}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_img, f"R: {right_pixels}", (mid + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_img, direction, (width//4, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("Arrow Analysis", debug_img)
        
        return direction
    
    def detect(self, frame):
        """
        Detect traffic signs in frame
        Returns: sign_number (0-5) or None if no sign detected
        """
        if not self.initialized:
            return None
        
        if frame is None or frame.size == 0:
            return None
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        detected_signs = []
        boxes = []
        confidences = []
        class_ids = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
            
            if len(indices) > 0:
                # Process kept detections
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
                    
                    # For turn signs, determine actual direction
                    if class_id in [4, 5] or "turn" in class_name.lower():
                        # Crop the sign region
                        x_safe = max(0, x)
                        y_safe = max(0, y)
                        x2_safe = min(frame.shape[1], x + w)
                        y2_safe = min(frame.shape[0], y + h)
                        
                        cropped = frame[y_safe:y2_safe, x_safe:x2_safe]
                        if cropped.size > 0:
                            direction = self.determine_arrow_direction(cropped)
                            class_name = direction
                    
                    # Map to number
                    sign_number = self.class_mapping.get(class_name, None)
                    if sign_number is not None:
                        detected_signs.append(sign_number)
                    
                    # Visualization
                    if self.show_visualization:
                        color = (0, 255, 255) if "turn" in class_name else (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if self.show_visualization:
            cv2.imshow("Sign Detection", frame)
        
        # Return most common detection or None
        if detected_signs:
            from collections import Counter
            most_common = Counter(detected_signs).most_common(1)[0][0]
            return most_common
        
        return None
    
    def get_sign_action(self, sign_number):
        """
        Convert sign number to action
        Returns: -1 for left, 0 for straight/stop, 1 for right
        """
        if sign_number is None:
            return 0
        
        if sign_number == 4:  # turn-left
            return -1
        elif sign_number == 5:  # turn-right
            return 1
        else:
            return 0  # straight or stop