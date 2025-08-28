import cv2
import numpy as np
import os

class TrafficSignDetector:
    def __init__(self, show_visualization=False):
        self.show_visualization = show_visualization
        self.initialized = False
        self.model = None
        
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
        
        # Try to load the enhanced model first
        self._try_load_enhanced_model()
        
        if not self.model and self._check_yolo_files():
            self._init_yolo()
    
    def _try_load_enhanced_model(self):
        """Try to load the best_model.h5 from reference project"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Look for best_model.h5 in urban_mode directory
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')
            
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                self.model_type = 'enhanced'
                self.sign_types = ['left', 'straight', 'right']
                print("Loaded enhanced sign detection model (best_model.h5)")
            else:
                print("best_model.h5 not found, will use YOLO instead")
        except Exception as e:
            print(f"Could not load enhanced model: {e}")
            self.model = None
    
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
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for sign detection")
            except:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU for sign detection")
            
            # Load class names
            with open(names_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
            
            self.confidence_threshold = 0.5
            self.initialized = True
            self.model_type = 'yolo'
            print("YOLO sign detector initialized")
            
        except Exception as e:
            print(f"Failed to initialize YOLO for signs: {e}")
            self.initialized = False
    
    def detect_with_enhanced_model(self, frame):
        """Detect signs using the enhanced model (best_model.h5)"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use color mask to find blue sign regions
        mask = cv2.inRange(hsv_frame, np.array([100, 160, 90]), np.array([160, 220, 220]))
        mask[:30, :] = 0  # Remove top portion
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if cv2.contourArea(sorted_contours[0]) > 30:
                x, y, w, h = cv2.boundingRect(sorted_contours[0])
                
                # Validate bounding box
                if 5 < x < 251 and 5 < y < 251 and x + w < 251 and y + h < 251:
                    sign = frame[y:y+h, x:x+w]
                    sign_resized = cv2.resize(sign, (25, 25)) / 255.0
                    
                    # Predict using model
                    prediction = self.model.predict(sign_resized.reshape(1, 25, 25, 3), verbose=0)
                    sign_idx = np.argmax(prediction)
                    confidence = prediction[0][sign_idx]
                    
                    if confidence > 0.7:  # Confidence threshold
                        sign_type = self.sign_types[sign_idx]
                        
                        # Map to number
                        if sign_type == 'left':
                            sign_number = 4  # turn-left
                        elif sign_type == 'right':
                            sign_number = 5  # turn-right
                        elif sign_type == 'straight':
                            sign_number = 3  # Straight Ahead Only
                        else:
                            sign_number = None
                        
                        if self.show_visualization:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            cv2.putText(frame, f"{sign_type}: {confidence:.2f}", (x, y-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        return sign_number
        except Exception as e:
            print(f"Error in enhanced sign detection: {e}")
        
        return None
    
    def determine_arrow_direction(self, cropped_img):
        """Determine arrow direction in sign"""
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
        
        return direction
    
    def detect(self, frame):
        """Main detection method - uses enhanced model if available, else YOLO"""
        if frame is None or frame.size == 0:
            return None
        
        # Use enhanced model if available
        if self.model is not None and hasattr(self, 'model_type') and self.model_type == 'enhanced':
            return self.detect_with_enhanced_model(frame)
        
        # Otherwise use YOLO
        if not self.initialized:
            return None
        
        # YOLO detection code (existing)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
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
        """Convert sign number to action"""
        if sign_number is None:
            return 0
        
        if sign_number == 4:  # turn-left
            return -1
        elif sign_number == 5:  # turn-right
            return 1
        else:
            return 0  # straight or stop