import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

class EnhancedSignDetector:
    """Enhanced sign detector using the better model from reference project"""
    
    def __init__(self):
        self.model = None
        self.sign_types = ['left', 'straight', 'right']
        self.load_model()
    
    def load_model(self):
        """Load the enhanced sign detection model"""
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.h5')
        
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print("Enhanced sign model loaded successfully")
            except Exception as e:
                print(f"Failed to load sign model: {e}")
                self.create_fallback_model()
        else:
            print("Sign model not found, creating fallback")
            self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a simple fallback model if the enhanced model is not available"""
        # This is a placeholder - you should use the actual best_model.h5
        pass
    
    def detect_sign(self, frame, hsv_frame):
        """Detect traffic signs in frame"""
        if self.model is None:
            return self.detect_sign_classical(frame, hsv_frame)
        
        # Use color mask to find sign regions
        mask = cv2.inRange(hsv_frame, np.array([100, 160, 90]), np.array([160, 220, 220]))
        mask[:30, :] = 0  # Remove top portion
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 'nothing'
            
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if cv2.contourArea(sorted_contours[0]) > 30:
                x, y, w, h = cv2.boundingRect(sorted_contours[0])
                
                # Validate bounding box
                if 5 < x < 251 and 5 < y < 251 and x + w < 251 and y + h < 251:
                    sign = frame[y:y+h, x:x+w]
                    sign = cv2.resize(sign, (25, 25)) / 255.0
                    
                    # Predict using model
                    prediction = self.model.predict(sign.reshape(1, 25, 25, 3), verbose=0)
                    sign_type = self.sign_types[np.argmax(prediction)]
                    
                    # Draw bounding box for visualization
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
                    return sign_type
        except Exception as e:
            print(f"Error in sign detection: {e}")
        
        return 'nothing'
    
    def detect_sign_classical(self, frame, hsv_frame):
        """Classical computer vision fallback for sign detection"""
        # Detect blue signs
        blue_mask = cv2.inRange(hsv_frame, np.array([100, 100, 50]), np.array([130, 255, 255]))
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'nothing'
        
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return 'nothing'
        
        x, y, w, h = cv2.boundingRect(largest)
        sign_roi = frame[y:y+h, x:x+w]
        
        # Simple arrow detection
        if sign_roi.size == 0:
            return 'nothing'
        
        # Convert to grayscale
        gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
        
        # Check which side has more white pixels (arrow)
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_white = cv2.countNonZero(left_half > 200)
        right_white = cv2.countNonZero(right_half > 200)
        
        if left_white > right_white * 1.5:
            return 'left'
        elif right_white > left_white * 1.5:
            return 'right'
        else:
            return 'straight'