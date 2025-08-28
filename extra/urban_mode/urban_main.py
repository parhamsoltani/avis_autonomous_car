
import cv2 as cv
import numpy as np
import sys
import os
import time
from enum import Enum
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avisengine import avisengine
from common.lane_detection import LaneDetector
from common.obstacle_detection import ObstacleDetector
from common.utils import translate, calculate_curve_speed
from urban_config import *
from urban_utils_enhanced import UrbanUtils
from sign_detector_enhanced import EnhancedSignDetector

class UrbanState(Enum):
    NORMAL_DRIVING = 1
    APPROACHING_CROSSWALK = 2
    WAITING_AT_CROSSWALK = 3
    TURNING = 4
    OBSTACLE_AVOIDANCE = 5

class UrbanMode:
    def __init__(self, ip='127.0.0.3', port=25004):
        self.car = avisengine.Car()
        self.car.connect(ip, port)
        
        # Initialize enhanced detectors
        self.utils = UrbanUtils()
        self.sign_detector = EnhancedSignDetector()
        
        # Load car mask from reference project
        self.car_mask = self.load_car_mask()
        
        # State management
        self.current_speed = 30  # Slower for urban
        self.state = UrbanState.NORMAL_DRIVING
        self.state_timer = 0
        self.sign_state = 'nothing'
        
        # Control parameters
        self.kp = 2
        self.ki = 0.1
        self.kd = 0.1
        self.previous_error = 0
        self.integral = 0
        
        print("Enhanced Urban Mode initialized")
    
    def load_car_mask(self):
        """Load or create car mask to filter out car body from detection"""
        mask_path = os.path.join(os.path.dirname(__file__), 'car_mask.npy')
        if os.path.exists(mask_path):
            return np.load(mask_path)
        else:
            # Create default mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Mark car body area
            mask[200:256, 80:176] = 1
            return mask
    
    def run(self):
        """Main urban mode loop with enhancements"""
        print("Starting Enhanced Urban Mode...")
        print("Press ESC to exit")
        
        # Initialize
        for _ in range(10):
            self.car.setSteering(0)
            self.car.setSpeed(10)
            self.car.getData()
        
        while True:
            try:
                # Get data from simulator
                self.car.getData()
                sensors = self.car.getSensors()
                frame = self.car.getImage()
                
                if frame is not None:
                    # Process frame with enhanced detection
                    steering, speed = self.process_urban_frame(frame, sensors)
                    
                    # Apply controls
                    self.car.setSteering(int(steering))
                    self.car.setSpeed(int(speed))
                    
                    if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
                        
            except Exception as e:
                print(f"Error in urban loop: {e}")
                continue
        
        self.car.stop()
        cv.destroyAllWindows()
        print("Urban Mode stopped")
    
    def process_urban_frame(self, frame, sensors):
        """Process frame using enhanced urban detection"""
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Apply car mask to remove car body from detection
        mask_inv = 1 - self.car_mask
        
        # Detect white lines for lane following
        white_mask = cv.inRange(frame, np.array([240, 240, 240]), 
                                np.array([255, 255, 255])) * mask_inv
        
        # Detect side walls/barriers
        side_mask = cv.inRange(frame, np.array([130, 0, 108]), 
                              np.array([160, 160, 200])) * mask_inv
        
        # Detect red signs (stop signs)
        red_mask = cv.inRange(hsv_frame, np.array([140, 70, 0]), 
                             np.array([255, 255, 255])) * mask_inv
        
        # Detect lines
        lines = self.utils.detect_lines(self.utils.region_of_interest(white_mask))
        two_line_mask, CURRENT_PXL = self.utils.mean_lines(white_mask, lines)
        
        # Check for horizontal line (crosswalk)
        horiz_detected = self.utils.horiz_lines(white_mask)
        
        # Detect traffic signs
        sign = self.sign_detector.detect_sign(frame, hsv_frame)
        if sign in ['left', 'straight', 'right']:
            self.sign_state = sign
        
        # Check for red sign
        red_sign = self.utils.red_sign_state(red_mask)
        
        # Calculate steering
        REFERENCE = 128
        error = REFERENCE - CURRENT_PXL
        
        # PID control
        self.integral += error * 0.05
        derivative = (error - self.previous_error) / 0.05
        steer = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.previous_error = error
        
        speed = self.current_speed
        
        # Handle crosswalk
        if horiz_detected:
            self.handle_crosswalk(frame, white_mask, side_mask, red_sign)
            return 0, 0  # Stop at crosswalk
        
        # Handle obstacles
        if sensors[1] < 700:
            self.handle_obstacle(side_mask)
            return 0, 0  # Stop for obstacle
        
        # Visualize
        self.visualize_urban(frame, white_mask, two_line_mask, side_mask, steer, speed)
        
        return steer, speed
    
    def handle_crosswalk(self, frame, white_mask, side_mask, red_sign):
        """Enhanced crosswalk handling"""
        print("Crosswalk detected")
        self.utils.stop_the_car(self.car)
        time.sleep(3)
        
        if not red_sign:
            mean_pix = self.utils.turn_where(white_mask)
            side_pix = self.utils.detect_side(side_mask)
            
            if self.sign_state == 'left':
                steering = -45 if side_pix > 128 else -50
                duration = 13 if side_pix > 128 else 12
                self.utils.turn_the_car(self.car, steering, duration)
            elif self.sign_state == 'straight':
                self.utils.turn_the_car(self.car, 0, 11)
            elif self.sign_state == 'right':
                steering = 65 if side_pix > 128 else 70
                duration = 9.5 if side_pix > 128 else 11
                self.utils.turn_the_car(self.car, steering, duration)
            else:
                # Use mean pixel for turn decision
                if mean_pix < 128:
                    if side_pix > 128:
                        self.utils.go_back(self.car, 4.5)
                        self.utils.turn_the_car(self.car, -100, 10)
                    else:
                        self.utils.turn_the_car(self.car, -80, 8)
                else:
                    if side_pix < 128:
                        self.utils.go_back(self.car, 8)
                    else:
                        self.utils.go_back(self.car, 4.5)
                    self.utils.turn_the_car(self.car, 100, 10)
            
            self.sign_state = 'nothing'
        else:
            print("Red sign detected - stopping")
    
    def handle_obstacle(self, side_mask):
        """Enhanced obstacle handling"""
        self.utils.stop_the_car(self.car)
        side_pix = self.utils.detect_side(side_mask)
        print(f'Obstacle detected, side_pix: {side_pix}')
        time.sleep(3)
        
        if side_pix > 128:
            # Obstacle on right, go left
            self.utils.turn_the_car(self.car, -100, 5.5)
            self.utils.turn_the_car(self.car, 100, 6.5)
            self.utils.turn_the_car(self.car, -100, 2.5)
        else:
            # Obstacle on left, go right
            self.utils.turn_the_car(self.car, 100, 4)
    
    def visualize_urban(self, frame, white_mask, two_line_mask, side_mask, steering, speed):
        """Enhanced visualization for urban mode"""
        # Create multi-panel display
        roi_display = cv.cvtColor(self.utils.region_of_interest(white_mask) * 255, 
                                  cv.COLOR_GRAY2BGR)
        two_line_display = cv.cvtColor(two_line_mask, cv.COLOR_GRAY2BGR)
        side_display = cv.cvtColor(side_mask, cv.COLOR_GRAY2BGR)
        
        # Add info to main frame
        show_frame = frame.copy()
        cv.putText(show_frame, f'Sign: {self.sign_state}', (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(show_frame, f'Speed: {speed}', (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(show_frame, f'Steering: {steering:.1f}', (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine displays
        top_row = np.concatenate([show_frame, roi_display], axis=1)
        bottom_row = np.concatenate([two_line_display, side_display], axis=1)
        result = np.concatenate([top_row, bottom_row], axis=0)
        
        # Resize for better visibility
        scale = 0.7
        height, width = result.shape[:2]
        result = cv.resize(result, (int(width * scale), int(height * scale)))
        
        cv.imshow("Urban Perception", result)

if __name__ == "__main__":
    urban = UrbanMode()
    urban.run()