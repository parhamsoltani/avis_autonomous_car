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
from crosswalk_detector import CrosswalkDetector
from sign_detector import TrafficSignDetector
from apriltag_detector import AprilTagDetector

class UrbanState(Enum):
    NORMAL_DRIVING = 1
    APPROACHING_CROSSWALK = 2
    WAITING_AT_CROSSWALK = 3
    TURNING = 4
    FOLLOWING_SIGN = 5
    APRILTAG_ACTION = 6

class UrbanMode:
    def __init__(self, ip='127.0.0.3', port=25004):
        self.car = avisengine.Car()
        self.car.connect(ip, port)
        
        # Initialize detectors with enhanced settings
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.crosswalk_detector = CrosswalkDetector(use_yolo=True, show_visualization=True)
        self.sign_detector = TrafficSignDetector(show_visualization=True)
        self.apriltag_detector = AprilTagDetector(show_visualization=True)
        
        # Load or create car mask
        self.car_mask = self.load_car_mask()
        
        # State management
        self.current_speed = 30  # Slower for urban
        self.previous_error = 0
        self.state = UrbanState.NORMAL_DRIVING
        self.state_timer = 0
        self.turn_direction = 0
        self.sign_state = 'nothing'
        
        # Control parameters
        self.kp = 2
        self.ki = 0.1
        self.kd = 0.1
        self.integral = 0
        
        # Detection history
        self.sign_history = deque(maxlen=10)
        self.apriltag_history = deque(maxlen=5)
        
        # Reference line position
        self.REFERENCE = 128
        
        print("Enhanced Urban Mode initialized")
    
    def load_car_mask(self):
        """Load or create car mask to filter out car body from detection"""
        mask_path = os.path.join(os.path.dirname(__file__), 'car_mask.npy')
        if os.path.exists(mask_path):
            return np.load(mask_path)
        else:
            # Create default mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Mark car body area (bottom center)
            mask[200:256, 80:176] = 1
            np.save(mask_path, mask)
            print("Created default car mask")
            return mask
    
    def detect_lines_urban(self, image):
        """Detect lines using Hough transform"""
        if image is None or not image.any():
            return []
        
        rho = 1
        angle = np.pi / 180
        min_threshold = 10
        lines = cv.HoughLinesP(image, rho, angle, min_threshold, np.array([]), 
                                minLineLength=8, maxLineGap=4)
        return lines if lines is not None else []
    
    def mean_lines_urban(self, frame, lines):
        """Calculate mean lines from detected lines"""
        a = np.zeros_like(frame)
        current_pix = 128
        
        if lines is None or len(lines) == 0:
            return a, current_pix
        
        try:
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
            
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.5:
                        continue
                    if slope <= 0:
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else:
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
            
            min_y = int(frame.shape[0] * 0.6)
            max_y = int(frame.shape[0])
            
            left_x_end = 0
            right_x_end = 256
            
            if left_line_y:
                poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))
                cv.line(a, (left_x_start, max_y), (left_x_end, min_y), [255, 255, 0], 5)
            
            if right_line_y:
                poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))
                cv.line(a, (right_x_start, max_y), (right_x_end, min_y), [255, 255, 0], 5)
            
            current_pix = (left_x_end + right_x_end) / 2
            
        except Exception as e:
            current_pix = 128
        
        return a, current_pix
    
    def region_of_interest_urban(self, image):
        """Apply region of interest mask"""
        if len(image.shape) != 2:
            return image
        
        height, width = image.shape
        mask = np.zeros_like(image)
        polygon = np.array([[
            (0, height),
            (0, 180),
            (80, 130),
            (256 - 80, 130),
            (width, 180),
            (width, height),
        ]], np.int32)
        
        cv.fillPoly(mask, polygon, 255)
        masked_image = image * mask
        masked_image[:170, :] = 0
        return masked_image
    
    def detect_horizontal_lines(self, mask):
        """Detect horizontal lines (crosswalks)"""
        roi = mask[160:180, 96:160]
        try:
            lines = self.detect_lines_urban(roi)
            if not lines:
                return False
            
            lines = np.array(lines).reshape(-1, 2, 2)
            for line in lines:
                if line[1, 0] != line[0, 0]:
                    slope = abs((line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0]))
                    if slope < 0.2:
                        return True
            return False
        except:
            return False
    
    def detect_turn_direction(self, mask):
        """Determine turn direction based on white lines"""
        roi = mask[100:190, :]
        lines = self.detect_lines_urban(roi)
        
        if not lines:
            return 128
        
        try:
            lines = np.array(lines).reshape(-1, 2, 2)
            horizontal_lines = []
            
            for line in lines:
                if line[1, 0] != line[0, 0]:
                    slope = abs((line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0]))
                    if slope < 0.2:
                        horizontal_lines.append(line)
            
            if horizontal_lines:
                mean_x = np.mean([line[:, :, 0] for line in horizontal_lines])
                return mean_x
        except:
            pass
        
        return 128
    
    def detect_side_position(self, side_mask):
        """Detect side position"""
        side_pix = np.mean(np.where(side_mask[150:190, :] > 0), axis=1)
        if len(side_pix) > 1:
            return side_pix[1]
        return 128
    
    def detect_red_sign(self, red_mask):
        """Detect red stop sign"""
        contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
            if sorted_contours and cv.contourArea(sorted_contours[0]) > 50:
                print('Red sign detected!')
                return True
        return False
    
    def stop_car_safely(self):
        """Safely stop the car"""
        self.car.setSteering(0)
        while self.car.getSpeed() > 0:
            self.car.setSpeed(-100)
            self.car.getData()
        self.car.setSpeed(0)
    
    def execute_turn(self, steering, duration):
        """Execute turn maneuver"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.car.getData()
            self.car.setSteering(steering)
            self.car.setSpeed(15)
    
    def reverse_car(self, duration):
        """Reverse the car"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.car.getData()
            self.car.setSpeed(-15)
        self.car.setSpeed(0)
    
    def run(self):
        """Main urban mode loop with enhancements"""
        print("Starting Enhanced Urban Mode...")
        print("Press ESC to exit")
        
        # Initialize car
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
        roi_mask = self.region_of_interest_urban(white_mask)
        lines = self.detect_lines_urban(roi_mask)
        two_line_mask, CURRENT_PXL = self.mean_lines_urban(white_mask, lines)
        
        # Check for horizontal line (crosswalk)
        horiz_detected = self.detect_horizontal_lines(white_mask)
        
        # Detect traffic signs using existing detector
        sign = self.sign_detector.detect(frame)
        if sign is not None:
            sign_action = self.sign_detector.get_sign_action(sign)
            if sign_action == -1:
                self.sign_state = 'left'
            elif sign_action == 1:
                self.sign_state = 'right'
            elif sign == 3:  # Straight Ahead Only
                self.sign_state = 'straight'
        
        # Check for red sign
        red_sign = self.detect_red_sign(red_mask)
        
        # Calculate steering
        error = self.REFERENCE - CURRENT_PXL
        
        # PID control
        dt = 0.05
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        steer = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.previous_error = error
        
        speed = self.current_speed
        
        # Handle crosswalk
        if horiz_detected:
            print("Crosswalk detected")
            self.stop_car_safely()
            time.sleep(3)
            
            if not red_sign:
                mean_pix = self.detect_turn_direction(white_mask)
                side_pix = self.detect_side_position(side_mask)
                
                if self.sign_state == 'left':
                    steering = -45 if side_pix > 128 else -50
                    duration = 13 if side_pix > 128 else 12
                    self.execute_turn(steering, duration)
                elif self.sign_state == 'straight':
                    self.execute_turn(0, 11)
                elif self.sign_state == 'right':
                    steering = 65 if side_pix > 128 else 70
                    duration = 9.5 if side_pix > 128 else 11
                    self.execute_turn(steering, duration)
                else:
                    # Use mean pixel for turn decision
                    if mean_pix < 128:
                        if side_pix > 128:
                            self.reverse_car(4.5)
                            self.execute_turn(-100, 10)
                        else:
                            self.execute_turn(-80, 8)
                    else:
                        if side_pix < 128:
                            self.reverse_car(8)
                        else:
                            self.reverse_car(4.5)
                        self.execute_turn(100, 10)
                
                self.sign_state = 'nothing'
            else:
                print("Red sign detected - stopping")
                return 0, 0
        
        # Handle obstacles
        if sensors[1] < 700:
            self.stop_car_safely()
            side_pix = self.detect_side_position(side_mask)
            print(f'Obstacle detected, side_pix: {side_pix}')
            time.sleep(3)
            
            if side_pix > 128:
                # Obstacle on right, go left
                self.execute_turn(-100, 5.5)
                self.execute_turn(100, 6.5)
                self.execute_turn(-100, 2.5)
            else:
                # Obstacle on left, go right
                self.execute_turn(100, 4)
        
        # Visualize
        self.visualize_urban(frame, white_mask, two_line_mask, side_mask, steer, speed)
        
        return steer, speed
    
    def visualize_urban(self, frame, white_mask, two_line_mask, side_mask, steering, speed):
        """Enhanced visualization for urban mode"""
        # Create displays
        roi_display = cv.cvtColor(self.region_of_interest_urban(white_mask) * 255, 
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
        cv.putText(show_frame, f'State: {self.state.name}', (10, 120),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Combine displays
        top_row = np.concatenate([show_frame, roi_display], axis=0)
        bottom_row = np.concatenate([two_line_display, side_display], axis=0)
        result = np.concatenate([top_row, bottom_row], axis=1)
        
        # Resize for better visibility
        scale = 0.5
        height, width = result.shape[:2]
        result = cv.resize(result, (int(width * scale), int(height * scale)))
        
        cv.imshow("Urban Perception", result)

if __name__ == "__main__":
    urban = UrbanMode()
    urban.run()