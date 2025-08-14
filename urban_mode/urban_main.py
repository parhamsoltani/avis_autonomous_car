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
        
        # Initialize detectors
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.crosswalk_detector = CrosswalkDetector(use_yolo=True, show_visualization=True)
        self.sign_detector = TrafficSignDetector(show_visualization=True)
        self.apriltag_detector = AprilTagDetector(show_visualization=True)
        
        # State management
        self.current_speed = BASE_SPEED
        self.previous_error = 0
        self.state = UrbanState.NORMAL_DRIVING
        self.state_timer = 0
        self.turn_direction = 0  # -1: left, 0: straight, 1: right
        
        # Detection history
        self.sign_history = deque(maxlen=10)
        self.apriltag_history = deque(maxlen=5)
        
        print("Urban Mode initialized")
    
    def run(self):
        """Main urban mode loop"""
        print("Starting Urban Mode...")
        print("Press ESC to exit")
        
        frame_count = 0
        
        while True:
            try:
                # Get data from simulator
                self.car.getData()
                
                # Get sensor data
                left_sens, mid_sens, right_sens = self.car.getSensors()
                
                # Get camera frame
                frame = self.car.getImage()
                
                if frame is not None:
                    frame_count += 1
                    
                    # Process based on current state
                    if self.state == UrbanState.NORMAL_DRIVING:
                        self.normal_driving(frame, left_sens, mid_sens, right_sens)
                        
                    elif self.state == UrbanState.APPROACHING_CROSSWALK:
                        self.approach_crosswalk(frame)
                        
                    elif self.state == UrbanState.WAITING_AT_CROSSWALK:
                        self.wait_at_crosswalk(frame)
                        
                    elif self.state == UrbanState.TURNING:
                        self.execute_turn(frame)
                        
                    elif self.state == UrbanState.FOLLOWING_SIGN:
                        self.follow_sign(frame)
                        
                    elif self.state == UrbanState.APRILTAG_ACTION:
                        self.handle_apriltag(frame)
                    
                    if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
                        
            except Exception as e:
                print(f"Error in urban loop: {e}")
                continue
        
        self.car.stop()
        cv.destroyAllWindows()
        print("Urban Mode stopped")
    
    def normal_driving(self, frame, left_sens, mid_sens, right_sens):
        """Normal urban driving behavior"""
        # Detect lanes
        processed_img, lane_center, warped = self.lane_detector.calc_steering(frame)
        
        # Check for crosswalk
        is_crosswalk, confidence = self.crosswalk_detector.detect(frame)
        if is_crosswalk and confidence > 0.5:
            self.state = UrbanState.APPROACHING_CROSSWALK
            self.state_timer = time.time()
            print(f"Crosswalk detected with confidence {confidence:.2f}")
            return
        
        # Check for traffic signs
        sign = self.sign_detector.detect(frame)
        if sign is not None:
            self.sign_history.append(sign)
            # Need consistent detection
            if len(self.sign_history) >= 5:
                from collections import Counter
                most_common = Counter(self.sign_history).most_common(1)[0]
                if most_common[1] >= 3:  # At least 3 out of 5 frames
                    self.turn_direction = self.sign_detector.get_sign_action(most_common[0])
                    self.state = UrbanState.FOLLOWING_SIGN
                    self.state_timer = time.time()
                    print(f"Sign detected: {self.sign_detector.class_names.get(most_common[0], 'Unknown')}")
                    self.sign_history.clear()
                    return
        
        # Check for AprilTags
        apriltag = self.apriltag_detector.detect_simple(frame)
        if apriltag is not None:
            self.apriltag_history.append(apriltag)
            if len(self.apriltag_history) >= 3:
                from collections import Counter
                most_common = Counter(self.apriltag_history).most_common(1)[0]
                if most_common[1] >= 2:
                    self.turn_direction = most_common[0]
                    self.state = UrbanState.APRILTAG_ACTION
                    self.state_timer = time.time()
                    print(f"AprilTag action: {most_common[0]}")
                    self.apriltag_history.clear()
                    return
        
        # Calculate steering based on lane detection
        error = lane_center - 256
        steering = (STEERING_P_GAIN * error + 
                   STEERING_D_GAIN * (error - self.previous_error))
        self.previous_error = error
        
        steering = translate(steering, -100, 100, -MAX_STEERING, MAX_STEERING)
        
        # Check obstacles
        obstacle_steer_adj, speed_adj = self.obstacle_detector.process_sensors(
            left_sens, mid_sens, right_sens)
        
        final_steering = int(np.clip(steering + obstacle_steer_adj, 
                                     -MAX_STEERING, MAX_STEERING))
        
        # Urban speed (slower than race)
        self.current_speed = BASE_SPEED + speed_adj
        self.current_speed = int(np.clip(self.current_speed, MIN_SPEED, MAX_SPEED))
        
        # Apply controls
        self.car.setSteering(final_steering)
        self.car.setSpeed(self.current_speed)
        
        # Visualize
        self.visualize(frame, processed_img, "Normal Driving", final_steering, self.current_speed)
    
    def approach_crosswalk(self, frame):
        """Slow down and stop at crosswalk"""
        # Gradually reduce speed
        self.current_speed = max(0, self.current_speed - 10)
        self.car.setSpeed(self.current_speed)
        self.car.setSteering(0)
        
        if self.current_speed == 0:
            self.state = UrbanState.WAITING_AT_CROSSWALK
            self.state_timer = time.time()
            print("Stopped at crosswalk")
        
        self.visualize(frame, None, "Approaching Crosswalk", 0, self.current_speed)
    
    def wait_at_crosswalk(self, frame):
        """Wait at crosswalk for required time"""
        elapsed = time.time() - self.state_timer
        
        self.car.setSpeed(0)
        self.car.setSteering(0)
        
        # Continue checking for signs while waiting
        sign = self.sign_detector.detect(frame)
        if sign is not None:
            action = self.sign_detector.get_sign_action(sign)
            if action != 0:
                self.turn_direction = action
                print(f"Sign detected at crosswalk: {self.sign_detector.class_names.get(sign, 'Unknown')}")
        
        # Check for AprilTags too
        apriltag = self.apriltag_detector.detect_simple(frame)
        if apriltag is not None and apriltag != 0:
            self.turn_direction = apriltag
            print(f"AprilTag detected at crosswalk")
        
        if elapsed >= CROSSWALK_WAIT_TIME:
            if self.turn_direction != 0:
                self.state = UrbanState.TURNING
                self.state_timer = time.time()
                direction_text = "left" if self.turn_direction == -1 else "right"
                print(f"Starting turn: {direction_text}")
            else:
                self.state = UrbanState.NORMAL_DRIVING
                print("Resuming normal driving")
        
        remaining = CROSSWALK_WAIT_TIME - elapsed
        self.visualize(frame, None, f"Waiting: {remaining:.1f}s", 0, 0)
    
    def execute_turn(self, frame):
        """Execute turn based on sign or AprilTag"""
        elapsed = time.time() - self.state_timer
        
        # Set steering and speed based on turn direction
        if self.turn_direction == -1:  # Left turn
            steering = -20
            speed = 30
        elif self.turn_direction == 1:  # Right turn
            steering = 20
            speed = 30
        else:  # Straight
            steering = 0
            speed = BASE_SPEED
        
        self.car.setSteering(steering)
        self.car.setSpeed(speed)
        
        if elapsed >= TURN_DURATION:
            self.state = UrbanState.NORMAL_DRIVING
            self.turn_direction = 0
            print("Turn complete")
        
        direction_text = "left" if self.turn_direction == -1 else ("right" if self.turn_direction == 1 else "straight")
        self.visualize(frame, None, f"Turning {direction_text}", steering, speed)
    
    def follow_sign(self, frame):
        """Follow traffic sign instruction"""
        # Similar to turn execution
        self.execute_turn(frame)
    
    def handle_apriltag(self, frame):
        """Handle AprilTag instruction"""
        # Similar to turn execution
        self.execute_turn(frame)
    
    def visualize(self, frame, processed, status, steering, speed):
        """Display visualization"""
        show_frame = frame.copy()
        
        # Add status info
        cv.putText(show_frame, f'Status: {status}', (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(show_frame, f'Speed: {speed}', (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(show_frame, f'Steering: {steering}', (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(show_frame, f'State: {self.state.name}', (10, 120),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv.imshow('Urban View', show_frame)
        if processed is not None:
            cv.imshow('Lane Detection', processed)

if __name__ == "__main__":
    urban = UrbanMode()
    urban.run()