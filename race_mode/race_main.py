import cv2 as cv
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avisengine import avisengine
from common.lane_detection import LaneDetector
from common.obstacle_detection import ObstacleDetector
from common.utils import translate, calculate_curve_speed
from race_config import *

class RaceMode:
    def __init__(self, ip='127.0.0.3', port=25004):
        self.car = avisengine.Car()
        self.car.connect(ip, port)
        
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        
        self.current_speed = BASE_SPEED
        self.previous_error = 0
        
    def run(self):
        """Main race mode loop"""
        print("Starting Race Mode...")
        
        while True:
            try:
                # Get data from simulator
                self.car.getData()
                
                # Get sensor data
                left_sens, mid_sens, right_sens = self.car.getSensors()
                
                # Get camera frame
                frame = self.car.getImage()
                
                if frame is not None:
                    # Detect lanes
                    processed_img, lane_center, warped = self.lane_detector.calc_steering(frame)
                    
                    # Calculate error from center
                    error = lane_center - LANE_CENTER_TARGET
                    
                    # PD controller for steering
                    steering = (STEERING_P_GAIN * error + 
                               STEERING_D_GAIN * (error - self.previous_error))
                    self.previous_error = error
                    
                    # Map to steering range
                    steering = translate(steering, -100, 100, -MAX_STEERING, MAX_STEERING)
                    
                    # Check for obstacles
                    obstacle_steer_adj, speed_adj = self.obstacle_detector.process_sensors(
                        left_sens, mid_sens, right_sens)
                    
                    # Combine steering adjustments
                    final_steering = int(np.clip(steering + obstacle_steer_adj, 
                                                 -MAX_STEERING, MAX_STEERING))
                    
                    # Calculate speed based on curve
                    self.current_speed = calculate_curve_speed(final_steering, BASE_SPEED)
                    self.current_speed += speed_adj
                    self.current_speed = int(np.clip(self.current_speed, MIN_SPEED, MAX_SPEED))
                    
                    # Apply controls
                    self.car.setSteering(final_steering)
                    self.car.setSpeed(self.current_speed)
                    
                    # Visualization
                    self.visualize(frame, processed_img, warped, lane_center, 
                                  final_steering, self.current_speed)
                    
                    if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
                        
            except Exception as e:
                print(f"Error in race loop: {e}")
                continue
        
        self.car.stop()
        cv.destroyAllWindows()
    
    def visualize(self, frame, processed, warped, lane_center, steering, speed):
        """Display visualization windows"""
        # Draw info on frame
        show_frame = frame.copy()
        cv.circle(show_frame, (lane_center, 300), 5, (255, 0, 0), cv.FILLED)
        
        # Add text info
        cv.putText(show_frame, f'Speed: {speed}', (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(show_frame, f'Steering: {steering}', (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show windows
        cv.imshow('Race View', show_frame)
        cv.imshow('Lane Detection', processed)
        cv.imshow('Warped View', warped)

if __name__ == "__main__":
    race = RaceMode()
    race.run()