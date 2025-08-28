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
        
        # Improved parameters from reference projects
        self.current_speed = 90  # Increased base speed
        self.previous_error = 0
        
        # Sensor smoothing
        self.sensors_array = np.array([1500, 1500, 1500])
        self.sensor_smooth_factor = 0.3
        
        # Steering smoothing
        self.steer_array = np.array(0)
        
        # Position tracking (left/right side of track)
        self.position = 'right'
        self.obstacle_avoidance_timer = 0
        
    def run(self):
        """Main race mode loop with improvements"""
        print("Starting Enhanced Race Mode...")
        counter = 0
        
        while True:
            try:
                # Get data from simulator
                self.car.getData()
                
                if counter > 4:  # Wait for stable connection
                    # Get and smooth sensor data
                    sensors = self.car.getSensors()
                    self.sensors_array = np.round(
                        self.sensor_smooth_factor * np.array(sensors) + 
                        (1 - self.sensor_smooth_factor) * self.sensors_array, 1
                    )
                    
                    # Get camera frame
                    frame = self.car.getImage()
                    
                    if frame is not None:
                        # Enhanced processing
                        steering, speed = self.process_frame_enhanced(frame)
                        
                        # Apply controls
                        self.car.setSteering(int(steering))
                        self.car.setSpeed(int(speed))
                        
                        # Visualization
                        self.visualize_enhanced(frame, steering, speed)
                        
                        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
                            break
                
                counter += 1
                        
            except Exception as e:
                print(f"Error in race loop: {e}")
                continue
        
        self.car.stop()
        cv.destroyAllWindows()
    
    def process_frame_enhanced(self, frame):
        """Enhanced frame processing using techniques from reference projects"""
        import time
        
        # Convert to HSV and apply median blur
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_frame = cv.medianBlur(hsv_frame, 7)
        
        # Enhanced lane detection using reference color ranges
        lane_mask = cv.inRange(hsv_frame, np.array([100, 10, 25]), np.array([120, 50, 60]))
        kernel = np.ones((2, 2), np.uint8)
        lane_mask = cv.erode(lane_mask, kernel, iterations=2)
        kernel = np.ones((3, 3), np.uint8)
        lane_mask = cv.dilate(lane_mask, kernel, iterations=2)
        
        # Find lane contours in specific ROI
        lane_contours, _ = cv.findContours(
            lane_mask[130:200, :], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        sorted_lanes = sorted(lane_contours, key=cv.contourArea, reverse=True)
        
        # Extract lane centers
        CURRENT_PXL = 128
        SECOND_PXL = 128
        
        if len(sorted_lanes) > 0:
            right_lane_mask = cv.drawContours(
                np.zeros((70, 256)), sorted_lanes, 0, 255, -1
            )
            CURRENT_PXL = np.mean(np.where(right_lane_mask > 0), axis=1)[1]
            CURRENT_PXL = np.nan_to_num(CURRENT_PXL, nan=128)
            
            if len(sorted_lanes) > 1:
                left_lane_mask = cv.drawContours(
                    np.zeros((70, 256)), sorted_lanes, 1, 255, -1
                )
                SECOND_PXL = np.mean(np.where(left_lane_mask > 0), axis=1)[1]
                SECOND_PXL = np.nan_to_num(SECOND_PXL, nan=128)
        
        # Detect yellow line for position tracking
        yellow_mask = cv.inRange(hsv_frame, np.array([28, 115, 154]), np.array([31, 180, 255]))
        YELLOW_PXL = np.mean(np.where(yellow_mask[140:190, :] > 0), axis=1)[1]
        YELLOW_PXL = np.nan_to_num(YELLOW_PXL, nan=128)
        
        self.position = 'left' if YELLOW_PXL > 128 else 'right'
        
        # Enhanced obstacle detection
        obstacle_mask = cv.inRange(hsv_frame, np.array([95, 0, 95]), np.array([180, 20, 160]))
        kernel = np.ones((2, 2), np.uint8)
        obstacle_mask = cv.erode(obstacle_mask, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mask = cv.dilate(obstacle_mask, kernel, iterations=1)
        
        # Find obstacle position
        obs_yellow = np.mean(np.where(yellow_mask[65:170, :] > 0), axis=1)[1]
        obs_yellow = np.nan_to_num(obs_yellow, nan=128)
        
        mean_obstacle = 0
        obstacle_points, _ = cv.findContours(
            obstacle_mask[50:200, :], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        if obstacle_points:
            sorted_obs = sorted(obstacle_points, key=cv.contourArea, reverse=True)
            if len(sorted_obs) > 0 and cv.contourArea(sorted_obs[0]) > 10:
                x, y, w, h = cv.boundingRect(sorted_obs[0])
                mean_obstacle = x + w // 2
        
        # Calculate steering with obstacle avoidance
        REFERENCE = 128
        kp = 2.2
        
        if self.position == 'left':
            if (self.sensors_array[0] < 1450 or self.sensors_array[1] < 1450) and (mean_obstacle < obs_yellow):
                error = REFERENCE - SECOND_PXL
                self.obstacle_avoidance_timer = time.time()
                steer = -(kp * error)
            elif (time.time() - self.obstacle_avoidance_timer) > 0.8 and min(self.sensors_array) > 1450:
                error = REFERENCE - SECOND_PXL
                steer = -(kp * error)
            else:
                error = REFERENCE - CURRENT_PXL
                steer = -(kp * error)
        else:
            if (self.sensors_array[2] < 1450 or self.sensors_array[1] < 1450) and (mean_obstacle > obs_yellow):
                error = REFERENCE - SECOND_PXL
                self.obstacle_avoidance_timer = time.time()
                steer = -(kp * error)
            else:
                error = REFERENCE - CURRENT_PXL
                steer = -(kp * error)
        
        # Smooth steering
        self.steer_array = np.round(0.85 * steer + 0.15 * self.steer_array, 1)
        
        # Dynamic speed control
        speed = 90
        if abs(self.steer_array) > 20:
            speed = 70
        elif abs(self.steer_array) > 30:
            speed = 60
        
        return self.steer_array, speed
    
    def visualize_enhanced(self, frame, steering, speed):
        """Enhanced visualization"""
        show_frame = frame.copy()
        
        # Add position indicator
        position_color = (0, 255, 0) if self.position == 'right' else (0, 255, 255)
        cv.putText(show_frame, f'Position: {self.position}', (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, position_color, 2)
        cv.putText(show_frame, f'Speed: {speed}', (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(show_frame, f'Steering: {steering:.1f}', (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw ROI boxes
        cv.rectangle(show_frame, (0, 50), (256, 100), (0, 255, 255), 1)
        cv.rectangle(show_frame, (0, 150), (256, 200), (0, 255, 255), 1)
        
        cv.imshow('Enhanced Race View', show_frame)

if __name__ == "__main__":
    race = RaceMode()
    race.run()