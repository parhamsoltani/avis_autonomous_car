import numpy as np

class ObstacleDetector:
    def __init__(self):
        self.obstacle_threshold = 150  # Distance threshold for obstacle detection
        self.side_threshold = 100      # Distance threshold for side sensors
        
    def process_sensors(self, left_sens, mid_sens, right_sens):
        """
        Process sensor data and determine avoidance action
        Returns: steering_adjustment, speed_adjustment
        """
        steering_adj = 0
        speed_adj = 0
        
        # Check for obstacles
        if mid_sens < self.obstacle_threshold:
            # Obstacle ahead - need to avoid
            if left_sens > right_sens:
                # More space on left, steer left
                steering_adj = -15
            else:
                # More space on right, steer right
                steering_adj = 15
            
            # Slow down when obstacle is detected
            speed_adj = -30
            
        elif left_sens < self.side_threshold:
            # Too close to left wall/obstacle
            steering_adj = 5
            
        elif right_sens < self.side_threshold:
            # Too close to right wall/obstacle
            steering_adj = -5
            
        return steering_adj, speed_adj