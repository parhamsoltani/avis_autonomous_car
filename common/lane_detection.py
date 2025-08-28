import cv2 as cv
import numpy as np
from collections import deque

class LaneDetector:
    def __init__(self):
        # Enhanced color ranges from reference projects
        # Better tuned for different lighting conditions
        self.YELLOW_LINE_COLOR = [28, 115, 154, 31, 180, 255]  # Enhanced yellow detection
        self.WHITE_LINE_COLOR = [0, 0, 210, 51, 18, 255]  # Enhanced white detection
        
        # For race mode - blue/purple lane detection
        self.BLUE_LANE_COLOR = [100, 10, 25, 120, 50, 60]  # From reference project
        
        self.LOWER_YELLOW = np.array([self.YELLOW_LINE_COLOR[0], self.YELLOW_LINE_COLOR[1], self.YELLOW_LINE_COLOR[2]])
        self.UPPER_YELLOW = np.array([self.YELLOW_LINE_COLOR[3], self.YELLOW_LINE_COLOR[4], self.YELLOW_LINE_COLOR[5]])
        
        self.LOWER_WHITE = np.array([self.WHITE_LINE_COLOR[0], self.WHITE_LINE_COLOR[1], self.WHITE_LINE_COLOR[2]])
        self.UPPER_WHITE = np.array([self.WHITE_LINE_COLOR[3], self.WHITE_LINE_COLOR[4], self.WHITE_LINE_COLOR[5]])
        
        self.LOWER_BLUE = np.array([self.BLUE_LANE_COLOR[0], self.BLUE_LANE_COLOR[1], self.BLUE_LANE_COLOR[2]])
        self.UPPER_BLUE = np.array([self.BLUE_LANE_COLOR[3], self.BLUE_LANE_COLOR[4], self.BLUE_LANE_COLOR[5]])
        
        self.W, self.H = 512, 512
        self.CAR_CENTER_WARP_FRAME = (250, 210)
        
        # Race points for perspective transform
        self.top_left = (160, 230)
        self.top_right = (352, 230)
        self.bottom_right = (self.W - 30, self.H - 120)
        self.bottom_left = (60, self.H - 120)
        
        # Tracking variables
        self.line_memory = deque(maxlen=5)
        self.steering_memory = deque(maxlen=3)
        
        # Contour filtering
        self.CONTOUR_MIN_SIZE = 250
        self.APPROX_MAX_SIZE = 16

    def warp_frame(self, frame):
        """Apply perspective transform to get bird's eye view"""
        src_points = np.float32([[self.top_left], [self.top_right], 
                                 [self.bottom_right], [self.bottom_left]])
        dst_points = np.float32([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]])
        matrix = cv.getPerspectiveTransform(src_points, dst_points)
        warped_frame = cv.warpPerspective(frame, matrix, (self.W, self.H))
        return warped_frame

    def create_mask_enhanced(self, frame, color_type='blue'):
        """Enhanced color mask creation with better filtering"""
        # Apply median blur first for noise reduction
        frame_blurred = cv.medianBlur(frame, 7)
        
        # Convert to HSV
        img_hsv = cv.cvtColor(frame_blurred, cv.COLOR_BGR2HSV)
        
        # Create mask based on color type
        if color_type == 'yellow':
            mask = cv.inRange(img_hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)
        elif color_type == 'white':
            mask = cv.inRange(img_hsv, self.LOWER_WHITE, self.UPPER_WHITE)
        else:  # blue/race mode
            mask = cv.inRange(img_hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        
        # Enhanced morphological operations
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_large = np.ones((3, 3), np.uint8)
        
        # Remove noise
        mask = cv.erode(mask, kernel_small, iterations=2)
        mask = cv.dilate(mask, kernel_large, iterations=2)
        
        return mask

    def create_mask(self, warped_frame, low, up):
        """Original mask creation method - kept for compatibility"""
        img_hsv = cv.cvtColor(warped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_hsv, low, up)
        
        # Additional filtering
        kernel = np.ones((3,3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        return mask

    def filter_grass(self, frame):
        """Filter out grass/green areas from the frame"""
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Define grass color range (green hues)
        lower_grass = np.array([35, 30, 30])
        upper_grass = np.array([85, 255, 255])
        
        # Create grass mask
        grass_mask = cv.inRange(hsv, lower_grass, upper_grass)
        
        # Invert grass mask to get non-grass areas
        non_grass_mask = cv.bitwise_not(grass_mask)
        
        return non_grass_mask

    def find_line_enhanced(self, frame, mask):
        """Enhanced line finding with better contour selection"""
        white_image = np.ones_like(frame, dtype=np.uint8) * 255
        line_center_x = None
        
        # Find contours with specific ROI
        mask_roi = mask[130:200, :]  # Focus on relevant area
        contours = cv.findContours(mask_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        
        if len(contours) != 0:
            # Sort by area
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
            
            # Take the largest valid contour
            for contour in sorted_contours:
                if cv.contourArea(contour) > self.CONTOUR_MIN_SIZE:
                    # Calculate center
                    M = cv.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"]) + 130  # Add offset for ROI
                        
                        # Draw contour
                        cv.drawContours(white_image[130:200, :], [contour], -1, (255, 0, 255), -1)
                        line_center_x = cx
                        break
        
        return white_image, line_center_x

    def find_line(self, frame, mask):
        """Original find line method - kept for compatibility"""
        white_image = np.ones_like(frame, dtype=np.uint8) * 255
        line_center_x = None
        
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        
        if len(contours) != 0:
            contours_distances_list = self.find_contours_distances(contours)
            
            if len(contours_distances_list) > 0:
                sorted_distances = sorted(contours_distances_list, key=lambda x: x[-1])
                
                if len(sorted_distances) > 0:
                    first_contour = sorted_distances[0][0]
                    white_image = cv.drawContours(white_image, [first_contour], 0, (255, 0, 255), -1)
                    line_center_x = sorted_distances[0][1][0]
        
        return white_image, line_center_x

    def find_contours_distances(self, contours):
        """Calculate distances of contours from car center"""
        distances_of_contours = []
        
        for contour in contours:
            if cv.contourArea(contour) > self.CONTOUR_MIN_SIZE:
                epsilon = 0.01 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < self.APPROX_MAX_SIZE:
                    moments = cv.moments(contour)
                    if moments['m00'] != 0:
                        contour_cx = int(moments['m10'] / moments['m00'])
                        contour_cy = int(moments['m01'] / moments['m00'])
                        
                        distance = np.hypot(contour_cx - self.CAR_CENTER_WARP_FRAME[0],
                                          contour_cy - self.CAR_CENTER_WARP_FRAME[1])
                        
                        distances_of_contours.append((contour, (contour_cx, contour_cy), distance))
        
        return distances_of_contours

    def calc_steering(self, frame):
        """Enhanced steering calculation with multiple detection methods"""
        # Apply grass filter first
        non_grass_mask = self.filter_grass(frame)
        
        # Warp the frame
        warped_frame = self.warp_frame(frame)
        
        # Try enhanced blue lane detection first (for race mode)
        blue_mask = self.create_mask_enhanced(warped_frame, 'blue')
        
        # Find lanes using enhanced method
        lane_img, lane_center = self.find_line_enhanced(frame, blue_mask)
        
        # If no blue lanes found, try yellow and white
        if lane_center is None:
            # Apply grass filter to warped frame
            warped_non_grass = cv.bitwise_and(warped_frame, warped_frame, 
                                             mask=non_grass_mask[:self.H, :self.W])
            
            # Create color masks
            yellow_mask = self.create_mask(warped_non_grass, self.LOWER_YELLOW, self.UPPER_YELLOW)
            white_mask = self.create_mask(warped_non_grass, self.LOWER_WHITE, self.UPPER_WHITE)
            
            # Find lines
            yellow_img, yellow_center_x = self.find_line(frame, yellow_mask)
            white_img, white_center_x = self.find_line(frame, white_mask)
            
            # Calculate average position
            if yellow_center_x is not None and white_center_x is not None:
                avg = (yellow_center_x + white_center_x) // 2
            elif yellow_center_x is not None:
                avg = yellow_center_x + 50  # Offset for single line
            elif white_center_x is not None:
                avg = white_center_x - 50  # Offset for single line
            else:
                # Use memory if available
                if self.line_memory:
                    avg = sum(self.line_memory) // len(self.line_memory)
                else:
                    avg = self.W // 2  # Default to center
            
            lane_center = avg
            
            # Combine images for visualization
            result = np.zeros(yellow_img.shape)
            result += yellow_img
            result += white_img
            result /= 2
            lane_img = result.astype(np.uint8)
        
        # Update memory
        self.line_memory.append(lane_center)
        
        # Smooth the steering
        if len(self.line_memory) > 1:
            lane_center = sum(self.line_memory) // len(self.line_memory)
        
        return lane_img, lane_center, warped_frame