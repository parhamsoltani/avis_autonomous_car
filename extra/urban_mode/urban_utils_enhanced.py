import cv2
import numpy as np
import time

class UrbanUtils:
    """Enhanced urban utilities based on reference project"""
    
    def detect_lines(self, image):
        """Detect lines using Hough transform"""
        if image is None or not image.any():
            return []
        
        rho = 1
        angle = np.pi / 180
        min_threshold = 10
        lines = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), 
                                minLineLength=8, maxLineGap=4)
        return lines if lines is not None else []
    
    def mean_lines(self, frame, lines):
        """Calculate mean lines from detected lines"""
        a = np.zeros_like(frame)
        current_pix = 128
        
        if not lines:
            return a, current_pix
        
        try:
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
            
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
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
            
            if left_line_y:
                poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))
                cv2.line(a, (left_x_start, max_y), (left_x_end, min_y), [255, 255, 0], 5)
            else:
                left_x_end = 0
            
            if right_line_y:
                poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))
                cv2.line(a, (right_x_start, max_y), (right_x_end, min_y), [255, 255, 0], 5)
            else:
                right_x_end = 256
            
            current_pix = (left_x_end + right_x_end) / 2
            
        except Exception as e:
            print(f"Error in mean_lines: {e}")
            current_pix = 128
        
        return a, current_pix
    
    def region_of_interest(self, image):
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
        
        cv2.fillPoly(mask, polygon, 255)
        masked_image = image * mask
        masked_image[:170, :] = 0
        return masked_image
    
    def horiz_lines(self, mask):
        """Detect horizontal lines (crosswalks)"""
        roi = mask[160:180, 96:160]
        try:
            lines = self.detect_lines(roi)
            if not lines:
                return False
            
            lines = np.array(lines).reshape(-1, 2, 2)
            slopes = []
            for line in lines:
                if line[1, 0] != line[0, 0]:
                    slope = (line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0])
                    slopes.append(abs(slope))
            
            return any(s < 0.2 for s in slopes)
        except:
            return False
    
    def turn_where(self, mask):
        """Determine turn direction based on white lines"""
        roi = mask[100:190, :]
        lines = self.detect_lines(roi)
        
        if not lines:
            return 128
        
        try:
            lines = np.array(lines).reshape(-1, 2, 2)
            horizontal_lines = []
            
            for line in lines:
                if line[1, 0] != line[0, 0]:
                    slope = (line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0])
                    if abs(slope) < 0.2:
                        horizontal_lines.append(line)
            
            if horizontal_lines:
                mean_x = np.mean([line[:, :, 0] for line in horizontal_lines])
                return mean_x
        except:
            pass
        
        return 128
    
    def detect_side(self, side_mask):
        """Detect side position"""
        side_pix = np.mean(np.where(side_mask[150:190, :] > 0), axis=1)
        if len(side_pix) > 1:
            return side_pix[1]
        return 128
    
    def red_sign_state(self, red_mask):
        """Detect red stop sign"""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if sorted_contours and cv2.contourArea(sorted_contours[0]) > 50:
                print('Red sign detected!')
                return True
        return False
    
    def stop_the_car(self, car):
        """Safely stop the car"""
        car.setSteering(0)
        while car.getSpeed() > 0:
            car.setSpeed(-100)
            car.getData()
        car.setSpeed(0)
        return True
    
    def turn_the_car(self, car, steering, duration):
        """Execute turn maneuver"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            car.getData()
            car.setSteering(steering)
            car.setSpeed(15)
    
    def go_back(self, car, duration):
        """Reverse the car"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            car.getData()
            car.setSpeed(-15)
        car.setSpeed(0)