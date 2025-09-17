import cv2 as cv
import numpy as np
from math import hypot

class LaneDetector:
    def __init__(self):
        self.W, self.H = 512, 512
        self.CAR_CENTER = (260, 400)

        # Perspective transform points
        self.top_left = (160, 230)
        self.top_right = (352, 230)
        self.bottom_right = (self.W - 30, self.H - 120)
        self.bottom_left = (60, self.H - 120)

        # Contour parameters
        self.CONTOUR_MIN_SIZE = 250
        self.APPROX_MAX_SIZE = 16

        # Color ranges
        self.LOWER_YELLOW = np.array([22, 102, 122])
        self.UPPER_YELLOW = np.array([30, 255, 255])

        self.LOWER_WHITE = np.array([0, 11, 148])
        self.UPPER_WHITE = np.array([41, 19, 255])

    def warp_frame(self, frame):
        """Exact implementation from"""
        src_points = np.float32([self.top_left, self.top_right,
                                self.bottom_right, self.bottom_left])
        dst_points = np.float32([[0, 0], [self.W, 0],
                                [self.W, self.H], [0, self.H]])
        matrix = cv.getPerspectiveTransform(src_points, dst_points)
        return cv.warpPerspective(frame, matrix, (self.W, self.H))

    def create_mask(self, frame, low, up):
        """Exact implementation from"""
        img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        return cv.inRange(img_hsv, low, up)

    def find_line(self, frame, mask):
        """Exact implementation from"""
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        line_center_x = None
        line_img = np.zeros_like(frame)

        if len(contours) > 0:
            candidates = []
            for c in contours:
                if cv.contourArea(c) > self.CONTOUR_MIN_SIZE:
                    epsilon = 0.01 * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    if len(approx) < self.APPROX_MAX_SIZE:
                        M = cv.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            dist = hypot(cx - self.CAR_CENTER[0], cy - self.CAR_CENTER[1])
                            candidates.append((c, (cx, cy), dist))

            if candidates:
                candidates.sort(key=lambda x: x[2])
                best = candidates[0]
                line_center_x = best[1][0]
                cv.drawContours(line_img, [best[0]], -1, (0, 255, 0), -1)

        return line_img, line_center_x

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        """Exact implementation from """
        if leftMax == leftMin:
            return rightMin
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = float(value - leftMin) / float(leftSpan)
        return rightMin + (valueScaled * rightSpan)

    def calc_steering(self, frame, prev_avg=None, color='yellow'):
        """Calculate steering based on lane detection"""
        warped_frame = self.warp_frame(frame)

        if color == 'yellow':
            mask = self.create_mask(warped_frame, self.LOWER_YELLOW, self.UPPER_YELLOW)
        else:  # white
            mask = self.create_mask(warped_frame, self.LOWER_WHITE, self.UPPER_WHITE)

        line_img, line_center_x = self.find_line(warped_frame, mask)

        if line_center_x is not None:
            target_x = line_center_x
        else:
            target_x = prev_avg if prev_avg is not None else self.CAR_CENTER[0]

        return line_img, target_x, warped_frame