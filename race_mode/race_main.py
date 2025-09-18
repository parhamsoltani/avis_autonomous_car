import cv2 as cv
import numpy as np
import sys
import os
from time import time
import onnxruntime as ort
from math import hypot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avisengine import avisengine
from race_config import *


W, H = 512, 512
CAR_CENTER = (260, 400)
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (W - 30, H - 120)
bottom_left = (60, H - 120)
CONTOUR_MIN_SIZE = 250
APPROX_MAX_SIZE = 16
LOWER_YELLOW = np.array([22, 102, 122])
UPPER_YELLOW = np.array([30, 255, 255])
classes = ["obstacle", "stop line"]

def warp_frame(frame):
    """Convert main view to bird-eye view"""
    src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_points = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    matrix = cv.getPerspectiveTransform(src_points, dst_points)
    return cv.warpPerspective(frame, matrix, (W, H))

def create_mask(frame, low, up):
    """Create a color mask (HSV)"""
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(img_hsv, low, up)

def find_line(frame, mask):
    """Find the line center from contours"""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    line_center_x = None
    line_img = np.zeros_like(frame)

    if len(contours) > 0:
        candidates = []
        for c in contours:
            if cv.contourArea(c) > CONTOUR_MIN_SIZE:
                epsilon = 0.01 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                if len(approx) < APPROX_MAX_SIZE:
                    M = cv.moments(c)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        dist = hypot(cx - CAR_CENTER[0], cy - CAR_CENTER[1])
                        candidates.append((c, (cx, cy), dist))
        if candidates:
            candidates.sort(key=lambda x: x[2])
            best = candidates[0]
            line_center_x = best[1][0]
            cv.drawContours(line_img, [best[0]], -1, (0, 255, 0), -1)

    return line_img, line_center_x

def translate(value, leftMin, leftMax, rightMin, rightMax):
    """Map a pixel range to a steering angle range"""
    if leftMax == leftMin:
        return rightMin
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def calc_steering(frame, prev_avg=None, debug=False):
    """Calculate lane center based on yellow line with memory of previous frame."""
    warped_frame = warp_frame(frame)
    yellow_mask = create_mask(warped_frame, LOWER_YELLOW, UPPER_YELLOW)
    yellow_img, yellow_center_x = find_line(warped_frame, yellow_mask)

    if yellow_center_x is not None:
        target_x = yellow_center_x
    else:
        target_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    if debug:
        cv.imshow("warped", warped_frame)
        cv.imshow("yellow mask", yellow_mask)

    return yellow_img, target_x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_mask(mask_vec, box, orig_w, orig_h):
    x1,y1,x2,y2 = box
    mask = sigmoid(mask_vec.reshape(160,160))
    mask = (mask>0.5).astype(np.uint8)*255
    mx1,my1,mx2,my2 = map(int,[x1/orig_w*160,y1/orig_h*160,x2/orig_w*160,y2/orig_h*160])
    mask = mask[my1:my2,mx1:mx2]
    mask = cv.resize(mask,(int(x2-x1),int(y2-y1)), interpolation=cv.INTER_NEAREST)
    return mask

def iou(b1,b2):
    x1,y1,x2,y2 = b1
    X1,Y1,X2,Y2 = b2
    inter = max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1))
    area1,area2 = (x2-x1)*(y2-y1),(X2-X1)*(Y2-Y1)
    return inter/(area1+area2-inter+1e-6)

class RaceMode:
    def __init__(self, ip='127.0.0.3', port=25004):
        self.car = avisengine.Car()
        self.car.connect(ip, port)
        self.car.setSensorAngle(20)

        # Load ONNX model
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'models', 'obstacle_segmentation.onnx')
        try:
            self.model = ort.InferenceSession(model_path)
            self.use_yolo = True
            print("Loaded ONNX model for obstacle detection")
        except:
            self.model = None
            self.use_yolo = False
            print("Using sensor-based obstacle detection")

        # Initialize variables
        self.center_line_x = 0
        self.obs = False
        self.start_time = 0
        self.frame_count = 0
        self.result = []
        self.steering = 0

    def run(self):
        """Main loop - exact implementation"""
        print("Starting Race Mode...")

        while True:
            self.car.getData()
            self.car.setSpeed(speed)  # speed from config

            if self.use_yolo:
                self.run_with_yolo()
            else:
                self.run_with_sensors()

            if cv.waitKey(1) & 0xFF == 27:  # ESC
                break

        self.car.stop()
        cv.destroyAllWindows()

    def run_with_sensors(self):
        """Sensor-only mode - exact from reference obstacle_detection_with_sensor.py"""
        sensors = self.car.getSensors()
        frame = self.car.getImage()

        if frame is None:
            return

        # Calculate lane center
        lane_img, self.center_line_x = calc_steering(frame, prev_avg=self.center_line_x, debug=False)

        # Speed control based on steering
        if abs(self.car.steering_value) < 10:
            self.car.setSpeed(speed)
        else:
            self.car.setSpeed(7)

        # Obstacle detection logic
        if sensors[2] > 1100:
            # No obstacle
            self.steering = translate(self.center_line_x, 0, 171, -30, 30)
            self.car.setSteering(int(self.steering))
        else:
            if self.obs == False:
                self.obs = True
                self.start_time = time()

        if self.obs and (time() - self.start_time < 10):
            self.steering = translate(self.center_line_x, 340, 400, -80, 80)
            self.car.setSteering(int(self.steering))
            if self.car.getSpeed() > 10:
                self.car.setSpeed(7)
        else:
            self.obs = False

        # Display
        show_frame = frame.copy()
        cv.circle(show_frame, (self.center_line_x, 300), 5, (0, 255, 0), cv.FILLED)
        cv.putText(show_frame, f"Steering: {int(self.steering)}",
                  (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv.putText(show_frame, f"Speed: {self.car.getSpeed()}",
                  (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv.imshow("frame", show_frame)
        cv.imshow("lane detection", lane_img)

    def run_with_yolo(self):
        """YOLO mode - exact from reference obstacle_detection_with_yolo.py"""
        frame = self.car.getImage()
        if frame is None:
            return

        # Lane Detection
        lane_img, self.center_line_x = calc_steering(frame, prev_avg=self.center_line_x)

        # Model Inference - run every 4 frames
        run_model = (self.frame_count % 4 == 0)
        if run_model:
            self.result = []
            inp = cv.resize(frame, (640, 640)).astype(np.float32)/255.0
            inp = np.transpose(inp, (2,0,1))[None]

            try:
                out0, out1 = self.model.run(None, {"images": inp})
                out0, out1 = out0[0].T, out1[0].reshape(32,-1)
                boxes, mask_coef = out0[:,:6], out0[:,6:]
                masks = mask_coef @ out1
                orig_h, orig_w = frame.shape[:2]

                for row, maskv in zip(boxes, masks):
                    prob = row[4:6].max()
                    if prob < 0.4: continue
                    xc, yc, w, h = row[:4]
                    cid = row[4:6].argmax()
                    x1, y1 = (xc-w/2)/640*orig_w, (yc-h/2)/640*orig_h
                    x2, y2 = (xc+w/2)/640*orig_w, (yc+h/2)/640*orig_h
                    mask_img = get_mask(maskv, (x1,y1,x2,y2), orig_w, orig_h)
                    self.result.append([x1, y1, x2, y2, classes[cid], prob, mask_img])

                # NMS
                self.result.sort(key=lambda x: x[5], reverse=True)
                final_result = []
                while self.result:
                    best = self.result.pop(0)
                    final_result.append(best)
                    self.result = [o for o in self.result if iou(o[:4], best[:4]) < 0.7]
                self.result = final_result
            except:
                pass

        self.frame_count += 1

        # Speed control
        if abs(self.car.steering_value) < 10:
            self.car.setSpeed(speed)
        else:
            self.car.setSpeed(7)

        # Steering control\]
        if self.result:
            l = self.result[0][4]
            s = self.result[0][5]

            if l == 'obstacle' and self.result[0][2] > 200:
                self.obs = True
                print(l, s)
            else:
                self.obs = False

        # CRITICAL FIX: The timing logic was wrong
        if self.obs:
            if not hasattr(self, 'obs_start_time'):
                self.obs_start_time = time()
        else:
            self.steering = translate(self.center_line_x, 0, 171, -30, 30)
            self.car.setSteering(int(self.steering))
            if hasattr(self, 'obs_start_time'):
                delattr(self, 'obs_start_time')

        # Apply obstacle avoidance steering
        if self.obs and hasattr(self, 'obs_start_time') and (time() - self.obs_start_time < 2):
            self.steering = translate(self.center_line_x, 380, 450, -90, 90)
            self.car.setSteering(int(self.steering))
            if self.car.getSpeed() > 10:
                self.car.setSpeed(7)
        elif self.obs and hasattr(self, 'obs_start_time'):
            self.obs = False
            delattr(self, 'obs_start_time')

        # Display
        show_frame = frame.copy()

        # Draw bounding boxes
        for x1, y1, x2, y2, label, prob, _ in self.result:
            cv.rectangle(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv.putText(show_frame, f"{label} {prob:.2f}", (int(x1), int(y1)-5),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv.circle(show_frame, (self.center_line_x, 300), 5, (0,255,0), -1)
        cv.putText(show_frame, f"Steering: {int(self.steering)}",
                  (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv.putText(show_frame, f"Speed: {self.car.getSpeed()}",
                  (10,45), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

        cv.imshow("Lane Detection", lane_img)
        cv.imshow("Object Detection", show_frame)

if __name__ == "__main__":
    race = RaceMode()
    race.run()
