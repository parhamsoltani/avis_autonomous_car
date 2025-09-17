import cv2 as cv
import numpy as np
import sys
import os
from time import time
import onnxruntime as ort
from math import hypot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avisengine import avisengine
from urban_config import *

# ------------------ Settings ------------------
classes = ['Proceed Forward', 'Proceed Left', 'Proceed Right', 'Stop', 'traffic light']
obstacle_classes = ["obstacle", "stop line"]  # Model classes but we'll filter out stop lines
NUM_CLASSES = len(classes)
CONF_THRES = 0.5
NMS_THRES = 0.6
W, H = 512, 512
CAR_CENTER = (260, 400)
top_left = (160, 230)
top_right = (352, 230)
bottom_right = (W - 30, H - 120)
bottom_left = (60, H - 120)
CONTOUR_MIN_SIZE = 250
APPROX_MAX_SIZE = 16
LOWER = np.array([0, 11, 148])
UPPER = np.array([41, 19, 255])

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
    """Calculate lane center based on white line with memory of previous frame."""
    warped_frame = warp_frame(frame)
    white_mask = create_mask(warped_frame, LOWER, UPPER)
    white_img, white_center_x = find_line(warped_frame, white_mask)

    if white_center_x is not None:
        target_x = white_center_x
    else:
        target_x = prev_avg if prev_avg is not None else CAR_CENTER[0]

    if debug:
        cv.imshow("warped", warped_frame)
        cv.imshow("white mask", white_mask)

    return white_img, target_x

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

def iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    inter = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))
    area1, area2 = (x2-x1)*(y2-y1), (X2-X1)*(Y2-Y1)
    return inter / (area1 + area2 - inter + 1e-6)

class UrbanMode:
    def __init__(self, ip='127.0.0.3', port=25004):
        self.car = avisengine.Car()
        self.car.connect(ip, port)

        # Load sign detection model
        sign_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       'models', 'sign_detection.onnx')
        try:
            self.sign_model = ort.InferenceSession(sign_model_path)
            print("Loaded sign detection model")
        except:
            print("Error: sign_detection.onnx not found")
            self.sign_model = None
            exit()

        # Load obstacle detection model
        obstacle_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           'models', 'obstacle_segmentation.onnx')
        try:
            self.obstacle_model = ort.InferenceSession(obstacle_model_path)
            print("Loaded obstacle detection model")
            self.has_obstacle_detection = True
        except:
            print("Warning: obstacle_segmentation.onnx not found - obstacle detection disabled")
            self.obstacle_model = None
            self.has_obstacle_detection = False

        # Initialize variables - EXACT from working version
        self.traffic_light_detected = False
        self.start_time = 0
        self.center_line_x = 0
        self.steering = 0

        # Direction flags - EXACT from working version
        self.Proceed_Forward = False
        self.Proceed_Left = False
        self.Proceed_Right = False
        self.Stop = False

        # Smoother obstacle detection variables
        self.obs = False
        self.frame_count = 0
        self.result = []

    def detect_obstacles(self, frame):
        """Detect obstacles only - ignore stop lines"""
        if not self.has_obstacle_detection or self.obstacle_model is None:
            return []

        # Model Inference - run every 4 frames
        run_model = (self.frame_count % 4 == 0)
        if run_model:
            self.result = []
            inp = cv.resize(frame, (640, 640)).astype(np.float32)/255.0
            inp = np.transpose(inp, (2,0,1))[None]

            try:
                out0, out1 = self.obstacle_model.run(None, {"images": inp})
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

                    # ONLY ADD OBSTACLES - IGNORE STOP LINES COMPLETELY
                    if obstacle_classes[cid] == "obstacle":
                        self.result.append([x1, y1, x2, y2, obstacle_classes[cid], prob, mask_img])

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

        return self.result

    def run(self):
        """Main loop - proper sign action + smooth obstacle avoidance"""
        print("Starting Urban Mode...")

        while True:
            self.car.getData()
            frame = self.car.getImage()
            if frame is None:
                continue

            # Lane Detection
            lane_img, self.center_line_x = calc_steering(frame, prev_avg=self.center_line_x)

            # Sign Detection - EXACT from working version
            orig_h, orig_w = frame.shape[:2]
            inp = cv.resize(frame, (384, 384)).astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None]

            outputs = self.sign_model.run(None, {"images": inp})
            out0 = outputs[0]
            out0 = out0.squeeze(0)
            out0 = out0.transpose(1, 0)

            boxes = out0[:, :4]
            scores = out0[:, 4:]

            sign_detections = []
            for i, box in enumerate(boxes):
                probs = scores[i]
                cid = probs.argmax()
                conf = probs[cid]
                if conf < CONF_THRES:
                    continue

                xc, yc, w, h = box
                x1 = (xc - w/2) / 384 * orig_w
                y1 = (yc - h/2) / 384 * orig_h
                x2 = (xc + w/2) / 384 * orig_w
                y2 = (yc + h/2) / 384 * orig_h

                sign_detections.append([x1, y1, x2, y2, classes[cid], conf])

            # NMS for signs
            sign_detections.sort(key=lambda x: x[5], reverse=True)
            final_signs = []
            while sign_detections:
                best = sign_detections.pop(0)
                final_signs.append(best)
                sign_detections = [d for d in sign_detections if iou(d[:4], best[:4]) < NMS_THRES]

            # Obstacle Detection
            obstacle_detections = self.detect_obstacles(frame)
            self.frame_count += 1

            # Control car - EXACT logic from working sign detection
            avg = self.center_line_x // 2
            traffic_light = False

            # Check for traffic signs - EXACT from working version
            if final_signs:
                l_2 = final_signs[0][4]
                s_2 = final_signs[0][5]

                if l_2 == 'traffic light' and s_2 > 0.8 and final_signs[0][2] > 190 and final_signs[0][2] < 350:
                    traffic_light = True
                    print(l_2, s_2, final_signs[0][2])
                elif l_2 == 'Proceed Forward' and s_2 > 0.8:
                    self.Proceed_Forward = True
                    print(l_2, s_2)
                elif l_2 == 'Proceed Left' and s_2 > 0.8:
                    self.Proceed_Left = True
                    print(l_2, s_2)
                elif l_2 == 'Proceed Right' and s_2 > 0.8:
                    self.Proceed_Right = True
                    print(l_2, s_2)
                elif l_2 == 'Stop' and s_2 > 0.8:
                    self.Stop = True
                    print(l_2, s_2)

            # Obstacle detection logic - smoother than race mode
            if obstacle_detections:
                l = obstacle_detections[0][4]
                s = obstacle_detections[0][5]

                if l == 'obstacle' and obstacle_detections[0][2] > 200:
                    if not self.obs:
                        self.obs = True
                        self.obs_start_time = time()
                        print(f"Obstacle detected: {l} {s:.2f}")
            else:
                # Reset obstacle state when no detections
                if self.obs and hasattr(self, 'obs_start_time') and (time() - self.obs_start_time >= 3):
                    self.obs = False
                    delattr(self, 'obs_start_time')

            # Traffic light handling - EXACT from working version
            if traffic_light:
                self.car.setSpeed(0)
                if not self.traffic_light_detected:
                    self.start_time = time()
                    self.traffic_light_detected = True

            # Traffic light sequence handling - EXACT from working version
            if self.traffic_light_detected:
                if time() - self.start_time > 3:  # Wait 3 seconds then follow sign
                    if self.Proceed_Right:
                        self.steering = translate(self.center_line_x, 0, 50, 0, 90)
                        self.car.setSteering(int(self.steering))
                        self.car.setSpeed(10)
                    elif self.Proceed_Left:
                        self.steering = translate(self.center_line_x, 0, 50, -90, 0)
                        self.car.setSteering(int(self.steering))
                        self.car.setSpeed(10)
                    elif self.Proceed_Forward:
                        self.steering = translate(avg, 90, 170, -15, 15)
                        self.car.setSteering(int(self.steering))
                        self.car.setSpeed(10)
                    elif self.Stop:
                        self.car.setSpeed(0)

                # Reset after 20 seconds
                if time() - self.start_time > 20:
                    self.traffic_light_detected = False
                    self.Proceed_Forward = False
                    self.Proceed_Left = False
                    self.Proceed_Right = False
                    self.Stop = False

            # Obstacle avoidance - smoother steering for urban environment
            elif self.obs and hasattr(self, 'obs_start_time') and (time() - self.obs_start_time < 3):
                # Smoother obstacle avoidance steering - gentler left turn
                self.steering = translate(self.center_line_x, 200, 400, -40, 40)
                self.car.setSteering(int(self.steering))
                self.car.setSpeed(8)  # Slightly slower but not too slow

            # Normal driving
            else:
                if not traffic_light:
                    self.steering = translate(avg, 90, 170, -15, 15)
                    self.car.setSteering(int(self.steering))
                    self.car.setSpeed(10)

            # Draw results
            show_frame = frame.copy()

            # Draw sign detections (green boxes)
            for x1, y1, x2, y2, label, conf in final_signs:
                cv.rectangle(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(show_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-5),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw obstacle detections (red boxes)
            for x1, y1, x2, y2, label, prob, _ in obstacle_detections:
                cv.rectangle(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                cv.putText(show_frame, f"{label} {prob:.2f}", (int(x1), int(y1)-5),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Draw lane center point
            cv.circle(show_frame, (self.center_line_x, 300), 5, (0,255,0), -1)

            # Status information
            cv.putText(show_frame, f"Steering: {int(self.steering)}", (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv.putText(show_frame, f"Speed: {self.car.getSpeed()}", (10, 60),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            # Show current state
            state_text = ""
            if self.traffic_light_detected:
                state_text = "TRAFFIC LIGHT"
                color = (0, 255, 0)
            elif self.obs and hasattr(self, 'obs_start_time'):
                state_text = "AVOIDING OBSTACLE"
                color = (0, 0, 255)
            else:
                state_text = "NORMAL"
                color = (255, 255, 255)

            cv.putText(show_frame, f"State: {state_text}", (10, 90),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show next action
            if self.Proceed_Forward or self.Proceed_Left or self.Proceed_Right or self.Stop:
                next_action = ""
                if self.Proceed_Forward:
                    next_action = "FORWARD"
                elif self.Proceed_Left:
                    next_action = "LEFT"
                elif self.Proceed_Right:
                    next_action = "RIGHT"
                elif self.Stop:
                    next_action = "STOP"

                cv.putText(show_frame, f"Next: {next_action}", (10, 120),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv.imshow("Urban Mode", show_frame)
            cv.imshow("Lane Detection", lane_img)

            if cv.waitKey(1) & 0xFF == 27:
                break

        self.car.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    urban = UrbanMode()
    urban.run()