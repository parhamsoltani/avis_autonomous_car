import cv2
import numpy as np
import os

try:
    import apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    print("AprilTag library not available. Install with: pip install apriltag")
    APRILTAG_AVAILABLE = False

class AprilTagDetector:
    def __init__(self, show_visualization=False):
        self.show_visualization = show_visualization
        self.detector = None
        
        # Tag ID to instruction mapping
        self.tag_instructions = {
            0: "No Entry",
            1: "Dead End", 
            2: "Proceed Right",
            3: "Proceed Left",
            4: "Proceed Forward",
            5: "Stop"
        }
        
        # Tag ID to action mapping (for navigation)
        self.tag_actions = {
            0: 0,   # No entry - stop
            1: 0,   # Dead end - stop
            2: 1,   # Right turn
            3: -1,  # Left turn
            4: 0,   # Straight
            5: 0    # Stop
        }
        
        # Physical tag size in cm
        self.tag_size_cm = 4.0
        self.focal_length = None
        
        if APRILTAG_AVAILABLE:
            self._init_detector()
    
    def _init_detector(self):
        """Initialize AprilTag detector"""
        try:
            options = apriltag.DetectorOptions(
                families='tag36h11',
                border=1,
                nthreads=4,
                quad_decimate=1.0,
                quad_blur=0.0,
                refine_edges=True,
                refine_decode=True,
                refine_pose=False,  # Don't need pose for 2D detection
                debug=False,
                quad_contours=True
            )
            self.detector = apriltag.Detector(options)
            print("AprilTag detector initialized")
        except Exception as e:
            print(f"Failed to initialize AprilTag detector: {e}")
            self.detector = None
    
    def enhance_image(self, image):
        """Enhance image for better tag detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def calculate_distance(self, tag_corners):
        """Calculate distance to tag based on its size in pixels"""
        if self.focal_length is None:
            # Estimate focal length (would need calibration in real scenario)
            self.focal_length = 500  # Placeholder value
        
        # Calculate tag width in pixels
        width1 = np.linalg.norm(tag_corners[0] - tag_corners[1])
        width2 = np.linalg.norm(tag_corners[2] - tag_corners[3])
        avg_width_pixels = (width1 + width2) / 2
        
        if avg_width_pixels == 0:
            return float('inf')
        
        # Calculate distance
        distance = (self.tag_size_cm * self.focal_length) / avg_width_pixels
        return distance
    
    def detect(self, frame):
        """
        Detect AprilTags in frame
        Returns: List of detected tags with their information
        """
        if not APRILTAG_AVAILABLE or self.detector is None:
            return []
        
        if frame is None or frame.size == 0:
            return []
        
        # Enhance image for better detection
        enhanced = self.enhance_image(frame)
        
        # Detect tags
        try:
            results = self.detector.detect(enhanced)
        except Exception as e:
            print(f"AprilTag detection error: {e}")
            return []
        
        detected_tags = []
        
        for r in results:
            # Filter by detection quality
            if r.decision_margin > 30:
                tag_info = {
                    'id': r.tag_id,
                    'center': r.center,
                    'corners': r.corners,
                    'instruction': self.tag_instructions.get(r.tag_id, "Unknown"),
                    'action': self.tag_actions.get(r.tag_id, 0),
                    'distance': self.calculate_distance(r.corners)
                }
                detected_tags.append(tag_info)
                
                if self.show_visualization:
                    self._visualize_tag(frame, r, tag_info)
        
        if self.show_visualization:
            cv2.imshow("AprilTag Detection", frame)
        
        return detected_tags
    
    def _visualize_tag(self, frame, detection, tag_info):
        """Draw tag visualization on frame"""
        # Draw bounding box
        for i in range(4):
            j = (i + 1) % 4
            pt1 = tuple(map(int, detection.corners[i]))
            pt2 = tuple(map(int, detection.corners[j]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw center
        center = tuple(map(int, detection.center))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Draw tag information
        y_offset = -30
        cv2.putText(frame, f"ID: {tag_info['id']}", 
                   (center[0] + 10, center[1] + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.putText(frame, tag_info['instruction'],
                   (center[0] + 10, center[1] + y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Dist: {tag_info['distance']:.1f}cm",
                   (center[0] + 10, center[1] + y_offset + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def get_closest_tag(self, tags):
        """Get the closest tag from a list of detected tags"""
        if not tags:
            return None
        
        return min(tags, key=lambda t: t['distance'])
    
    def detect_simple(self, frame):
        """
        Simple detection that returns the action of the closest tag
        Returns: action (-1: left, 0: straight/stop, 1: right) or None
        """
        tags = self.detect(frame)
        if tags:
            closest = self.get_closest_tag(tags)
            return closest['action']
        return None

# Fallback detector if AprilTag library is not available
class SimpleAprilTagDetector:
    """Simplified detector using classical CV when apriltag library is not available"""
    
    def __init__(self, show_visualization=False):
        self.show_visualization = show_visualization
        print("Using simplified AprilTag detector (library not available)")
    
    def detect(self, frame):
        """
        Simplified detection using contour detection
        This is a placeholder - real AprilTag detection requires the library
        """
        if frame is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_tags = []
        
        for contour in contours:
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for square-like shapes (4 corners)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > 500 and area < 10000:  # Size filter
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # This is a placeholder - can't determine actual tag ID without library
                        tag_info = {
                            'id': 0,
                            'center': (cx, cy),
                            'corners': approx.reshape(-1, 2),
                            'instruction': "Unknown",
                            'action': 0,
                            'distance': 50.0  # Placeholder
                        }
                        detected_tags.append(tag_info)
                        
                        if self.show_visualization:
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        if self.show_visualization:
            cv2.imshow("Simple Tag Detection", frame)
        
        return detected_tags
    
    def detect_simple(self, frame):
        """Returns None as we can't determine actual tag actions without the library"""
        return None

# Use the appropriate detector based on library availability
if not APRILTAG_AVAILABLE:
    AprilTagDetector = SimpleAprilTagDetector