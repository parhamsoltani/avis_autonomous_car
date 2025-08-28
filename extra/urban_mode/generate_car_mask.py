"""
Generate car mask to filter out car body from detection
Run this once to create the car_mask.npy file
"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avisengine import avisengine

def generate_car_mask():
    print("Generating car mask...")
    print("Position the car in the simulator and press 's' to save mask")
    
    car = avisengine.Car()
    car.connect("127.0.0.1", 25001)
    
    # Let the connection stabilize
    for _ in range(10):
        car.setSpeed(0)
        car.setSteering(0)
        car.getData()
    
    mask = None
    
    while True:
        car.getData()
        frame = car.getImage()
        
        if frame is not None:
            # Create mask for car body (lower center portion)
            mask = np.zeros((256, 256), dtype=np.uint8)
            
            # Mark the car body area - adjust these values based on your car
            mask[200:256, 80:176] = 1  # Bottom center area where car appears
            
            # Visualize
            display = frame.copy()
            display[mask == 1] = [0, 0, 255]  # Show masked area in red
            
            cv2.imshow("Car Mask Preview", display)
            cv2.imshow("Mask", mask * 255)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                np.save('car_mask.npy', mask)
                print("Car mask saved to car_mask.npy")
                break
            elif key == 27:  # ESC
                print("Cancelled")
                break
    
    car.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_car_mask()