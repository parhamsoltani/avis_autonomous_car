import numpy as np

def translate(value, leftMin, leftMax, rightMin, rightMax):
    """Map value from one range to another"""
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def calculate_curve_speed(steering_angle, base_speed):
    """Calculate appropriate speed based on steering angle"""
    # More steering = slower speed
    steering_factor = abs(steering_angle) / 30.0  # Normalize to 0-1
    speed_reduction = steering_factor * 40  # Max 40% speed reduction
    return int(base_speed * (1 - speed_reduction * 0.01))