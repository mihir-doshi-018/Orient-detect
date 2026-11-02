import cv2
import numpy as np
import math

def calculate_orientation(contour, frame_gray):
    """
    Uses PCA + asymmetry to estimate orientation, then snaps 
    to the nearest 90-degree discrete L-block state (0, 90, 180, 270).
    """
    pts = contour.reshape(contour.shape[0], 2).astype(np.float64)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    
    dx, dy = eigenvectors[0, 0], eigenvectors[0, 1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 180
    
    final_angle = angle_deg
    x, y, w, h = cv2.boundingRect(contour)
    
    # Resolve 180-degree ambiguity using image moments
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        box_cx, box_cy = x + w / 2.0, y + h / 2.0
        vec_x, vec_y = cx - box_cx, cy - box_cy
        
        if (vec_x * dx + vec_y * dy) < 0:
            final_angle = (angle_deg + 180) % 360

    # Return the real calculated orientation angle (0 - 360) and centroid
    real_angle = round(float(final_angle), 2)
    return real_angle, (int(mean[0, 0]), int(mean[0, 1]))