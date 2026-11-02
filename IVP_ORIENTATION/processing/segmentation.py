import cv2
import numpy as np

def segment_blocks(frame, roi_pts, min_area=80, max_area=8000):
    """
    Blacks out everything outside the conveyor ROI polygon first, 
    then precisely detects building block colors (Yellow, Blue, Red, Green).
    """
    height, width, _ = frame.shape
    pts = np.array(roi_pts, dtype=np.int32)
    
    # 1. Create a strict binary ROI mask for the conveyor zone
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 255)
    
    # 2. Crop/Blackout everything outside the polygon *before* any color processing
    # This completely eliminates background machinery and reflections
    frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # 3. Convert the isolated conveyor region to HSV
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    
    # 4. Precise HSV color ranges for the physical blocks
    # Yellow blocks
    mask_yellow = cv2.inRange(hsv, np.array([20, 90, 90]), np.array([35, 255, 255]))
    
    # Blue blocks
    mask_blue = cv2.inRange(hsv, np.array([100, 90, 90]), np.array([130, 255, 255]))
    
    # Green blocks
    mask_green = cv2.inRange(hsv, np.array([36, 60, 60]), np.array([85, 255, 255]))
    
    # Red blocks (Red wraps around 0 and 179 in OpenCV HSV)
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combine all valid block color masks together
    mask = cv2.bitwise_or(mask_yellow, mask_blue)
    mask = cv2.bitwise_or(mask, mask_green)
    mask = cv2.bitwise_or(mask, mask_red)
    
    # 5. Clean up noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. Find contours strictly from the isolated block color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.2 <= aspect_ratio <= 5.0:
                valid_contours.append(c)
                
    return valid_contours, mask