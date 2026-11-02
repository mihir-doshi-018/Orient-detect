def calculate_correction(current_angle, target_angle=0.0):
    """
    Calculates the shortest turning path correction restricted between 0 to 45 degrees 
    for 90-degree symmetry, along with the rotation direction (Clockwise or Counter-Clockwise).
    """
    # 1. Find the nearest 90-degree target quadrant
    nearest_target = round(current_angle / 90.0) * 90.0
    
    # 2. Calculate signed delta to see which way is closer
    delta = nearest_target - current_angle
    
    # 3. Normalize delta to the range [-45, 45] for 90-degree symmetry
    while delta > 45.0:
        delta -= 90.0
    while delta < -45.0:
        delta += 90.0

    # 4. Determine correction magnitude and direction
    correction = abs(delta)
    
    # Threshold for alignment (e.g., within 0.5 or 1.0 degree)
    if correction < 0.5:
        direction = "None"
    elif delta > 0:
        direction = "Clockwise"
    else:
        direction = "Counter-Clockwise"
        
    return round(float(correction), 2), direction