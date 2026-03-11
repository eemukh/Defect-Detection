import cv2 as cv
import numpy as np
from collections import Counter

# ==========================================
# 1. COLOR LOGIC (Optimized from previous steps)
# ==========================================
def color_name_from_bgr(bgr):
    bgr_arr = np.uint8([[bgr]])
    hsv = cv.cvtColor(bgr_arr, cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Fix 1: Shadow/Black Check
    if v < 50: 
        return "Black"

    # Fix 2: White/Gray Check
    if s < 40:
        if v < 180: return "Gray"
        else: return "White"

    # Fix 3: Expanded Red Range (Captures "Orange-ish" markers)
    if h < 20 or h >= 170: return "Red"
    elif 20 <= h < 35: return "Yellow"
    elif 35 <= h < 85: return "Green"
    elif 85 <= h < 130: return "Blue"
    elif 130 <= h < 160: return "Purple"
    else: return "Unknown"

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
def analyze_captured_frame(img):
    # 1. Determine Background Color
    # We sample the edges of the image to guess the background
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    height, width = hsv_img.shape[:2]
    
    # Create a mask for the "Background"
    # We check if the image is mostly Green or Black/Dark
    # A simple robust way: Assume the most common color in the image is background
    
    # Count colors in the whole image to find background
    # (Downscale for speed)
    small_img = cv.resize(img, (100, 100))
    bg_counts = Counter()
    for row in small_img:
        for pixel in row:
            cname = color_name_from_bgr(pixel)
            bg_counts[cname] += 1
            
    background_color = bg_counts.most_common(1)[0][0]
    # print(f"Detected Background: {background_color}") # Debugging

    # 2. Create the Mask
    # We generate a mask of everything that is NOT the background
    h, s, v = cv.split(hsv_img)
    
    if background_color == "Green":
        # Filter out Green (Hue 35-85)
        # We use inRange to make a "Green Mask" and then invert it
        green_mask = cv.inRange(hsv_img, (35, 50, 50), (85, 255, 255))
        object_mask = cv.bitwise_not(green_mask)
        
    elif background_color == "Black":
        # Filter out Dark pixels (Value < 50)
        # This matches your old logic for the black table
        _, object_mask = cv.threshold(v, 50, 255, cv.THRESH_BINARY)
        
    else:
        # Fallback for other backgrounds (White/Gray tables)
        # Use standard Otsu thresholding on Gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (7,7), 0)
        _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Check if we need to invert (if background is white)
        if np.sum(thresh == 255) > np.sum(thresh == 0):
             object_mask = cv.bitwise_not(thresh)
        else:
             object_mask = thresh

    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    object_mask = cv.morphologyEx(object_mask, cv.MORPH_OPEN, kernel)

    # 3. Find Contours on the Mask
    contours, _ = cv.findContours(object_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, "No object detected"

    # Assume largest contour is the part
    contour = max(contours, key=cv.contourArea)
    
    # Double check: if contour covers > 90% of screen, we probably failed
    screen_area = width * height
    if cv.contourArea(contour) > 0.9 * screen_area:
        return None, None, "Background detection error"

    # 4. Final Color Analysis (Inside the object only)
    final_mask = np.zeros_like(h, dtype=np.uint8)
    cv.drawContours(final_mask, [contour], -1, 255, -1)
    
    object_pixels = img[final_mask == 255]
    
    if len(object_pixels) < 500:
        return None, None, "Object too small"

    color_counts = Counter()
    for pixel in object_pixels:
        cname = color_name_from_bgr(pixel)
        color_counts[cname] += 1

    total = len(object_pixels)
    percentages = {c: (n / total * 100) for c, n in color_counts.items()}
    
    return percentages, contour, "Success"

# ==========================================
# 3. DRAWING/ANNOTATION LOGIC
# ==========================================
def annotate_result(img, contour, percentages):
    out = img.copy()
    
    if not percentages:
        return out

    # Determine Dominant Color
    dominant_color = max(percentages, key=percentages.get)
    
    # Logic for Defects
    check_list = percentages.copy()
    if dominant_color in check_list:
        check_list.pop(dominant_color)

    defect_found = False
    defect_label = ""
    DEFECT_THRESHOLD = 5.0  # Kept at 5% based on your previous test

    # Check for defects
    for color, pct in check_list.items():
        if pct > DEFECT_THRESHOLD:
            defect_found = True
            defect_label = f" SWIRL DEFECT FOUND)"
            break 

    # Draw Contours and Labels
    if defect_found: 
        # RED for FAIL
        color = (0, 0, 255)
        text = defect_label
    else:
        # GREEN for PASS
        color = (0, 255, 0)
        text = f"OK"

    cv.drawContours(out, [contour], -1, color, 3)
    
    # Text positioning
    x, y, w, h = cv.boundingRect(contour)
    # Ensure text doesn't go off-screen
    text_y = y - 10 if y - 10 > 20 else y + h + 20
    cv.putText(out, text, (x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return out


# ==========================================
# 4. MAIN WORKFLOW LOOP
# ==========================================
def main():
    cv.namedWindow("Swirl Detector", cv.WINDOW_NORMAL)
    #cv.resizeWindow("Swirl Detector", 500, 700)

    cap = cv.VideoCapture(0)
    
    # Set resolution (optional, helps with FPS. Adjust as needed)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- SWIRL DETECTOR RUNNING ---")
    print("1. Align object in the camera.")
    print("2. Press 's' to SCREENSHOT and SCAN.")
    print("3. Press 'q' to QUIT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show live feed with instructions
        display_frame = frame.copy()
        cv.putText(display_frame, "Press 's' to SCAN", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv.imshow("Swirl Detector", display_frame)

        key = cv.waitKey(1) & 0xFF

        # --- TRIGGER: User presses 's' ---
        if key == ord('s'):
            print("Capturing...")
            
            # 1. Run Analysis on the current frame
            percentages, contour, status = analyze_captured_frame(frame)
            
            # 2. Annotate the Result
            if percentages:
                result_image = annotate_result(frame, contour, percentages)
                
                # Print results to console
                print("\n--- SCAN RESULTS ---")
                print(f"Full Breakdown: {percentages}")
            else:
                # Handle empty/error cases
                result_image = frame.copy()
                cv.putText(result_image, f"Error: {status}", (50, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"\nScan Failed: {status}")

            # 3. Pause and Show Result
            # We add instruction to continue
            cv.putText(result_image, "Press any key to continue...", (10, result_image.shape[0] - 20), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv.imshow("Swirl Detector", result_image)
            cv.waitKey(0) # Waits indefinitely for a key press
            print("Returning to live feed...\n")

        # --- EXIT: User presses 'q' ---
        elif key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()