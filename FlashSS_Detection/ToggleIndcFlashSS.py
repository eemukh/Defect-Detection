import cv2 as cv
import numpy as np
import sys

# --- Configuration ---
# 0 = dogbone, 1 = keychain
current_option = 0 
mode_labels = {0: "Dogbone", 1: "Keychain"}

class dogbone:
    normal_min = 200
    normal_max = 260

class keychain:
    normal_min = 135
    normal_max = 160

# --- Detection Function ---
def process_part(image, option):
    print(f"Processing image for: {mode_labels[option]}...")
    
    # 1. Establish Gaussian Pyramid
    gaussian = []  
    gaussian_layer = image.copy()

    for i in range(4):
        gaussian_layer = cv.pyrDown(gaussian_layer)
        gaussian.append(gaussian_layer)

    # 2. Establish Laplacian Pyramid
    Laplacian = [gaussian[-1]] 

    for i in range(3, 0, -1):
        size = (gaussian[i - 1].shape[1], gaussian[i - 1].shape[0])
        gaussian_expanded = cv.pyrUp(gaussian[i], dstsize=size)
        laplacian_layer = cv.subtract(gaussian[i-1], gaussian_expanded)
        Laplacian.append(laplacian_layer)

    # 3. Edge Detection
    edged_images = []
    for i in range(4):
        laplacian_layer = Laplacian[i]
        edges = cv.Canny(laplacian_layer, 200, 200)
        edged_images.append(edges)

    # 4. Sharpening
    dogbone_kernel = np.array([[0.50, 0.50, 0.50], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    keychain_kernel = np.array([[0.67, 0.74, 0.62], [0.65, 0.68, 0.66], [0.74, 0.62, 0.67]])

    if option == 0: 
        kernel = dogbone_kernel
    else:
        kernel = keychain_kernel

    sharpened_image = cv.filter2D(edged_images[0], -1, kernel)

    # 5. Contour and Analysis
    Contour, _ = cv.findContours(sharpened_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(Contour) > 0:
        test_area = cv.contourArea(Contour[0])
        print(f"Contour Area: {test_area}")

        if option == 0: # Dogbone Logic
            if test_area < dogbone.normal_min * 0.98:
                status = "SHORT SHOT"
                color = (0, 0, 255) 
            elif test_area > dogbone.normal_max * 1.04:
                status = "FLASH"
                color = (0, 0, 255) 
            else:
                status = "NORMAL"
                color = (0, 255, 0) 
        else: # Keychain Logic
            if test_area < keychain.normal_min:
                status = "SHORT SHOT"
                color = (0, 0, 255)
            elif test_area > keychain.normal_max * 1.02:
                status = "FLASH"
                color = (0, 0, 255)
            else:
                status = "NORMAL"
                color = (0, 255, 0)
        
        print(f"Result: {status}\n")
    else:
        print("No contours found.")
        status = "No Contour"
        color = (255, 0, 0)

    # 6. Visualization
    resized_result = cv.resize(sharpened_image, (500, 500))
    if len(resized_result.shape) == 2:
        resized_result_color = cv.cvtColor(resized_result, cv.COLOR_GRAY2BGR)
    else:
        resized_result_color = resized_result

    # Display status on the result window
    cv.putText(resized_result_color, f"{mode_labels[option]}: {status}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.imshow('Last Scan Result', resized_result_color)


# --- Main Loop ---

video_capture = cv.VideoCapture(0)
cv.namedWindow("Webcam Feed")

if not video_capture.isOpened():
    print("Error: Could not open video source.")
else:
    print("System Ready.")
    print("Controls: 's' = Scan, 't' = Toggle Mode, 'q' = Quit")
    
    while True:
        ret, frame = video_capture.read()

        if not ret or frame is None:
            print("Error: Can't receive frame. Exiting ...")
            break

        # --- UI Overlay ---
        # Draw current mode on the live feed so you don't have to check the terminal
        mode_text = f"Mode: {mode_labels[current_option]}"
        cv.putText(frame, mode_text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 0), 2, cv.LINE_AA)

        cv.imshow("Webcam Feed", frame)

        keypress = cv.waitKey(1) & 0xFF

        if keypress == ord('s'):
            # Pass the current_option state to the processor
            process_part(frame, current_option)
            
        elif keypress == ord('t'):
            # Toggle between 0 and 1
            current_option = 1 - current_option
            print(f"Switched to {mode_labels[current_option]}")
            
        elif keypress == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()