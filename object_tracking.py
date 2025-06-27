import cv2

# Template matching method
# - TM_CCOEFF_NORMED is good for brightness/contrast changes
MATCHING_METHOD = cv2.TM_CCOEFF_NORMED

# Minimum threshold for considering a template match valid (0.0 to 1.0)
MIN_THRESHOLD = 0.8

# Display settings for text overlay
FONT_SIZE = 0.5
FONT_COLOR = (0, 255, 0)  # Green color for text (BGR format)
FONT_THICKNESS = 2

def get_frame(camera):
    """
    Capture a single frame from the camera.

    Args:
        camera: OpenCV VideoCapture object

    Returns:
        Frame flipped horizontally for mirror effect

    Raises:
        Exception: If frame cannot be read from camera
    """
    is_success, frame = camera.read()
    if not is_success:
        raise Exception("Failed to read frame from camera")

    # Flip horizontally to create mirror effect (more intuitive for user)
    return cv2.flip(frame, 1)

def get_roi(frame):
    """
    Allow user to select Region of Interest (ROI) from the frame.

    Args:
        frame: Image frame to select ROI from

    Returns:
        Tuple of (x, y, width, height) coordinates of selected ROI
    """
    # Display interactive window for ROI selection
    # - fromCenter=False: Selection starts from corner
    # - showCrosshair=True: Shows crosshair for precise selection
    roi = cv2.selectROI(
        "Select ROI", frame, fromCenter=False, showCrosshair=True
    )

    # Clean up by closing the selection window
    cv2.destroyWindow("Select ROI")

    # Small delay to ensure window is properly closed
    cv2.waitKey(1)

    return roi

def get_template(roi, frame):
    """
    Extract template image from the selected ROI.

    Args:
        roi: (x, y, width, height) tuple defining region
        frame: Source image to extract from

    Returns:
        The template image (ROI portion of frame)
    """
    roi_x, roi_y, roi_width, roi_height = roi

    # Array slicing to extract region: frame[y:y+h, x:x+w]
    template = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    return template

def get_result(result):
    """
    Analyze template matching result to find best match location.

    Args:
        result: Template matching result matrix from cv2.matchTemplate()

    Returns:
        tuple: (max_val, max_loc) - best match value and location
    """
    # Find min/max values and their locations in the result matrix
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if match confidence meets minimum threshold
    if max_val < MIN_THRESHOLD:
        return 0.0, (0, 0)  # No good match found

    return max_val, max_loc

def get_tl_and_br_coordinates(max_loc, roi):
    """
    Calculate bounding box coordinates from match location and ROI size.

    Args:
        max_loc: (x,y) of top-left corner of best match
        roi: Original ROI dimensions (width, height used)

    Returns:
        tuple: (top_left, bottom_right) coordinates
    """
    top_left_coordinates = max_loc
    top_left_x, top_left_y = top_left_coordinates

    # Unpack ROI dimensions (only need width/height)
    _, _, roi_width, roi_height = roi

    # Calculate bottom-right by adding template dimensions to top-left
    bottom_right_coordinates = (
        top_left_x + roi_width,
        top_left_y + roi_height
    )

    return top_left_coordinates, bottom_right_coordinates

def draw_rectangle(frame, max_val, top_left, bottom_right):
    """
    Draw bounding box and confidence text on the frame.

    Args:
        frame: Image to draw on
        max_val: Match confidence value (0.0-1.0)
        top_left: (x,y) of bounding box top-left corner
        bottom_right: (x,y) of bounding box bottom-right corner
    """
    top_left_x, top_left_y = top_left

    # Position text 10 pixels above the bounding box
    text_position = (top_left_x, top_left_y - 10)

    # Draw green rectangle around matched area
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Display match confidence percentage
    cv2.putText(
        frame,
        f"Match: {max_val:.2f}",  # Format to 2 decimal places
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SIZE,
        FONT_COLOR,
        FONT_THICKNESS
    )

def main():
    """Main program loop for template matching tracking."""
    # Initialize camera capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    # Get initial frame and let user select ROI
    sample_frame = get_frame(camera=camera)
    roi = get_roi(frame=sample_frame)

    # Check if ROI was properly selected (width/height > 0)
    if roi[2] == 0 or roi[3] == 0:
        print("Error: Invalid ROI selection")
        camera.release()
        return

    # Extract template from selected ROI
    template = get_template(roi, frame=sample_frame)

    # Main processing loop
    while True:
        try:
            # Get current frame from camera
            frame = get_frame(camera=camera)
        except Exception as e:
            print(f"Camera error: {e}")
            break

        # Perform template matching
        try:
            result = cv2.matchTemplate(frame, template, MATCHING_METHOD)
        except Exception as e:
            print(f"Processing error: {e}")

        # Get best match location and confidence
        res_val, res_loc = get_result(result)

        # Only draw if match confidence meets threshold
        if res_val >= MIN_THRESHOLD:
            top_left, bottom_right = get_tl_and_br_coordinates(res_loc, roi)
            draw_rectangle(frame, res_val, top_left, bottom_right)

        # Always show the current frame (prevent freezing)
        cv2.imshow("Camera Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup resources
    camera.release()
    cv2.destroyAllWindows()

main()
