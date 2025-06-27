import cv2

input("Look at the camera. Press Enter to start face detection...")

# Path to the pre-trained Haar Cascade model for face detection
HAARCASCADE_MODEL = "haarcascade_frontalface_default.xml"

# Load the pre-trained face detector model from OpenCV's data directory
# Note: cv2.data.haarcascades contains the path to OpenCV's built-in Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAARCASCADE_MODEL)

# Initialize video capture from default camera (index 0)
cap = cv2.VideoCapture(0)

# Read a single frame from the camera
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab initial frame from camera")
    exit()  # Exit if frame capture fails

# Mirror the frame horizontally for more intuitive user experience (like a mirror)
mirrored_frame = cv2.flip(frame, 1)

# Convert the color frame to grayscale (Haar Cascades work on grayscale images)
gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)

# Detect faces using the Haar Cascade classifier
# Parameters:
# - scaleFactor: How much the image size is reduced at each scale (1.1 = 10% reduction)
# - minNeighbors: How many neighbors each candidate rectangle should have to retain it
# - minSize: Minimum possible object size (not specified here)
# - flags: (legacy parameter, not needed in newer OpenCV)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

# Drawing parameters for the face bounding boxes
BORDER_COLOR = (0, 255, 0)  # Green color in BGR format
BORDER_THICKNESS = 2         # Thickness of the rectangle border in pixels

# Draw rectangles around detected faces
for (x_coord, y_coord, width, height) in faces:
    cv2.rectangle(
        mirrored_frame,                   # Image to draw on
        (x_coord, y_coord),               # Top-left corner coordinates
        (x_coord + width, y_coord + height),  # Bottom-right corner coordinates
        BORDER_COLOR,                     # Rectangle color
        BORDER_THICKNESS                  # Line thickness
    )

# Display the resulting frame with face detections
cv2.imshow("Face Detection", mirrored_frame)

# Wait indefinitely for a key press (0 = wait forever)
# Note: The window will close when any key is pressed
cv2.waitKey(0)

# Release the camera resource and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
