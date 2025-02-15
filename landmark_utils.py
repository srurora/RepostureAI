import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define keypoint indices
keypoint_indices = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# Function to calculate angle between three keypoints
def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None  # Handle missing points

    a, b, c = np.array(a), np.array(b), np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

# Extract keypoint coordinates
def extract_keypoints(results, width, height):
    if not results.pose_landmarks:
        return None  # No landmarks detected

    landmarks = results.pose_landmarks.landmark
    keypoints = {
        key: [landmarks[idx].x * width, landmarks[idx].y * height]
        if landmarks[idx].visibility > 0.5 else None  # Ignore low-confidence points
        for key, idx in keypoint_indices.items()
    }
    return keypoints

# Function to check squat posture based on angles
def check_squat_posture(keypoints):
    if not keypoints:
        return "No person detected"

    # Corrected key names
    knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
    hip_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"])

    if knee_angle is None or hip_angle is None:
        return "Keypoints missing"

    if knee_angle < 90:
        return "Too deep, rise up a little!"
    elif knee_angle > 120:
        return "Squat lower!"
    elif hip_angle < 60:
        return "Lean forward a bit!"
    else:
        return "Good Squat!"
    
# Function to process frame and annotate
def process_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape
    keypoints = extract_keypoints(results, w, h)
    
    feedback = check_squat_posture(keypoints) if keypoints else "No person detected"

    # Draw keypoints if detected
    if keypoints:
        for _, point in keypoints.items():
            if point:  # Check if keypoint exists
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    cv2.putText(image, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image, feedback