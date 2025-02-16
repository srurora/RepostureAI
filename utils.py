import numpy as np
from keypoints import custom_connections, keypoint_indices
import cv2
import streamlit as st

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


def draw_keypoints(image, keypoints):
    """Draws keypoints on the detected pose."""
    if keypoints:
        for key, coords in keypoints.items():
            if coords:
                x, y = int(coords[0]), int(coords[1])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle

def extract_keypoints(results, width, height):
    """Extracts keypoint coordinates from pose landmarks."""
    if not results.pose_landmarks:
        return None  # No landmarks detected
    
    landmarks = results.pose_landmarks.landmark
    keypoints = {
        key: [landmarks[idx].x * width, landmarks[idx].y * height]
        if landmarks[idx].visibility > 0.5 else None  # Ignore low-confidence points
        for key, idx in keypoint_indices.items()
    }
    # print(keypoints)
    return keypoints

# def draw_connections(image, results):
#     """Draws custom connections between keypoints."""
#     h, w, _ = image.shape
#     for start_idx, end_idx in custom_connections:
#         if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
#             start_landmark = results.pose_landmarks.landmark[start_idx]
#             end_landmark = results.pose_landmarks.landmark[end_idx]
#             if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
#                 start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
#                 end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
#                 cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Blue line

def draw_connections(image, results, color=(255, 0, 0), highlight_connections=[]):
    """Draws custom connections between keypoints, with specific connections highlighted."""
    h, w, _ = image.shape
    for start_idx, end_idx in custom_connections:
        # Default color for all connections
        connection_color = color
        
        # If this connection is in the highlight list, change the color
        if (start_idx, end_idx) in highlight_connections:
            connection_color = (0, 0, 255)  # Red for highlighted connections
        
        if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
                cv2.line(image, (start_x, start_y), (end_x, end_y), connection_color, 2)  # Use the determined color


def check_screen_distance(keypoints):
    """Checks if the person is too close to the screen based on Euclidean shoulder distance."""
    if keypoints.get("left_shoulder") and keypoints.get("right_shoulder"):
        left_shoulder = np.array(keypoints["left_shoulder"])
        right_shoulder = np.array(keypoints["right_shoulder"])
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        # print(shoulder_distance)
        if shoulder_distance > 60 and shoulder_distance<105:  # Threshold for being too close
            return True
    return False



