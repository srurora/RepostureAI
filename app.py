import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Set up Gemini API Key
genai.configure(api_key="AIzaSyDepQ1kX5v-VQuzHAb6qjHgFC58d6NTd1o")

def get_workout_recommendation(prompt):
    """Send prompt to Gemini API and return response."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
st.title("Posture & Rep Counter and AI-Powered Workout Guide")

# Sidebar options for Exercise Tracker
exercise = st.sidebar.selectbox("Choose Exercise", ["Squats", "Arm Raises"])
sets = st.sidebar.number_input("Number of Sets", min_value=1, max_value=10, value=3)
reps = st.sidebar.number_input("Reps per Set", min_value=1, max_value=50, value=10)

# Sidebar options for Workout Plan Generator
st.sidebar.subheader("AI-Powered Workout Plan")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, step=1)
fitness_level = st.sidebar.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])

# State tracking for rep counter and webcam status
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0
if "direction" not in st.session_state:  # 1 = going down, 0 = going up
    st.session_state.direction = 0  

# Start/Stop Webcam Buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Start Webcam", key="start_webcam"):
        st.session_state.webcam_active = True
        st.session_state.rep_count = 0  # Reset rep count when starting the webcam

with col2:
    if st.button("Stop Webcam", key="stop_webcam"):
        st.session_state.webcam_active = False

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Hip or Shoulder
    b = np.array(b)  # Knee or Elbow
    c = np.array(c)  # Ankle or Wrist
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

# Function to check squat or arm raise posture based on angles
def check_posture(keypoints, exercise_type):
    if not keypoints:
        return "No person detected", False

    if exercise_type == "Squats":
        knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        hip_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"])

        if knee_angle < 90:
            return "Too deep, rise up a little!", False
        elif knee_angle > 120:
            return "Squat lower!", False
        elif hip_angle < 60:
            return "Lean forward a bit!", False
        else:
            return "Good Squat!", True  # Correct posture

    elif exercise_type == "Arm Raises":
        shoulder_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        
        if shoulder_angle < 30:
            return "Raise your arms higher!", False
        elif shoulder_angle > 160:
            return "Lower your arms a bit!", False
        else:
            return "Good Arm Raise!", True  # Correct posture

# Function to generate workout recommendation from Gemini
def generate_workout_plan(age, height, weight, fitness_level):
    prompt = f"Design a workout plan for a {age}-year-old, {height} cm tall, {weight} kg person with {fitness_level} fitness level. Include only two exercises: Bicep Curls and Squats. Specify reps, rest time, slow motion, and hold duration."
    return get_workout_recommendation(prompt)

# Workout Plan Generator based on user input
if st.sidebar.button("Get Workout Plan"):
    workout_plan = generate_workout_plan(age, height, weight, fitness_level)
    st.subheader("🏋️‍♂️ Your Personalized Workout Plan")
    st.write(workout_plan)

# Webcam processing for exercise posture tracking
if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            keypoints = {
                "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
            }

            feedback, is_correct = check_posture(keypoints, exercise)

            # Logic for counting reps based on posture direction
            if exercise == "Squats":
                knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])

                if knee_angle < 90 and st.session_state.direction == 0:  # Going down
                    st.session_state.direction = 1
                elif knee_angle > 120 and st.session_state.direction == 1:  # Going up
                    st.session_state.rep_count += 1
                    st.session_state.direction = 0

            elif exercise == "Arm Raises":
                shoulder_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
                if shoulder_angle < 30 and st.session_state.direction == 0:  # Going up
                    st.session_state.direction = 1
                elif shoulder_angle > 160 and st.session_state.direction == 1:  # Going down
                    st.session_state.rep_count += 1
                    st.session_state.direction = 0

            color = (0, 255, 0) if is_correct else (0, 0, 255)

            def draw_stick_figure(image, landmarks, color):
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_pos = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                        end_pos = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                        cv2.line(image, start_pos, end_pos, color, 3)

            draw_stick_figure(frame, results.pose_landmarks.landmark, color)
            cv2.putText(frame, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Displaying rep count with larger, more visible text
        rep_count_text = f"Reps: {st.session_state.rep_count}"
        cv2.putText(frame, rep_count_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        frame_window.image(frame, channels="RGB")

        if not st.session_state.webcam_active:
            break

    cap.release()
    cv2.destroyAllWindows()

