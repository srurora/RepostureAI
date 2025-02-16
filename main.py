import streamlit as st
import cv2
import mediapipe as mp
from keypoints import custom_connections, keypoint_indices, joint_triplets
from utils import calculate_angle, draw_connections, draw_keypoints, extract_keypoints, check_screen_distance
import time
import google.generativeai as genai
import random
# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configure Gemini API
genai.configure(api_key="AIzaSyDepQ1kX5v-VQuzHAb6qjHgFC58d6NTd1o")

def get_workout_recommendation(prompt):
    """Send prompt to Gemini API and return response."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Set up the pages in the sidebar
page = st.sidebar.radio("Choose a Page", ["Posture & Rep Counter", "AI-Powered Workout Guide","Reaction Game"])

# AI-powered workout guide page
if page == "AI-Powered Workout Guide":
    st.title("AI-Powered Workout Plan Generator")

    # Sidebar inputs for workout plan generation
    st.sidebar.subheader("AI-Powered Workout Plan")
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1)
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, step=1)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, step=1)
    fitness_level = st.sidebar.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])

    # Function to generate workout recommendation from Gemini
    def generate_workout_plan(age, height, weight, fitness_level):
        prompt = f"Design a workout plan for a {age}-year-old, {height} cm tall, {weight} kg person with {fitness_level} fitness level. Include only two exercises: Bicep Curls and Squats. Specify reps, rest time, slow motion, and hold duration."
        return get_workout_recommendation(prompt)

    # Workout Plan Generator
    if st.sidebar.button("Get Workout Plan"):
        workout_plan = generate_workout_plan(age, height, weight, fitness_level)
        st.subheader("🏋️‍♂️ Your AI-Powered Workout Plan")
        st.write(workout_plan)
elif page == "Posture & Rep Counter":
    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()


    def display_angles(angles):
        """Displays the joint angles in a scrollable text box (replaces content)."""
        angle_message = ""
        for joint, angle in angles.items():
            if angle is not None:
                angle_message += f"{joint}: {angle:.2f}°\n"
            else:
                angle_message += f"{joint}: Not Detected\n"

        return angle_message

    def setBuffer(category):
        buffer = 20
        if category=='easy':
            buffer = 30
        elif category=="moderate":
            buffer = 25
        return buffer

    def setup_sidebar():
        """Sets up the Streamlit sidebar for user inputs."""
        st.sidebar.title("Exercise Settings")
        category = st.sidebar.selectbox("Exercise Category", ["Easy", "Moderate", "Hard"])
        exercise = st.sidebar.selectbox("Exercise", ["Bicep Curl", "Hand Raises"])
        # sets = st.sidebar.number_input("Number of Sets", min_value=1, max_value=10, value=3)
        reps = st.sidebar.number_input("Reps per Set", min_value=1, max_value=50, value=10)
        
        buffer = setBuffer(category)
        # Store values in session state
        st.session_state.buffer = buffer
        st.session_state.exercise_category = category
        st.session_state.exercise = exercise
        # st.session_state.sets = sets
        st.session_state.reps = reps
        st.session_state.current_rep = 0  # Initialize rep count
        st.session_state.currstage = "straight"  # Initialize current stage
        st.session_state.speed_check = 0  # Speed check flag
        st.session_state.min_time1 = 0  # Time tracking
        st.session_state.max_time1 = 0
        st.session_state.current = 0
        st.session_state.min_angle = []  # List to store angles
        
        return category, exercise, reps

    def handle_webcam_controls():
        """Handles webcam start and stop buttons placed next to each other."""
        col1, col2 = st.sidebar.columns(2)  # Create two columns in the sidebar
        
        with col1:
            start = st.button("Start Webcam")  # Place start button in the first column
        
        with col2:
            stop = st.button("Stop Webcam")  # Place stop button in the second column

        return start, stop


    def calculate_joint_angles(keypoints):
        """Calculates angles for specified joint triplets."""
        angles = {}
        # Compute middle hip point
        if keypoints.get("right_hip") and keypoints.get("left_hip"):
            middle_hip = [
                (keypoints["right_hip"][0] + keypoints["left_hip"][0]) / 2,
                (keypoints["right_hip"][1] + keypoints["left_hip"][1]) / 2
            ]
            keypoints["middle_hip"] = middle_hip
        
        for joint, (a, b, c) in joint_triplets.items():
            angles[joint] = calculate_angle(keypoints.get(a), keypoints.get(b), keypoints.get(c))
        
        return angles

    def hand_raise_rep_counter(keypoints, angles):
        """Counts the reps for a hand raise based on angle and proper form."""
        rep_message = ""
        tbuffer = st.session_state.buffer  # Angle tolerance
        langle = angles.get("left_elbow")  # Left elbow angle
        rangle = angles.get("right_elbow")  # Right elbow angle
        cangle1 = angles.get("left_shoulder")  # Left shoulder angle
        cangle2 = angles.get("right_shoulder")  # Right shoulder angle
        cangle3 = angles.get("left_hip")  # Left hip angle
        cangle4 = angles.get("right_hip")  # Right hip angle
        cangle5 = angles.get("left_knee")  # Left knee angle
        cangle6 = angles.get("right_knee")  # Right knee angle
        
        color = (255, 0, 0)  # Default color (blue)
        highlight_connections = []  # Connections to highlight in red

        # Check if all necessary angles are available (not None)
        if None not in [cangle1, cangle2, cangle3, cangle4, cangle5, cangle6, langle, rangle]:
            if cangle3 > 160 and cangle4 > 160 and cangle5 > 160 and cangle6 > 160:  # Ensure standing straight
                if langle> (180 - tbuffer) and langle < (180 + tbuffer) and rangle > (180 - tbuffer) and langle < (180 + tbuffer):
                    if cangle1 < (5 + tbuffer) and cangle2 < (5 + tbuffer):
                        # Arm fully extended (hand raised)
                        st.session_state.currstage = "straight"
                        color = (0, 255, 0)  # Green for full extension
                        if st.session_state.speed_check == 0:
                            st.session_state.speed_check = 1
                            st.session_state.max_time1 = time.time()
                            st.session_state.current = st.session_state.min_time1
                            st.session_state.min_time1 = 0
                    elif cangle1 > (80 - tbuffer) and cangle2 > (80 - tbuffer) and st.session_state.currstage == "straight":
                        # Hand raise completed (arms fully down)
                        st.session_state.current_rep += 1
                        st.session_state.currstage = "down"
                        st.session_state.min_angle.append(langle)
                        if st.session_state.speed_check == 1:
                            st.session_state.speed_check = 0
                            st.session_state.min_time1 = time.time()
                            st.session_state.current_after = st.session_state.max_time1
                            st.session_state.max_time1 = 0
                        rep_message = f"Rep {st.session_state.current_rep}: Good hand raise! Lower your arms."
                else:
                    highlight_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]  # Arm connections
                    rep_message = f"Please position your arms properly."
            else:
                highlight_connections = [(23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (12, 24)]  # Lower body connections
                rep_message = f"Please stand straight."

            # Track reps completed
            if st.session_state.current_rep >= st.session_state.reps:
                rep_message = f"Completed {st.session_state.reps} reps."
        else:
            rep_message = "Unable to detect all necessary joints for rep counting."

        return rep_message, highlight_connections

        
    def bicep_curl_rep_counter(keypoints, angles):
        """Counts the reps for a bicep curl based on angle and proper form."""
        rep_message = ""
        tbuffer = st.session_state.buffer  # Angle tolerance
        langle = angles.get("left_elbow")  # Left elbow angle
        rangle = angles.get("right_elbow")  # Right elbow angle
        cangle1 = angles.get("left_shoulder")  # Left shoulder angle
        cangle2 = angles.get("right_shoulder")  # Right shoulder angle
        cangle3 = angles.get("left_hip")  # Left hip angle
        cangle4 = angles.get("right_hip")  # Right hip angle
        cangle5 = angles.get("left_knee")  # Left knee angle
        cangle6 = angles.get("right_knee")  # Right knee angle
        
        color = (255, 0, 0)  # Default color (blue)
        highlight_connections = []  # Connections to highlight in red

        # Check if all necessary angles are available (not None)
        if None not in [cangle1, cangle2, cangle3, cangle4, cangle5, cangle6, langle, rangle]:
            if cangle3 > 150 and cangle4 > 150:  # Ensure standing straight
                if cangle1 > (90 - tbuffer) and cangle1 < (90 + tbuffer) and cangle2 > (90 - tbuffer) and cangle2 < (90 + tbuffer):
                    if langle > (175 - tbuffer) and rangle > (175 - tbuffer):
                        # Arm fully extended
                        st.session_state.currstage = "straight"
                        color = (0, 255, 0)  # Green for full extension
                        if st.session_state.speed_check == 0:
                            st.session_state.speed_check = 1
                            st.session_state.max_time1 = time.time()
                            st.session_state.current = st.session_state.min_time1
                            st.session_state.min_time1 = 0
                        # rep_message = f"Rep {st.session_state.current_rep + 1}: position your arm."
                    elif langle < (20 + tbuffer) and rangle < (20 + tbuffer) and st.session_state.currstage == "straight":
                        # Bicep curl completed
                        st.session_state.current_rep += 1
                        st.session_state.currstage = "curl"
                        st.session_state.min_angle.append(langle)
                        if st.session_state.speed_check == 1:
                            st.session_state.speed_check = 0
                            st.session_state.min_time1 = time.time()
                            st.session_state.current_after = st.session_state.max_time1
                            st.session_state.max_time1 = 0
                        rep_message = f"Rep {st.session_state.current_rep}: Good curl! Bring your arms up."
                else:
                    highlight_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]  # Arm connections
                    rep_message = f"Please position your arms properly."
            else:
                highlight_connections = [(23, 25), (25, 27), (24, 26), (26, 28),(11,23),(12,24)]  # Lower body connections
                rep_message = f"Please stand straight."
            
            # Track reps completed
            if st.session_state.current_rep >= st.session_state.reps:
                rep_message = f"Completed {st.session_state.reps} reps."
        else:
            rep_message = "Unable to detect all necessary joints for rep counting."

        return rep_message, highlight_connections



    def process_frame(image):
        """Processes a frame by detecting pose keypoints and drawing them."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        message = ""
        angle_message =""
        
        if results.pose_landmarks:
            h, w, _ = image.shape
            keypoints = extract_keypoints(results, w, h)
            # add screen constraints
            if check_screen_distance(keypoints):
                draw_keypoints(image, keypoints)
                # draw_connections(image, results)
                angles = calculate_joint_angles(keypoints)
                angle_message=display_angles(angles)
                
                # Perform Bicep Curl rep counting only if the exercise is "Bicep Curl"
                if st.session_state.exercise.lower() == "bicep curl":
                    message, highlights = bicep_curl_rep_counter(keypoints, angles)
                elif st.session_state.exercise.lower() == "hand raises":
                    message, highlights = hand_raise_rep_counter(keypoints, angles) 
                
                draw_connections(image, results, (255, 0, 0), highlights)

            else:
                message = "Please move away from the camera to make your whole body visible."
        
        return image, message, angle_message

    def run_webcam():
        """Handles the webcam stream and frame processing."""
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])
        message_placeholder = st.empty()
        rep_placeholder = st.sidebar.empty()
        # angle_rep_placeholder = st.empty()
        

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, message,angle_message = process_frame(frame)
            frame = cv2.flip(frame, 1)
            frame_window.image(frame, channels="BGR")
            message_placeholder.warning(message)
            # angle_rep_placeholder.warning(angle_message)
            if st.session_state.current_rep <= st.session_state.reps:
                # st.sidebar.write(f"Rep Count: {st.session_state.current_rep} / {st.session_state.reps}")
                rep_placeholder.markdown(f"Rep Count: {st.session_state.current_rep} / {st.session_state.reps}")

        cap.release()

    # Main execution
    st.title("RE POSTURE")
    exercise_category, exercise, reps = setup_sidebar()
    start_button, stop_button = handle_webcam_controls()

    if start_button:
        run_webcam()
elif page == "Reaction Game":
    st.subheader("Reaction Time Game")
    
    # Game Variables
    screen_width = 640
    screen_height = 480
    circle_radius = 30
    circles = [(100, 150), (100, 300), (100, 450), (540, 150), (540, 300), (540, 450)]
    active_circle = random.choice(circles)
    last_activation_time = time.time()
    score = 0
    start_time = time.time()
    game_duration = 30  # Play for 30 seconds
    
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    score_placeholder = st.sidebar.empty()
    timer_placeholder = st.sidebar.empty()
    warning_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        elapsed_time = time.time() - start_time
        if elapsed_time > game_duration:
            break  # End the game after 30 seconds
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        keypoints = {}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
                
                # Check if the finger is touching the active circle
                if ((x - active_circle[0]) ** 2 + (y - active_circle[1]) ** 2) ** 0.5 < circle_radius+2.0:
                    score += 1
                    active_circle = random.choice(circles)  # Activate new circle
                    last_activation_time = time.time()
        

        
        # Draw circles
        for circle in circles:
            color = (0, 255, 0) if circle == active_circle else (255, 0, 0)
            cv2.circle(frame, circle, circle_radius, color, -1)
        
        frame_window.image(frame, channels="BGR")
        score_placeholder.write(f"Score: {score}")
        timer_placeholder.write(f"Time Left: {int(game_duration - elapsed_time)}s")
    
    st.write(f"Game Over! Your final score is {score}")
    cap.release()