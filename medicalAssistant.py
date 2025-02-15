# import streamlit as st
# import requests

# # Hugging Face API settings
# # Hugging Face API details
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
# HEADERS = {"Authorization": "Bearer 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"}

# #AIzaSyCzHdLklEwSQFpadyyStCDZdn61PR0XxJc

# def query_huggingface(prompt):
#     """Send a prompt to Hugging Face model and return the response."""
#     response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
#     if response.status_code == 200:
#         return response.json()[0]["generated_text"]
#     else:
#         return "Error fetching response. Try again."

# # Streamlit App
# st.title("AI-Powered Fitness Guide 🏋️‍♂️")

# st.write("Enter your details to receive a personalized exercise recommendation.")

# # User Input Form
# with st.form("fitness_form"):
#     age = st.number_input("Age", min_value=10, max_value=100, step=1)
#     height = st.number_input("Height (cm)", min_value=100, max_value=250, step=1)
#     weight = st.number_input("Weight (kg)", min_value=30, max_value=200, step=1)
#     fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
    
#     submitted = st.form_submit_button("Get Workout Plan")

# if submitted:
#     prompt = (
#         f"Design a workout for a {age}-year-old, {height} cm tall, {weight} kg person with {fitness_level} fitness level. "
#         "Include only two exercises: Bicep Curls and Squats. Specify reps, rest time, slow motion, and hold duration."
#     )
    
#     response = query_huggingface(prompt)
    
#     st.subheader("Your Personalized Workout Plan")
#     st.write(response)




import streamlit as st
import google.generativeai as genai

# Set up Gemini API Key
genai.configure(api_key="AIzaSyCzHdLklEwSQFpadyyStCDZdn61PR0XxJc")

def get_workout_recommendation(prompt):
    """Send prompt to Gemini API and return response."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
st.title("💪 AI-Powered Workout Guide")

st.write("Enter your details to receive a personalized workout plan.")

# User Input Form
with st.form("fitness_form"):
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, step=1)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, step=1)
    fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
    
    submitted = st.form_submit_button("Get Workout Plan")

if submitted:
    # Create prompt for Gemini API
    prompt = (
        f"Design a workout plan for a {age}-year-old, {height} cm tall, {weight} kg person with {fitness_level} fitness level. "
        "Include only two exercises: Bicep Curls and Squats. Specify reps, rest time, slow motion, and hold duration."
    )

    # Fetch AI response
    response = get_workout_recommendation(prompt)

    # Display AI-generated workout
    st.subheader("🏋️‍♂️ Your Personalized Workout Plan")
    st.write(response)

