import streamlit as st
import google.generativeai as genai

# Set up Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")

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
