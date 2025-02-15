# RepostureAI
An AI-powered app that corrects posture in real-time, tracks reps, and ensures safe, effective workouts for injury-free fitness
# Posture & Rep Counter and AI-Powered Workout Guide

This project uses computer vision and AI to track your exercise form (specifically squats and arm raises) and provide real-time feedback. It also includes an AI-powered workout generator that suggests personalized workout plans based on your age, height, weight, and fitness level.

## Features
- **Real-time Posture Feedback**: Track squats and arm raises with visual feedback on your form.
- **Rep Counter**: Counts the number of reps for squats and arm raises, ensuring accuracy.
- **AI Workout Plan Generator**: Uses Gemini AI to generate personalized workout plans based on user information.
- **Exercise Tracking**: Count reps and provide suggestions for improvement during exercises.
  
## Prerequisites

Ensure that you have the following installed:
- **Python 3.x** (Recommended version: 3.7+)
- **Streamlit**: A framework for building interactive web applications in Python
- **OpenCV**: A library for computer vision
- **Mediapipe**: A framework for building multimodal applied machine learning pipelines
- **Google Gemini API**: For generating personalized workout plans
  
You can install the required dependencies by running the following command:

```bash
pip install streamlit opencv-python mediapipe google-generativeai
