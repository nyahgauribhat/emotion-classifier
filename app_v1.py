import streamlit as st
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import pygame
from PIL import Image
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Load the face classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Define the emotion labels and the corresponding music files
emotion_labels = ['angry', 'disgust', 'fear','happy','neutral','sad', 'surprised']
emotion_music = {'angry':'music/angry.mp3',
                 'disgust': 'music/disgust.mp3',
                 'fear':'music/fear.mp3',
                 'happy':'music/happy.mp3',
                 'neutral':'music/neutral.mp3',
                 'sad':'music/sad.mp3', 
                 'surprised' :'music/surprised.mp3' }

def load_html(file_path):
    with open('emotion_text'+file_path+'.html', 'r') as file:
        return file.read()

# Function to display the HTML content in Streamlit
def display_html(content):
    st.markdown(content, unsafe_allow_html=True)

# Function to play music for the detected emotion
def play_music(emotion):
    pygame.mixer.music.load(emotion_music[emotion])
    pygame.mixer.music.play()
    pygame.time.wait(3000)  # Play for 5 seconds
    pygame.mixer.music.stop()

# Streamlit app
st.title("Real-Time Emotion Detector")

# Create a video capture object
cap = cv2.VideoCapture(0)

# start_detection = st.button("Start Detection")
# stop_detection = False

start_button = st.button('start detection')
stop_button = st.button('stop detection', key='stop_button')

if start_button:
    stframe = st.empty()
    music_thread = None
    last_played_label = None
    emotion_text = st.empty()

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Play the corresponding music for 5 seconds in a separate thread
                # if label != last_played_label:
                if music_thread is None or not music_thread.is_alive():
                    emotion_text.text(f"Current Emotion: {label}")
                    music_thread = threading.Thread(target=play_music, args=(label,))
                    music_thread.start()
                    last_played_label = label

                    html_content = load_html(label)
                    display_html(html_content)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the image from OpenCV to PIL format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)

        # Display the frame
        stframe.image(img_pil)

        # Check if "Stop Detection" button is pressed
        if stop_button:
            break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
