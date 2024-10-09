import streamlit as st
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import pygame
from PIL import Image

# Initialize pygame mixer
pygame.mixer.init()

# Load the face classifier and the emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

# Define the emotion labels and the corresponding sound files
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_sounds = {
    'Angry': 'sounds/angry.wav',
    'Disgust': 'sounds/disgust.wav',
    'Fear': 'sounds/fear.wav',
    'Happy': 'sounds/happy.wav',
    'Neutral': 'sounds/neutral.wav',
    'Sad': 'sounds/sad.wav',
    'Surprise': 'sounds/surprise.wav'
}

# Function to play sound for the detected emotion
def play_sound(emotion):
    pygame.mixer.music.load(emotion_sounds[emotion])
    pygame.mixer.music.play()
    pygame.mixer.music.stop()

# Streamlit app
st.title("Real-Time Emotion Detector")

# Create a video capture object
cap = cv2.VideoCapture(1)
frame_counter = 0

if st.button("Start Detection"):
    stframe = st.empty()
    while True:
        _, frame = cap.read()
        frame_counter += 1

        # Skip processing for 3 frames, process every 4th frame
        if frame_counter % 4 != 0:
            continue

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

                # Play the corresponding sound for 5 seconds
                play_sound(label)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the image from OpenCV to PIL format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)

        # Display the frame
        stframe.image(img_pil)

        # Break the loop if 'q' is pressed
        if st.button("Stop Detection"):
            break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
