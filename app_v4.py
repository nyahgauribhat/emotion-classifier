import streamlit as st
import cv2
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pygame
import threading
import datetime

pygame.mixer.init()

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load the model with error handling

@st.cache_resource
def load_emotion_classifier():
    try:
        emotion_classifier = load_model('model.h5')
        return emotion_classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

emotion_classifier = load_emotion_classifier()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emotion_music = {
    'angry': 'music/angry.wav',
    'disgust': 'music/disgust.mp3',
    'fear': 'music/fear.wav',
    'happy': 'music/happy.wav',
    'neutral': 'music/neutral.wav',
    'sad': 'music/sad.wav',
    'surprised': 'music/surprised.wav'
}

# st.image("static/poster.png")
         
def load_html(file_path):
    with open('static/emotion_text/' + file_path + '.html', 'r') as file:
        return file.read()

def display_html(content, placeholder):
    # placeholder.markdown(content, unsafe_allow_html=True)
    placeholder.html(content)

def play_music(emotion):
    pygame.mixer.music.load(emotion_music[emotion])
    pygame.mixer.music.play()
    # pygame.time.wait(10000)
    # pygame.mixer.music.stop()
    return True

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #333333;
        font-size: 2.5em;
        margin-bottom: 20px;
        animation: fadeIn 2s;
    }
    .header {
        text-align: center;
        background-color: #4caf50;
        color: white;
        padding: 15px;
        font-size: 2.5em;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #4caf50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-title {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        background-color: #4caf50;
        color: white;
        padding: 10px;
        position: fixed;
        bottom: 0;
        width: 100%;
        border-radius: 10px;
        margin-top: 20px;
    }
    .button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .video-frame {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .emotion-text {
        text-align: center;
        font-size: 2em;
        color: #333333;
        margin-top: 20px;
        animation: fadeIn 1s;
    }
    .progress {
        text-align: center;
        font-size: 1.2em;
        color: #333333;
        margin-top: 20px;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

with open ('intro.html','r') as file: 
    html_content = file.read()
    
# st.markdown('<div class="header"> Emotion Detector Application </div>', unsafe_allow_html=True)
print(html_content)
st.markdown(html_content, unsafe_allow_html=True)


st.sidebar.title('Controls')
start_button = st.sidebar.button('Start Detection')
stop_button = st.sidebar.button('Stop Detection', key='stop_button')
music_toggle = st.sidebar.checkbox('Enable Music', value=True)
audio_start_time = None

def addSecs(tm, secs):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    return fulldate.time()



if start_button:
    start_time = time.time()
    stframe = st.empty()
    music_thread = None
    last_played_label = None
    emotion_text = st.empty()
    progress_bar = st.progress(0)
    html_content_area = st.empty()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Error: Could not open video capture.")
        exit()

    while True:
        if audio_start_time!=None:
            restart_time = addSecs(audio_start_time,10)
            if(datetime.datetime.now().time()<restart_time):
                if cap.isOpened():
                    cap.release()
                    cv2.destroyAllWindows()
                continue
        if audio_start_time!=None:
            pygame.mixer.music.stop()
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from video capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        fontscale = 2.5

        for (x, y, w, h) in faces:
            if w < 100 or h < 100:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)

                if music_toggle:
                    if music_thread is None or not music_thread.is_alive():
                        emotion_text.markdown(f'<p class="emotion-text">Detected Emotion: {label}</p>', unsafe_allow_html=True)
                        html_content = load_html(label)
                        html_content_area.empty()
                        display_html(html_content, html_content_area)
                        music_thread = threading.Thread(target=play_music, args=(label,))
                        music_thread.start()
                        audio_start_time = datetime.datetime.now().time()
                        last_played_label = label

                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'no faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        stframe.image(img_pil, use_column_width=True)

        if stop_button:
            progress_bar.empty()
            pygame.mixer.music.stop()
        # time.sleep(10000)
                


# cap.release()
# cv2.destroyAllWindows()