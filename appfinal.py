import streamlit as st 
import cv2
import time 
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image 
import numpy as np
import threading 
import datetime 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

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
    'disgust': 'music/angry.wav',
    'fear': 'music/angry.wav',
    'happy': 'music/happy.wav',
    'neutral': 'music/neutral.wav',
    'sad': 'music/sad.wav',
    'surprised': 'music/surprised.wav'
}

def play_music(emotion):
    st.audio(emotion_music[emotion], format="audio/wav", loop=False)

def load_html(file_path):
    with open('static/emotion_text/' + file_path + '.html', 'r') as file:
        return file.read()

def display_html(content, placeholder):
    placeholder.html(content, unsafe_allow_html=True)
    with open('./static/emotion_text/style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

class EmotionProcessor(VideoProcessorBase):
    def init(self):
        self.model = emotion_classifier
        self.face_classifier = face_classifier
        self.label = None
        self.music_toggle = True
        self.last_played_time = None  # Track the last time music was played
        self.play_interval_seconds = 120 

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 150 or h < 150:
                    continue
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.model.predict(roi)[0]
                emotion_label = emotion_labels[prediction.argmax()]
                return emotion_label, (x, y, w, h)

        return None, None
    def should_play_music(self):
        # Check if it's time to play music again
        if self.last_played_time is None:
            return True  # No previous time, allow to play
        elapsed_time = time.time() - self.last_played_time
        if elapsed_time > self.play_interval_seconds:
            return True
        return False

def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        label, face_rect = self.detect_emotion(img)
        if label is not None:
            x, y, w, h = face_rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Play music only if the interval has passed and music toggle is enabled
            if self.music_toggle and self.should_play_music():
                play_music(label)
                self.last_played_time = time.time()  # Update the last played time

        return img

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');

    * {
        font-family: "Poppins", sans-serif;
    }
    .stApp {
        background-color: white;
        font-family: "Poppins", sans-serif;
    }
    .title {
        text-align: center;
        color: #333333;
        font-size: 2.5em;
        margin-bottom: 20px;
        animation: fadeIn 2s;
        font-family: "Poppins", sans-serif;
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
        background-color: white;
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
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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

# WebRTC Streamer
def stopMusic():
    st.image("static/poster.png")

def addSecs(tm, secs):
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    return fulldate.time()

def startMusic(htmlcontent, htmlcontentarea, musicthread):
    display_html(htmlcontent, htmlcontentarea)
    musicthread.start()

# Display initial placeholder image
def initialPlaceholder():
    if 'initialload' not in st.session_state:
        st.image("static/poster.png")
        st.session_state['initialload'] = True

initialPlaceholder()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
st.sidebar.title('Controls')
start_button = st.sidebar.button('Start Detection')  # Removed on_click to let WebRTC handle it
stop_button = st.sidebar.button('Stop Detection', on_click=stopMusic)
about_us_button = st.sidebar.markdown('<a href="https://sangeetandi.com" target="_blank">About Us</a>', unsafe_allow_html=True)
music_toggle = st.sidebar.checkbox('Enable Music', value=True)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.music_toggle = music_toggle
