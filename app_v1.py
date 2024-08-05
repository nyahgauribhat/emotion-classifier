from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np 
# from playsound import playsound
import pygame
import streamlit as st
from PIL import Image
import threading
pygame.mixer.init()

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear','Happy','Neutral','Sad', 'Surprised']
emotion_music = {'Angry':'music/angry.mp3',
                 'Disgust': 'music/disgust.mp3',
                 'Fear':'music/fear.mp3',
                 'Happy':'music/happy.mp3',
                 'Neutral':'music/neutral.mp3',
                 'Sad':'music/sad.mp3', 
                 'Surprised' :'music/surprised.mp3' }

def play_music(emotion): 
    pygame.mixer.music.load(emotion_music[emotion])
    pygame.mixer.music.play()
    pygame.time.wait(1000*60*2)
    pygame.mixer.music.stop()

st.markdown(
    """
    <style> 
    .stApp{
        background-color: #e0e0e0;
        font-family: 'Arial', sans-serif; 

    }
    .title{
        text-align: center; 
        color: #333333;
        font-size: 2.5em;
        margin-bottom: 20px;
        animation: fadeIn 2s;
    }
    .header{
        text-align: center;
        background-color: #4caf50;
        color: white;
        padding: 15px;
        font-size: 2.5em;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;

    }
    .sidebar .sidebar-content{
        background-color: #4caf50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-title{
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;

    }
    .footer{
        text-align: center;
        background-color: #4caf50;
        color: white;
        padding: 10px;
        position:fixed;
        bottom: 0;
        width: 100%;
        border-radius: 10px;
        margin-top: 20px;
    }
    .button{
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .video-frame{
        display: flex;
        justify-content: center;
        margin-top: 20px;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .emotion-text{
        text-align: center;
        font-size: 2em;
        color: #333333;
        margin-top: 20px;
        animation: fadeIn 1s;
    }
    .progress{
        text-align: center;
        font-size: 1.2em;
        color: #333333;
        margin-top: 20px;
    }
    @keyframes fadeIn{
    from {opacity:0;}
    to {opacity:1;}
    }
    }

    </style>

    """,
    unsafe_allow_html=True

)

st.markdown('<div class = "header"> Emotion Detector Application </div>', unsafe_allow_html=True)

st.sidebar.title('Controls')
start_button = st.sidebar.button('Start Detection')
stop_button = st.sidebar.button('Stop Detection',key = 'stop_button')
music_toggle = st.sidebar.checkbox('Enable Music', value = True)
# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('videos/emovideo2.mov')
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
frame_counter=0
if start_button:
    stframe=st.empty()
    music_thread = None
    last_played_label = None
    emotion_text=st.empty()
    progress_bar = st.progress(0)
    while True:
        ret, frame = cap.read()
        frame_counter+=1


        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray)
        fontscale = 2.5
        for (x,y,w,h) in faces:
            print(w,h)
            if w<300 or h<300:
                continue
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray])!= 0: 
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis = 0) 

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x,y)
               
                if music_toggle:
                    if music_thread is None or not music_thread.is_alive(): 
                        emotion_text.markdown(f'<p class = "emotion-text"> Detected Emotion: {label}</p>', unsafe_allow_html=True)
                        music_thread = threading.Thread(target = play_music, args =(label,))
                        music_thread.start()
                        last_played_label = label


                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, fontscale,(0,255,0),2)




            else:
                cv2.putText(frame, 'no faces', (30,80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,(0,255,0),2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        stframe.image(img_pil, use_column_width=True)

        if stop_button:
            progress_bar.empty()
            break
cap.release()
cv2.destroyAllWindows()

st.markdown('<div class = "footer">Developed by Nyah</div>',unsafe_allow_html=True)