from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np 
import threading 
# from playsound import playsound
import pygame
import streamlit as st
from PIL import Image

st.title('emotion detector')
pygame.mixer.init()

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')
emotion_labels = ['angry', 'disgust', 'fear','happy','neutral','sad', 'surprised']
emotion_music = {'angry':'music/angry.mp3',
                 'disgust': 'music/disgust.mp3',
                 'fear':'music/fear.mp3',
                 'happy':'music/happy.mp3',
                 'neutral':'music/neutral.mp3',
                 'sad':'music/sad.mp3', 
                 'surprised' :'music/surprised.mp3' }
# def play_music(music_file): 
#     playsound(music_file)
def play_music(music_file): 
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()
    sleep(3)
    pygame.mixer.music.stop()
# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('videos/emovideo2.mov')
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
frame_counter=0
start_button = st.button('start detection')
stop_button = st.button('stop detection', key='stop_button')
if start_button:
    stframe=st.empty()
    while True:
        ret, frame = cap.read()
        frame_counter+=1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        stframe.image(img_pil)
        

        if frame_counter%15!=0 and frame_counter!=1:
            continue
        # if not ret:
        #     print("Error: Failed to capture image")
        #     continue
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
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, fontscale,(0,255,0),2)

                music_file = emotion_music.get(label)
                if music_file:
                    play_music(music_file)

                    # threading.Thread(target = play_music, args = (music_file,)).start()
                    # sleep(5)
            else:
                cv2.putText(frame, 'no faces', (30,80), cv2.FONT_HERSHEY_SIMPLEX, fontscale,(0,255,0),2)

        # cv2.imshow('emotion_detector', frame)
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_pil = Image.fromarray(frame)
        # stframe.image(img_pil)

        if stop_button:
            break
cap.release()
cv2.destroyAllWindows()
    





