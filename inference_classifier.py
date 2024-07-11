import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict
import google. generativeai as genai
import os
from gtts import gTTS
import pygame
from io import BytesIO

def speak(text):
    # Create a BytesIO object to store the audio
    audio_stream = BytesIO()
    
    # Create gTTS object
    tts = gTTS(text=text, lang='en')
    
    # Save the speech to the BytesIO object
    tts.write_to_fp(audio_stream)
    
    # Rewind the BytesIO object
    audio_stream.seek(0)
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load audio stream into pygame mixer
    pygame.mixer.music.load(audio_stream)
    
    # Play the audio
    pygame.mixer.music.play()
    
    # Wait until the audio is done playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


os.environ['GOOGLE_API_KEY']="AIzaSyCN7WugKz_m_gB85tWgEfTg8EJc84Gx0ho"

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model0 = genai.GenerativeModel('gemini-pro')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'I', 1: 'Thirsty', 2: 'Hungry'}

capture_duration = 10
start_time = time.time()

predicted_characters = []

while True:

    data_aux = []
    x_ = []
    y_ = []

    
    ret, frame = cap.read()

    

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Resize data_aux to have only 42 features
        data_aux = data_aux[:42]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        

        predicted_characters.append(predicted_character)
        elapsed_time = time.time() - start_time
        if elapsed_time >= capture_duration:
            cap.release()
            break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

char_freq = defaultdict(int)
for item in predicted_characters:
    char_freq[item] += 1
char_freq = dict(char_freq)
char_freq = {k: v for k, v in char_freq.items() if v >= 15}

sent_char = set(char_freq.keys())

cap.release()
cv2.destroyAllWindows()


query = "Create a declarative sentence using following set of words: " + str(sent_char)

response = model0.generate_content(query)
output = response.text
print(output)

speak(output)