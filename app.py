# host.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import time
import mediapipe as mp
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
from CNNModel import CNNModel

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

# === CSS ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
* { font-family: 'Montserrat', sans-serif !important; }
video { width: 100% !important; height: auto !important; }
@media (max-width: 480px) {
  .block-container { padding: 0 0.5rem; }
}
.stButton>button {
    animation: pulse 2s infinite;
    transition: all 0.3s ease;
    border-radius: 12px !important;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.card {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.main-title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #2c3e50 !important;
    text-align: center;
}
.prediction-badge {
    font-size: 1.5rem;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 30px;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# === Load Model ===
@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("trained.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === Hand Detector ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# === Alphabet Mapping ===
alphabet_map = {i: chr(65 + i) for i in range(26)}

# === Prediction Function ===
def predict_sign(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return frame, None

    landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    data = []
    for lm in landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    data = np.array(data).reshape(1, -1)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(2)

    with torch.no_grad():
        out = model(data)
        pred = torch.argmax(out, dim=1).item()
        letter = alphabet_map[pred]

    h, w, _ = frame.shape
    cv2.putText(frame, letter, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return frame, letter

# === WebRTC Real-Time Detection ===
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    processed, _ = predict_sign(img, model)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# === Load Sign Image or Generate ===
def load_sign_image(letter):
    path = f"alphabets/{letter}.jpg"
    if os.path.exists(path):
        return Image.open(path)
    else:
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
        w, h = draw.textsize(letter, font=font)
        draw.text(((200-w)/2, (200-h)/2), letter, fill='black', font=font)
        return img

# === Guess the Letter Game ===
def guess_game():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🎮 Guess the Sign Language Character")

    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'round' not in st.session_state:
        st.session_state.round = 0
    if 'target' not in st.session_state:
        st.session_state.target = random.choice(list(alphabet_map.values()))
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()

    time_left = max(0, 10 - int(time.time() - st.session_state.start_time))
    if time_left == 0:
        st.warning(f"⏰ Time's up! The correct answer was {st.session_state.target}")
        if st.button("Next"):
            st.session_state.target = random.choice(list(alphabet_map.values()))
            st.session_state.start_time = time.time()
        return

    st.warning(f"⏳ Time left: {time_left}s")
    st.image(load_sign_image(st.session_state.target), width=200)

    choice = st.selectbox("Your guess:", list(alphabet_map.values()))
    if st.button("Submit"):
        if choice == st.session_state.target:
            st.success("✅ Correct!")
            st.session_state.score += 1
        else:
            st.error(f"❌ Incorrect! It was {st.session_state.target}")
        st.session_state.round += 1
        if st.button("Next"):
            st.session_state.target = random.choice(list(alphabet_map.values()))
            st.session_state.start_time = time.time()
    st.info(f"Score: {st.session_state.score} / {st.session_state.round}")
    st.markdown('</div>', unsafe_allow_html=True)

# === UI Routing ===
st.markdown('<p class="main-title">Sign Language Recognition System</p>', unsafe_allow_html=True)

mode = st.sidebar.selectbox("Select Mode", ["Live Detection", "English to Sign", "Guess the Character"])

if mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📷 Live Sign Detection")
    webrtc_streamer(
        key="live",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        mode=WebRtcMode.SENDRECV
    )
    st.markdown('</div>', unsafe_allow_html=True)

elif mode == "English to Sign":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔤 English to Sign Language")
    text = st.text_input("Enter a word:")
    if st.button("Convert"):
        if not text:
            st.error("Please enter a word.")
        else:
            filtered = ''.join([c.upper() for c in text if c.isalpha()])
            imgs = [load_sign_image(c) for c in filtered]
            cols = st.columns(5)
            for i, img in enumerate(imgs):
                with cols[i % 5]:
                    st.image(img, caption=filtered[i], use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif mode == "Guess the Character":
    guess_game()
