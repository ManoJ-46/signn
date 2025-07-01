"""
Sign Language Recognition System
Streamlit Cloud–ready with correct image paths and mobile support
"""

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
from pathlib import Path
from CNNModel import CNNModel

# ───────────────────────── Path helper ─────────────────────────
def get_base_path() -> Path:
    """Return the directory where this script resides."""
    return Path(__file__).parent

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ──────────────────────── CSS (mobile + styling) ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
* { font-family: 'Montserrat', sans-serif !important; }
video { width:100% !important; height:auto !important; }
@media (max-width:480px) { .block-container { padding:0 0.5rem; } }
.card { background: rgba(255,255,255,0.8); backdrop-filter: blur(10px);
       border-radius:15px; padding:25px; box-shadow:0 8px 16px rgba(0,0,0,0.1);
       margin-bottom:20px; border:none; }
.main-title { font-size:2.5rem !important; font-weight:700 !important;
               color:#2c3e50 !important; text-align:center;
               margin-bottom:1rem; text-shadow:2px 2px 4px rgba(0,0,0,0.1); }
.prediction-badge { font-size:1.5rem; font-weight:bold; padding:10px 20px;
                     border-radius:30px; box-shadow:0 4px 8px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Mediapipe & Model ─────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

@st.cache_resource
def load_model():
    path = get_base_path() / "trained.pth"
    model = CNNModel()
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model

model = load_model()
alphabet = {i: chr(65 + i) for i in range(26)}

# ───────────────────────── Prediction ─────────────────────────
def predict_frame(frame_bgr, model, classes):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    if not res.multi_hand_landmarks:
        return frame_bgr, None
    for hand in res.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr, hand, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        xs, ys, zs = [], [], []
        data = {}
        for lm in hand.landmark:
            xs.append(lm.x); ys.append(lm.y); zs.append(lm.z)
        for i, lm_name in enumerate(mp_hands.HandLandmark):
            lm = hand.landmark[i]
            data[f"{lm_name.name}_x"] = lm.x - min(xs)
            data[f"{lm_name.name}_y"] = lm.y - min(ys)
            data[f"{lm_name.name}_z"] = lm.z - min(zs)
        arr = np.reshape(np.array(list(data.values())), (1, 63, 1))
        tensor = torch.from_numpy(arr).float()
        with torch.no_grad():
            _, pred = torch.max(model(tensor).data, 1)
        ch = classes[pred.item()]
        h, w = frame_bgr.shape[:2]
        x1, y1 = int(min(xs)*w)-10, int(min(ys)*h)-10
        x2, y2 = int(max(xs)*w)+10, int(max(ys)*h)+10
        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,0,0), 4)
        cv2.putText(frame_bgr, ch, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)
        return frame_bgr, ch
    return frame_bgr, None

# ─────────────────────── Video callback ───────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    proc, _ = predict_frame(img, model, alphabet)
    return av.VideoFrame.from_ndarray(proc, format="bgr24")

# ─────────────────────── Image loader ───────────────────────
def load_sign_image(letter):
    base = get_base_path()
    p = base / "alphabets" / f"{letter}.jpg"
    if p.exists():
        return Image.open(str(p))
    # fallback
    img = Image.new("RGB", (200,200), "white")
    d = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 100)
    except: font = ImageFont.load_default()
    tw = d.textlength(letter, font=font)
    d.text(((200-tw)/2,50), letter, fill="black", font=font)
    return img

# ────────────────────── Game logic ──────────────────────
def guess_game():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Guess the Sign")
    if 'score' not in st.session_state:
        st.session_state.score = 0
        st.session_state.total = 0
        st.session_state.letter = random.choice(list(alphabet.values()))
        st.session_state.start = time.time()
        st.session_state.locked = False
        st.session_state.result = None

    TIME_LIMIT = 10
    elapsed = int(time.time() - st.session_state.start)
    left = max(0, TIME_LIMIT - elapsed)
    if not st.session_state.locked:
        st_autorefresh(interval=1000, key="timer")

    st.info(f"Score: {st.session_state.score}/{st.session_state.total}")
    st.warning(f"Time left: {left}s")

    img = load_sign_image(st.session_state.letter)
    st.image(img, caption="Which letter is this?", width=300)

    if left == 0 and not st.session_state.locked:
        st.session_state.total += 1
        st.session_state.result = f"Time's up! It was '{st.session_state.letter}'"
        st.session_state.locked = True

    if not st.session_state.locked:
        guess = st.selectbox("Your guess:", list(alphabet.values()))
        if st.button("Submit"):
            st.session_state.total += 1
            if guess == st.session_state.letter:
                st.session_state.score += 1
                st.session_state.result = "Correct!"
            else:
                st.session_state.result = f"Incorrect! It was '{st.session_state.letter}'"
            st.session_state.locked = True

    if st.session_state.locked:
        st.success(st.session_state.result) if "Correct" in st.session_state.result else st.error(st.session_state.result)
        if st.button("Next"):
            st.session_state.letter = random.choice(list(alphabet.values()))
            st.session_state.start = time.time()
            st.session_state.locked = False
            st.session_state.result = None

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────── Main UI ───────────────────────
st.markdown('<p class="main-title">Sign Language Recognition</p>', unsafe_allow_html=True)
mode = st.sidebar.selectbox("Select Mode:", ["Live Detection","English to Sign","Guess the Sign"])

if mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Detection")
    rtc = {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    webrtc_streamer(
        key="live", video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV, rtc_configuration=rtc,
        media_stream_constraints={"video":True,"audio":False}
    )
    st.caption("Grant camera permission.")
    st.subheader("Or use photo capture (mobile)")
    with st.expander("Tap to Snap"):
        f = st.camera_input("Take a photo")
        if f:
            img = Image.open(f).convert("RGB")
            fr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            fr = cv2.flip(fr,1)
            p, pred = predict_frame(fr, model, alphabet)
            c1, c2 = st.columns(2)
            c1.image(p, caption="Processed", use_column_width=True)
            c2.markdown(f'<div class="prediction-badge">Detected: {pred}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif mode == "English to Sign":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("English to Sign Language")
    text = st.text_input("Enter text:")
    if st.button("Convert"):
        seq = ''.join([c.upper() for c in text if c.isalpha()])
        if not seq:
            st.warning("No letters to convert.")
        else:
            cols = st.columns(5)
            for i, ch in enumerate(seq):
                img = load_sign_image(ch)
                with cols[i%5]:
                    st.image(img, caption=ch, use_column_width=True)
                if (i+1)%5==0 and (i+1)<len(seq):
                    cols = st.columns(5)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    guess_game()
