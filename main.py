import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# 1. Настройка страницы
st.set_page_config(page_title="Raccoon AI", layout="wide")
st.title("🦝 Raccoon AI System")

# 2. ЗАГРУЗКА МОДЕЛЕЙ
MODELS = {
    "face": ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "pose": ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
}

for name, (path, url) in MODELS.items():
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

def get_raccoon(name, h):
    full_path = os.path.join(os.getcwd(), name)
    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        if img is not None: return cv2.resize(img, (h, h))
    return np.zeros((h, h, 3), dtype=np.uint8) + 128

# 3. ИНИЦИАЛИЗАЦИЯ MEDIAPIPE (Один раз при запуске)
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELS["face"][0]), 
    running_mode=VisionRunningMode.VIDEO)
pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELS["pose"][0]), 
    running_mode=VisionRunningMode.VIDEO)

face_det = mp.tasks.vision.FaceLandmarker.create_from_options(face_opts)
pose_det = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts)

# 4. НОВАЯ ФУНКЦИЯ ОБРАБОТКИ (Вместо старого класса)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Зеркалим для удобства
    h, w, _ = img.shape
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ts = int(time.time() * 1000)

    face_res = face_det.detect_for_video(mp_img, ts)
    pose_res = pose_det.detect_for_video(mp_img, ts)

    emotion = "normal.png"

    if pose_res.pose_landmarks:
        p = pose_res.pose_landmarks[0]
        l_wrist, r_wrist = p[15], p[16]
        nose, l_ear, r_ear = p[0], p[7], p[8]
        l_sh, r_sh = p[11], p[12]

        if abs(nose.x - l_ear.x) < abs(nose.x - r_ear.x) * 0.4:
            emotion = "pls.png"
        elif l_wrist.y < nose.y - 0.1 and r_wrist.y < nose.y - 0.1:
            emotion = "beg.png" if abs(l_wrist.x - r_wrist.x) < 0.15 else "cinema.png"
        elif (l_wrist.z < l_sh.z - 0.6) or (r_wrist.z < r_sh.z - 0.6):
            emotion = "gun.png"
        elif l_wrist.y < nose.y and abs(l_wrist.x - nose.x) < 0.2:
            emotion = "hard.png"

    if emotion == "normal.png" and face_res.face_landmarks:
        f = face_res.face_landmarks[0]
        m_open = abs(f[13].y - f[14].y)
        if 0.018 < m_open < 0.045:
            emotion = "cool.png"
        elif m_open >= 0.045:
            emotion = "shock.png"

    raccoon = get_raccoon(emotion, h)
    combined = np.hstack((img, raccoon))
    
    return av.VideoFrame.from_ndarray(combined, format="bgr24")

# 5. ЗАПУСК СТРИМА (Новый синтаксис)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="raccoon-filter",
    video_frame_callback=video_frame_callback, # <--- ТЕПЕРЬ ТАК
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
