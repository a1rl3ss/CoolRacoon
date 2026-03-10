import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# 1. Настройка страницы Streamlit
st.set_page_config(page_title="Raccoon AI System", layout="wide")
st.title("🦝 Raccoon AI System")
st.write("Нажми **START** и разреши доступ к камере. Если видео не грузится, попробуй выключить VPN.")

# 2. ЗАГРУЗКА МОДЕЛЕЙ MEDIAPIPE
MODELS = {
    "face": ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "pose": ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
}

for name, (path, url) in MODELS.items():
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

# Функция загрузки картинок енота
def get_raccoon(name, h):
    full_path = os.path.join(os.getcwd(), name)
    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        if img is not None: return cv2.resize(img, (h, h))
    # Заглушка, если файл не найден
    blank = np.zeros((h, h, 3), dtype=np.uint8) + 50
    cv2.putText(blank, name, (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return blank

# Инициализация детекторов
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

# 3. ФУНКЦИЯ ОБРАБОТКИ КАДРОВ
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Зеркалим
    h, w, _ = img.shape
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ts = int(time.time() * 1000)

    # Детекция (лицо и поза)
    face_res = face_det.detect_for_video(mp_img, ts)
    pose_res = pose_det.detect_for_video(mp_img, ts)

    emotion = "normal.png"

    # --- ЛОГИКА ЖЕСТОВ ---
    if pose_res.pose_landmarks:
        p = pose_res.pose_landmarks[0]
        l_wrist, r_wrist = p[15], p[16]
        nose, l_ear, r_ear = p[0], p[7], p[8]
        l_sh, r_sh = p[11], p[12]

        # 1. Поворот головы влево -> pls.png
        if abs(nose.x - l_ear.x) < abs(nose.x - r_ear.x) * 0.4:
            emotion = "pls.png"
        
        # 2. Руки вверх -> beg.png или cinema.png
        elif l_wrist.y < nose.y - 0.1 and r_wrist.y < nose.y - 0.1:
            if abs(l_wrist.x - r_wrist.x) < 0.15:
                emotion = "beg.png"
            else:
                emotion = "cinema.png"
        
        # 3. Пистолет (рука вперед) -> gun.png
        elif (l_wrist.z < l_sh.z - 0.6 and l_wrist.presence > 0.5) or \
             (r_wrist.z < r_sh.z - 0.6 and r_wrist.presence > 0.5):
            emotion = "gun.png"
            
        # 4. Рука у головы -> hard.png
        elif l_wrist.y < nose.y and abs(l_wrist.x - nose.x) < 0.2:
            emotion = "hard.png"

    # --- ЛОГИКА ЛИЦА (если нет жестов) ---
    if emotion == "normal.png" and face_res.face_landmarks:
        f = face_res.face_landmarks[0]
        mouth_open = abs(f[13].y - f[14].y)
        
        # Сигарета/ручка во рту -> cool.png
        if 0.018 < mouth_open < 0.045:
            emotion = "cool.png"
        # Рот открыт широко -> shock.png
        elif mouth_open >= 0.045:
            emotion = "shock.png"
        # Грустный смайлик губами -> sad.png
        elif f[61].y > f[13].y + 0.008 and f[291].y > f[13].y + 0.008:
            emotion = "sad.png"

    # Склеиваем видео и енота
    raccoon_img = get_raccoon(emotion, h)
    combined = np.hstack((img, raccoon_img))
    
    return av.VideoFrame.from_ndarray(combined, format="bgr24")

# 4. КОНФИГУРАЦИЯ СЕРВЕРОВ (ДЛЯ ПРОХОДА ЧЕРЕЗ NAT/БРАНДМАУЭРЫ)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

# 5. ЗАПУСК ВЕБ-ИНТЕРФЕЙСА
webrtc_streamer(
    key="raccoon-ai-stream",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
