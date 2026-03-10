import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# 1. Настройка страницы
st.set_page_config(page_title="Raccoon AI Lite", layout="wide")
st.title("🦝 Raccoon AI System")

# Загрузка моделей (если их нет)
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
    return np.zeros((h, h, 3), dtype=np.uint8) + 50

# 2. Облегченный процессор
class RaccoonProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "normal.png"
        
        # Настройка MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        self.face_det = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODELS["face"][0]),
                running_mode=mp.tasks.vision.RunningMode.VIDEO))
        self.pose_det = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODELS["pose"][0]),
                running_mode=mp.tasks.vision.RunningMode.VIDEO))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Обрабатываем только каждый 4-й кадр, чтобы не лагало
        if self.frame_count % 4 == 0:
            img_small = cv2.resize(img, (320, 240)) # Уменьшаем для скорости
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
            ts = int(time.time() * 1000)

            face_res = self.face_det.detect_for_video(mp_img, ts)
            pose_res = self.pose_det.detect_for_video(mp_img, ts)

            new_emotion = "normal.png"
            
            # Быстрая проверка жестов
            if pose_res.pose_landmarks:
                p = pose_res.pose_landmarks[0]
                if p[15].y < p[0].y - 0.1: new_emotion = "beg.png"
                elif p[15].z < -0.5: new_emotion = "gun.png"

            if new_emotion == "normal.png" and face_res.face_landmarks:
                f = face_res.face_landmarks[0]
                m_open = abs(f[13].y - f[14].y)
                if 0.015 < m_open < 0.04: new_emotion = "cool.png"
                elif m_open >= 0.04: new_emotion = "shock.png"
            
            self.last_emotion = new_emotion

        # Отрисовка енота (всегда)
        h, w, _ = img.shape
        raccoon = get_raccoon(self.last_emotion, h)
        combined = np.hstack((cv2.flip(img, 1), raccoon))
        
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

# 3. Конфигурация и запуск
webrtc_streamer(
    key="raccoon-lite",
    video_processor_factory=RaccoonProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    async_processing=True,
)
