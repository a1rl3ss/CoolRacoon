import cv2
import mediapipe as mp
import numpy as np
import os
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Настройка страницы
st.set_page_config(page_title="Raccoon AI", layout="wide")

class VideoProcessor:
    def __init__(self):
        # Инициализируем модели ВНУТРИ процессора, чтобы избежать ошибок импорта
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_raccoon(self, name, h):
        if os.path.exists(name):
            img = cv2.imread(name)
            if img is not None:
                return cv2.resize(img, (h, h))
        # Если картинки нет, создаем пустой квадрат (черный)
        return np.zeros((h, h, 3), dtype=np.uint8)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Конвертация для MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb_img)
        face_res = self.face_mesh.process(rgb_img)

        emotion = "normal.png"

        # --- ТВОЯ ЛОГИКА (ПЕРЕНЕСЕНА ОДИН-В-ОДИН) ---
        if pose_res.pose_landmarks:
            p = pose_res.pose_landmarks.landmark
            nose = p[0]
            l_ear, r_ear = p[7], p[8]
            l_wrist, r_wrist = p[15],
