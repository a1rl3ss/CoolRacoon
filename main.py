import cv2
import mediapipe as mp
import numpy as np
import os
import time
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Загрузка моделей MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

def get_raccoon(name, h):
    full_path = os.path.join(os.getcwd(), name)
    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        if img is not None: return cv2.resize(img, (h, h))
    return np.zeros((h, h, 3), dtype=np.uint8)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb_img)
        face_res = face_mesh.process(rgb_img)

        emotion = "normal.png"

        # ТВОЯ ЛОГИКА ИЗ ПИТОНА
        if pose_res.pose_landmarks:
            p = pose_res.pose_landmarks.landmark
            nose = p[0]
            l_ear, r_ear = p[7], p[8]
            l_wrist, r_wrist = p[15], p[16]
            l_sh, r_sh = p[11], p[12]

            # Проверка: видит ли нейронка руки вообще?
            hands_visible = l_wrist.visibility > 0.5 or r_wrist.visibility > 0.5

            # 1. Поворот головы (pls)
            if abs(nose.x - l_ear.x) < abs(nose.x - r_ear.x) * 0.4:
                emotion = "pls.png"
            
            # 2. Руки выше носа (beg/cinema)
            elif l_wrist.y < nose.y - 0.1 and r_wrist.y < nose.y - 0.1:
                emotion = "beg.png" if abs(l_wrist.x - r_wrist.x) < 0.15 else "cinema.png"
            
            # 3. Рука у лица (hard) - добавил приоритет выше пушки
            elif l_wrist.visibility > 0.5 and l_wrist.y < nose.y and abs(l_wrist.x - nose.x) < 0.2:
                emotion = "hard.png"

            # 4. Пистолет (Z-координата) - только если рука РЕАЛЬНО видна
            elif (l_wrist.visibility > 0.5 and l_wrist.z < l_sh.z - 0.7) or \
                 (r_wrist.visibility > 0.5 and r_wrist.z < r_sh.z - 0.7):
                emotion = "gun.png"

        # Лицо (если нет жестов)
        if emotion == "normal.png" and face_res.multi_face_landmarks:
            f = face_res.multi_face_landmarks[0].landmark
            mouth_open = abs(f[13].y - f[14].y)
            if 0.018 < mouth_open < 0.045:
                emotion = "cool.png"
            elif mouth_open >= 0.045:
                emotion = "shock.png"

        raccoon = get_raccoon(emotion, h)
        result = np.hstack((img, raccoon))
        
        return av.VideoFrame.from_ndarray(result, format="bgr24")

st.title("🦝 Raccoon System")

webrtc_streamer(
    key="raccoon",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
