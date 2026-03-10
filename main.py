import streamlit as st
import os

# Сначала импортируем CV2, потом MediaPipe (важен порядок на Linux)
import cv2
try:
    import mediapipe as mp
except ImportError:
    st.error("MediaPipe не установлен. Проверь requirements.txt")

import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Проверка наличия атрибута solutions
if not hasattr(mp, 'solutions'):
    st.error("Ошибка инициализации MediaPipe. Попробуй нажать 'Reboot App'")

# Настройка моделей
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

class VideoProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face.FaceMesh(
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
        return np.zeros((h, h, 3), dtype=np.uint8)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb_img)
        face_res = self.face_mesh.process(rgb_img)

        emotion = "normal.png"

        if pose_res.pose_landmarks:
            p = pose_res.pose_landmarks.landmark
            nose = p[0]
            l_ear, r_ear = p[7], p[8]
            l_wrist, r_wrist = p[15], p[16]
            l_sh, r_sh = p[11], p[12]

            l_vis = l_wrist.visibility > 0.5
            r_vis = r_wrist.visibility > 0.5

            # 1. Поворот головы (pls)
            if abs(nose.x - l_ear.x) < abs(nose.x - r_ear.x) * 0.4:
                emotion = "pls.png"
            
            # 2. Руки выше носа (beg/cinema)
            elif l_vis and r_vis and l_wrist.y < nose.y - 0.1 and r_wrist.y < nose.y - 0.1:
                emotion = "beg.png" if abs(l_wrist.x - r_wrist.x) < 0.15 else "cinema.png"
            
            # 3. Рука у лица (hard)
            elif l_vis and l_wrist.y < nose.y and abs(l_wrist.x - nose.x) < 0.2:
                emotion = "hard.png"

            # 4. Пистолет (gun)
            elif (l_vis and l_wrist.z < l_sh.z - 0.7) or (r_vis and r_wrist.z < r_sh.z - 0.7):
                emotion = "gun.png"

        if emotion == "normal.png" and face_res.multi_face_landmarks:
            f = face_res.multi_face_landmarks[0].landmark
            mouth_open = abs(f[13].y - f[14].y)
            if 0.018 < mouth_open < 0.045:
                emotion = "cool.png"
            elif mouth_open >= 0.045:
                emotion = "shock.png"

        raccoon = self.get_raccoon(emotion, h)
        combined = np.hstack((img, raccoon))
        
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

st.title("🦝 Raccoon System")

webrtc_streamer(
    key="raccoon-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
