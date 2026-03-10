import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

# 1. ЗАГРУЗКА МОДЕЛЕЙ
MODELS = {
    "face": ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "pose": ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
}

for name, (path, url) in MODELS.items():
    if not os.path.exists(path):
        print(f"Загружаю модель {name}...")
        urllib.request.urlretrieve(url, path)

# Настройка MediaPipe
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELS["face"][0]), 
    running_mode=VisionRunningMode.VIDEO)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELS["pose"][0]), 
    running_mode=VisionRunningMode.VIDEO)

def get_raccoon(name, h):
    full_path = os.path.join(os.getcwd(), name)
    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        if img is not None: return cv2.resize(img, (h, h))
    blank = np.zeros((h, h, 3), dtype=np.uint8)
    cv2.putText(blank, f"MISSING {name}", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return blank

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_raccoon = None

with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_det, \
     mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_det:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Обработка каждого 2-го кадра для FPS
        if frame_count % 2 == 0:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)

            face_res = face_det.detect_for_video(mp_img, timestamp)
            pose_res = pose_det.detect_for_video(mp_img, timestamp)

            emotion = "normal.png"

            if pose_res.pose_landmarks:
                p = pose_res.pose_landmarks[0]
                # Берем точки с проверкой их "уверенности"
                l_wrist, r_wrist = p[15], p[16]
                nose, l_ear, r_ear = p[0], p[7], p[8]
                l_shoulder, r_shoulder = p[11], p[12]

                # 1. ПОВОРОТ ГОЛОВЫ ВЛЕВО
                if abs(nose.x - l_ear.x) < abs(nose.x - r_ear.x) * 0.4:
                    emotion = "pls.png"
                
                # 2. РУКИ ВВЕРХ (Cinema / Beg)
                elif l_wrist.y < nose.y - 0.1 and r_wrist.y < nose.y - 0.1:
                    emotion = "beg.png" if abs(l_wrist.x - r_wrist.x) < 0.15 else "cinema.png"
                
                # 3. ПИСТОЛЕТ (Gun) - Исправленная логика с проверкой видимости
                else:
                    # Проверяем, что рука вообще в кадре (presence > 0.5) и выше пояса
                    l_visible = l_wrist.presence > 0.5 and l_wrist.y < l_shoulder.y + 0.3
                    r_visible = r_wrist.presence > 0.5 and r_wrist.y < r_shoulder.y + 0.3
                    
                    # Только если ОДНА рука вытянута вперед (z < -0.7)
                    if l_visible and l_wrist.z < l_shoulder.z - 0.7 and not (r_wrist.y < nose.y):
                        emotion = "gun.png"
                    elif r_visible and r_wrist.z < r_shoulder.z - 0.7 and not (l_wrist.y < nose.y):
                        emotion = "gun.png"
                    
                    # 4. HARD
                    elif l_wrist.y < nose.y and abs(l_wrist.x - nose.x) < 0.2:
                        emotion = "hard.png"

            # 5. ЛИЦО (Shock, Sad, Cool)
            if emotion == "normal.png" and face_res.face_landmarks:
                f = face_res.face_landmarks[0]
                mouth_open = abs(f[13].y - f[14].y)
                
                if 0.018 < mouth_open < 0.045:
                    emotion = "cool.png"
                elif mouth_open >= 0.045:
                    emotion = "shock.png"
                elif f[61].y > f[13].y + 0.008 and f[291].y > f[13].y + 0.008:
                    emotion = "sad.png"

            last_raccoon = get_raccoon(emotion, h)

        # Вывод
        current_raccoon = last_raccoon if last_raccoon is not None else get_raccoon("normal.png", h)
        res = np.hstack((frame, current_raccoon))
        cv2.imshow('RACCOON AI SYSTEM', res)
        
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()