import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from PIL import ImageFont, ImageDraw, Image
from model import SignLSTM  # 기존 모델 불러오기

# 한글 폰트 설정 (macOS 내장 AppleGothic 폰트 사용)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(FONT_PATH, 30)

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------------------------------------------------
# 1. MediaPipe 랜드마크 -> OpenPose 67 포맷 매핑
# ----------------------------------------------------
def extract_keypoints(results):
    """
    MediaPipe Holistic 결과를 OpenPose(BODY_25 + 양손) 포맷(67x3)으로 변환합니다.
    """
    # 67개의 키포인트, (x, y, z) 3차원으로 초기화 (기본값 0)
    keypoints = np.zeros((67, 3), dtype=np.float32)
    
    # 1. Body 25 포맷 매핑
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        
        def set_kp(idx, pose_idx):
            keypoints[idx] = [pose[pose_idx].x, pose[pose_idx].y, pose[pose_idx].z]
            
        def set_mid_kp(idx, pose_idx1, pose_idx2):
            x = (pose[pose_idx1].x + pose[pose_idx2].x) / 2
            y = (pose[pose_idx1].y + pose[pose_idx2].y) / 2
            z = (pose[pose_idx1].z + pose[pose_idx2].z) / 2
            keypoints[idx] = [x, y, z]

        # BODY_25 매핑 (OpenPose 기준)
        set_kp(0, 0)   # Nose
        set_mid_kp(1, 11, 12) # Neck (어깨 중심)
        set_kp(2, 12)  # RShoulder
        set_kp(3, 14)  # RElbow
        set_kp(4, 16)  # RWrist
        set_kp(5, 11)  # LShoulder
        set_kp(6, 13)  # LElbow
        set_kp(7, 15)  # LWrist
        set_mid_kp(8, 23, 24) # MidHip (골반 중심)
        set_kp(9, 24)  # RHip
        set_kp(10, 26) # RKnee
        set_kp(11, 28) # RAnkle
        set_kp(12, 23) # LHip
        set_kp(13, 25) # LKnee
        set_kp(14, 27) # LAnkle
        set_kp(15, 5)  # REye
        set_kp(16, 2)  # LEye
        set_kp(17, 8)  # REar
        set_kp(18, 7)  # LEar
        set_kp(19, 29) # LBigToe
        set_kp(20, 31) # LSmallToe
        set_kp(21, 29) # LHeel (MediaPipe에는 Heel 구분이 명확치 않아 Toe 사용)
        set_kp(22, 30) # RBigToe
        set_kp(23, 32) # RSmallToe
        set_kp(24, 30) # RHeel

    # 2. 왼손 21 포맷 매핑
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            keypoints[25 + i] = [lm.x, lm.y, lm.z]
            
    # 3. 오른손 21 포맷 매핑
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            keypoints[46 + i] = [lm.x, lm.y, lm.z]
            
    return keypoints

def normalize_keypoints(kp_data):
    """
    학습 코드와 동일하게 프레임 단위로 Min-Max 스케일링 수행
    """
    kp_flat = kp_data.reshape(-1)
    valid_mask = kp_flat != 0
    if np.any(valid_mask):
        min_val = np.min(kp_flat[valid_mask])
        max_val = np.max(kp_flat[valid_mask])
        if max_val - min_val > 0:
            kp_flat[valid_mask] = (kp_flat[valid_mask] - min_val) / (max_val - min_val)
    return kp_flat

# ----------------------------------------------------
# 2. 모델 로드 및 설정
# ----------------------------------------------------
# 라벨 맵 준비 (학습 시와 동일한 순서 유지)
df = pd.read_csv("dataset_info.csv")
unique_labels = df['label'].unique()
idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
num_classes = len(unique_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✅ 사용 장치: {device}")

# 모델 생성 및 가중치 불러오기 (학습 시와 동일한 하이퍼파라미터)
model = SignLSTM(input_size=201, hidden_size=256, num_layers=3, num_classes=num_classes, dropout=0.3).to(device)
model.load_state_dict(torch.load("sign_lstm_model.pth", map_location=device))
model.eval()

# ----------------------------------------------------
# 3. 실시간 웹캠 테스트
# ----------------------------------------------------
# 프레임 버퍼 (시퀀스 길이). 수어 단어 하나의 대략적인 길이에 맞춰 조절 (예: 30 프레임)
SEQUENCE_LENGTH = 30
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# 카메라 인덱스: 0=아이폰(Continuity Camera), 1=내장 웹캠
# 내장 웹캠이 안 잡히면 숫자를 0 또는 2로 바꿔보세요
cap = cv2.VideoCapture(1)
current_word = "Waiting..."
confidence = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        # MediaPipe 처리를 위해 BGR -> RGB 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 랜드마크 추출
        results = holistic.process(image)
        
        # 화면 출력을 위해 다시 RGB -> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 화면에 손 랜드마크만 그리기
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # 1. 키포인트 추출 및 정규화
        kp = extract_keypoints(results)
        kp_flat = normalize_keypoints(kp)
        
        # 2. 버퍼에 추가
        frame_buffer.append(kp_flat)
        
        # 3. 버퍼가 꽉 찼을 때 모델 추론
        if len(frame_buffer) == SEQUENCE_LENGTH:
            # 텐서 변환: (1, seq_len, input_size)
            input_data = np.array(frame_buffer)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                # Softmax로 확률 계산
                probs = torch.softmax(output, dim=1)
                conf, predicted = torch.max(probs, 1)
                
                # 특정 신뢰도(Confidence) 이상일 때만 단어 갱신
                if conf.item() > 0.5:
                    current_word = idx_to_label[predicted.item()]
                    confidence = conf.item()
                
        # 화면에 결과 출력 (PIL을 사용하여 한글 렌더링)
        cv2.rectangle(image, (0,0), (640, 50), (245, 117, 16), -1)
        
        display_text = f"예측: {current_word} ({confidence*100:.1f}%)"
        
        # OpenCV 이미지 -> PIL 이미지로 변환하여 한글 텍스트 그리기
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 8), display_text, font=font, fill=(255, 255, 255))
        image = np.array(img_pil)
        
        cv2.imshow('Sign Language Translation', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
