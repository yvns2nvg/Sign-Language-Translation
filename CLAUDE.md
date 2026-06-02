# CLAUDE.md

이 파일은 Claude Code가 이 프로젝트에서 작업할 때 따르는 규칙이다.

## 프로젝트 개요

KSL(한국 수어) 실시간 인식 시스템.
MediaPipe로 손/포즈 키포인트를 추출하고 TCN 모델로 분류한다.

## 폴더 구조

```
Sign-Language-Translation/
├── model/                         # 모델 코드 + 가중치 (source-of-truth)
│   ├── realtime_inference.py      # 실시간 추론 스크립트 (canonical)
│   ├── ksl_tcn_model.py           # TCN 학습 스크립트
│   ├── train_lstm.py              # LSTM 학습 스크립트
│   ├── ksl_tcn_improved.pt        # TCN 모델 가중치 (val_top1 99.6%, 현재 best)
│   ├── ksl_tcn_improved_meta.pkl  # TCN 메타 (label_names 3000클래스)
│   ├── ksl_lstm_model.pt          # LSTM 모델 가중치 (val_acc 82.7%)
│   ├── ksl_lstm_meta.pkl          # LSTM 메타
│   └── ksl_dtw_knn_model.pkl      # DTW+KNN 모델 (폴백용)
├── android_app/                   # Android 클라이언트
├── CLAUDE.md                      # 이 파일
└── realtime_inference.py          # 루트 실행 편의용 사본 (model/ 가 canonical)
```

## 모델 우선순위

자동 감지 순서: **TCN > LSTM > DTW+KNN**

| 모델 | 파일 | val 정확도 | 특징 |
|------|------|-----------|------|
| TCN (현재 best) | `ksl_tcn_improved.pt` | top1 99.6% | input_dim 206, 3000클래스 |
| LSTM | `ksl_lstm_model.pt` | 82.7% | input_dim 40, 3000클래스 |
| DTW+KNN | `ksl_dtw_knn_model.pkl` | - | 폴백, C라이브러리 불필요 |

## 실행 방법

```bash
conda activate sign_lang_env

# TCN 자동 선택 (ksl_tcn_improved.pt 있으면 자동 사용)
python model/realtime_inference.py

# 특정 모델 지정
python model/realtime_inference.py --model model/ksl_lstm_model.pt
python model/realtime_inference.py --model model/ksl_dtw_knn_model.pkl
```

조작: `SPACE` 녹화 시작/중지 → 예측 | `R` 취소 | `Q/ESC` 종료

## 핵심 기술 사항

### 특징 추출 (extract_improved_features_batch)
- input: (N, T, 67, 3) — 왼손21 + 오른손21 + 포즈25 키포인트
- output: (N, T, 206) — 관절각도40 + 손끝좌표30 + 손목관계13 + 손크기6 + 몸통참조14 + delta103
- **학습과 추론에서 완전히 동일한 함수를 써야 한다**

### TemporalBlock 아키텍처
state_dict 키가 `conv1/bn1/conv2/bn2` 플랫 구조이므로 `nn.Sequential` 로 묶으면 로드 실패한다.
반드시 named attribute 방식으로 정의할 것:
```python
self.conv1 = nn.Conv1d(...)
self.bn1   = nn.BatchNorm1d(...)
self.conv2 = nn.Conv1d(...)
self.bn2   = nn.BatchNorm1d(...)
```

### 한글 표시
OpenCV는 한글을 렌더링하지 못한다. PIL로 우회한다:
```python
from PIL import ImageFont, ImageDraw, Image
font = ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', size)
```

### 카메라 인덱스 (macOS)
- Continuity Camera(iPhone)가 연결되면 index 0이 iPhone이 될 수 있다.
- `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)` 으로 맥북 내장 카메라를 강제 지정한다.

## 환경

```bash
conda activate sign_lang_env  # Python 3.10, /opt/miniconda3/envs/sign_lang_env
pip install opencv-python mediapipe torch joblib dtaidistance Pillow
```

## 작업 규칙

- `model/realtime_inference.py` 가 canonical이다. 루트 사본은 편의용이며 변경 후 동기화한다.
- 모델 가중치 파일(`.pt`, `.pkl`)을 수정하지 않는다. 학습은 별도 스크립트로만 수행한다.
- 특징 추출 함수를 변경하면 반드시 학습과 추론 양쪽 모두 동일하게 맞춘다.
- 구형 파일(`sign_lstm_best.pth`, `sign_lstm_model.pth`, `model.py`)은 루트에 잔류 중이며 현재 사용하지 않는다.

## 검증 진입점

```bash
# 모델 로드 테스트
python -c "
import os; os.chdir('model')
from realtime_inference import load_model
fn, labels, mtype = load_model('auto')
print('OK:', mtype, len(labels), 'classes')
"
```
