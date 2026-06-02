"""프로젝트 구조 정합성 체크."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")

REQUIRED_MODEL_FILES = [
    "ksl_tcn_improved.pt",
    "ksl_tcn_improved_meta.pkl",
    "ksl_lstm_model.pt",
    "ksl_lstm_meta.pkl",
    "ksl_dtw_knn_model.pkl",
    "realtime_inference.py",
    "ksl_tcn_model.py",
    "train_lstm.py",
]

REQUIRED_ROOT_FILES = [
    "CLAUDE.md",
    "VALIDATION.md",
    "realtime_inference.py",
    ".gitignore",
]

ROOT_MODEL_DUPLICATES = [
    "ksl_tcn_improved.pt",
    "ksl_tcn_improved_meta.pkl",
    "ksl_lstm_model.pt",
    "ksl_lstm_meta.pkl",
    "ksl_dtw_knn_model.pkl",
]

errors = []
warnings = []

# 1. model/ 필수 파일 존재 여부
for f in REQUIRED_MODEL_FILES:
    path = os.path.join(MODEL_DIR, f)
    if not os.path.exists(path):
        errors.append(f"[MISSING] model/{f}")

# 2. 루트 필수 파일 존재 여부
for f in REQUIRED_ROOT_FILES:
    path = os.path.join(ROOT, f)
    if not os.path.exists(path):
        errors.append(f"[MISSING] {f}")

# 3. 루트에 중복 가중치 파일 없어야 함
for f in ROOT_MODEL_DUPLICATES:
    path = os.path.join(ROOT, f)
    if os.path.exists(path):
        warnings.append(f"[DUPLICATE] 루트에 {f} 존재 — model/ 버전이 canonical")

# 4. 루트 realtime_inference.py 가 런처인지 확인
launcher_path = os.path.join(ROOT, "realtime_inference.py")
if os.path.exists(launcher_path):
    content = open(launcher_path).read()
    if "os.execv" not in content and len(content) > 5000:
        warnings.append("[SYNC] 루트 realtime_inference.py 가 런처가 아닌 전체 스크립트 — model/ 와 동기화 확인 필요")

# 결과 출력
print("=" * 50)
print("healthcheck")
print("=" * 50)
if errors:
    for e in errors:
        print(e)
if warnings:
    for w in warnings:
        print(w)
if not errors and not warnings:
    print("OK — 모든 검사 통과")
print("=" * 50)

sys.exit(1 if errors else 0)
