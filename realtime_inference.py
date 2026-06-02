"""루트 편의 런처 — 실제 스크립트는 model/realtime_inference.py"""
import os, sys

os.execv(sys.executable, [
    sys.executable,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "realtime_inference.py"),
] + sys.argv[1:])
