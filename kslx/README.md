# kslx — 누수 없는 재설계 파이프라인

전체 실험 결과와 수치는 [`RESULTS.md`](RESULTS.md) 에 정리돼 있다.

`model/` (기존 TCN, 3000클래스 val 99.6%)의 val 분할이 데이터 누수라는 게
데이터만으로 증명됐다: 같은 take(수어사×단어)의 5개 카메라 앵글(F/D/L/R/U)
파일에 든 3D 키포인트가 비트 단위로 동일하다 — 멀티뷰 삼각측량 결과 하나를
5개 파일에 복사해 넣은 것이다. 랜덤 분할에서는 val 샘플의 다른-앵글 사본이
train 에 그대로 들어가 있으니 정확도가 부풀려진다.

`kslx/` 는 `model/` 을 건드리지 않고 병렬로 만든 새 패키지다. 원칙:

- **3D 키포인트를 쓰지 않는다.** `*_keypoints_2d` 만 쓴다 (`layout.py` 상단 주석).
- **분할 프로토콜을 이름으로 명시한다** (`splits.py`): `random`(기존 방식, 참고용),
  `take_out`, `angle_out`, `signer_out`(실사용 성능에 가장 가까움). 조건이 안 맞으면
  (예: 수어사 1명뿐이라 signer_out 불가) 조용히 이상한 숫자를 내는 대신
  `SplitError` 를 던진다.
- **모델 없이 데이터만으로 누수를 정량화한다** (`leak_report.py`).

## 빠른 시작

```bash
pip install torch numpy mediapipe opencv-python pillow

# 1) AI Hub WORD 키포인트 JSON → npz 캐시
python -m kslx.data.build_dataset --root <004.수어영상 경로> \
    --out data/kslx/word_271.npz --signers 1-16 --words 1-271 --workers 14

# 2) 데이터만으로 누수 증명 (모델 불필요)
python -m kslx.leak_report --root <004.수어영상 경로> --signers 1-16 --words 1-271

# 3) 4개 분할 프로토콜 학습 + 비교
python -m kslx.train --data data/kslx/word_271.npz \
    --protocol random take_out angle_out signer_out --epochs 100 --tag base

# 4) 강건성 평가 (yaw/회전/손 결측/스케일/노이즈/속도) — 모델 선택은 이걸로
python -m kslx.eval_robust --data data/kslx/word_271.npz --ckpt runs/signer_out_base.pt
```

전체 결과(4개 프로토콜, 증강 전/후 강건성 비교)는 [`RESULTS.md`](RESULTS.md) 참고.
**최종 추천 체크포인트는 `runs/signer_out_aug.pt`** (signer_out top-1 97.8%,
모든 강건성 변형에서 1%p 이내 열화).

## 웹캠 실시간 검증 (`realtime.py`)

학습 데이터는 AI Hub 5카메라 원본(OpenPose 포맷: pose25+hand21×2+face70)이고
웹캠은 MediaPipe(pose33+hand21×2+face478)를 쓰므로 인덱스 체계가 다르다.
`adapters/mediapipe_adapter.py` 가 이 변환을 맡는다 — pose/hand 는 관절 대응이
명확해 신뢰도가 높고, **얼굴 34점은 MediaPipe 공식 얼굴 영역 상수에서 균등
서브샘플링한 근사치**라 학습 시 얼굴 좌표와 완전히 일치하진 않는다 (미검증,
성능이 이상하면 `--no-face` 로 먼저 비교할 것).

체크포인트(`runs/signer_out_aug.pt`)와 MediaPipe 모델 번들
(`kslx/mp_models/holistic_landmarker.task`)은 **이 저장소에 이미 포함돼 있다**
— 클론 후 별도 다운로드 없이 바로 실행 가능하다.

```bash
pip install torch opencv-python mediapipe pillow numpy kiwipiepy

python -m kslx.realtime --ckpt runs/signer_out_aug.pt
```

기본은 `--mode auto`: 손 움직임 에너지로 "단어 끝 → 다음 단어 시작"을 자동
감지해서(`kslx/stream.py`) 매번 SPACE를 누를 필요 없이 이어서 인식한다.
`--mode manual` 로 기존 SPACE 방식도 가능하고, 실행 중 `M` 키로 전환된다.
조작: `SPACE`(manual: 녹화 시작/중지, auto: 현재 구간 강제 종료) | `M` 모드 전환 |
`ENTER` 지금까지 모은 단어로 문장 즉시 완성 | `C` 문장 버퍼 비움 |
`R` 취소 | `Q`/`ESC` 종료.

★ 자동 분할은 정식 학습된 경계 검출기가 아니라 휴리스틱이다 — 카메라가 손을
순간적으로 놓치면 단어가 중간에 잘못 끊길 수 있다. 화면에 뜨는 `energy` 값을
보면서 `--start-energy`/`--end-energy`/`--end-hold` 를 조정하거나, 안 될 때는
`--mode manual` 로 돌아갈 것 (`kslx/stream.py` 상단 주석 참고).

### 단어 → 문장 (`sentence.py`)

수어는 한국어를 그대로 옮긴 게 아니라 조사/어미가 거의 없는 독립 문법을
쓰는 언어라서(자세한 근거는 `kslx/sentence.py` 상단 주석), 인식된 단어를
그냥 나열하면 "나 학교 가다"처럼 어색하다. 단어가 하나씩 인식될 때마다
모았다가 한동안(`--sentence-end-hold`, 기본 ~1.5초) 손이 멈추면
`kiwipiepy`로 품사를 태깅해 조사/어미를 붙인 문장으로 합친다 —
"나는 학교에 가요"처럼. 완전 오프라인, 규칙 기반(LLM/API 없음)이라 품질에
한계가 있다 — 특히 연속된 명사 두 개(예: "학교 친구 만나다")는 복합명사인지
별개 성분인지 구분을 못 해 어색해질 수 있다. 실제 예시와 한계는
`RESULTS.md`§9 참고.

다른 체크포인트로 직접 학습한 경우가 아니면 `runs/*.pt` 는 커밋하지 않는다
(`.gitignore` — `signer_out_aug.pt` 하나만 예외로 포함돼 있다).

## 파일 지도

```
kslx/
├── layout.py                  키포인트 레이아웃 (89점: pose13 + 손21×2 + 얼굴34)
│                              ★ 3D 를 쓰면 안 되는 이유가 상단 주석에 있음
├── normalize.py               정규화(목 원점 + 어깨너비 스케일) · 리샘플 · 속도특징
│                              ★ 어깨 동시 미검출 시 클립 중앙값으로 fallback
│                              (고정 최솟값으로 나누면 좌표가 폭발하는 버그를 수정함)
├── splits.py                  분할 프로토콜 + 사후 감사(leak 있으면 에러)
├── augment.py                  학습 배치 증강 (미러/회전/전단/스케일/노이즈/손 결측)
│                              ★ 이거 없으면 손 결측 시 정확도가 14%로 붕괴함 (RESULTS.md §5-6)
├── train.py                   학습 + 프로토콜 비교 (--aug 로 증강 on/off)
├── leak_report.py              데이터만으로 누수 정량화 (원본 JSON 스캔)
├── leak_report_npz.py          동일 주장을 별도로 만들어진 npz 캐시로 독립 재확인
├── eval_robust.py              강건성 평가 — 모델 선택은 이걸로
├── stream.py                    에너지 기반 자동 단어 경계 감지 (휴리스틱, 미검증)
├── sentence.py                  단어 나열 -> 문장 (kiwipiepy 품사 태깅 + 조사/어미 규칙)
├── realtime.py                 웹캠 데모 (Windows, 기본 auto/M으로 manual 전환)
│                              runs/signer_out_aug.pt + mp_models/*.task 이미 포함
├── models/conv_transformer.py  1D DepthwiseConv + Transformer, ~1.1M params
├── adapters/mediapipe_adapter.py  MediaPipe → 89점 레이아웃 변환 (얼굴은 근사)
└── data/
    ├── aihub.py                JSON 스캐너 + 로더 + 형태소(수어구간) 파서
    ├── build_dataset.py        npz 캐시 빌더 (멀티프로세스)
    └── word_labels.json        WORD#### → 한글 (3000단어)
```
