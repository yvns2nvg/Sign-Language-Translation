# model/

KSL 수어 인식 모델 코드와 가중치가 있는 폴더. 이 폴더가 source-of-truth다.

## 현재 모델

| 파일 | 종류 | 정확도 | 비고 |
|------|------|--------|------|
| `ksl_tcn_improved.pt` | TCN | val top1 99.6% | **현재 best, 자동 선택됨** |
| `ksl_tcn_improved_meta.pkl` | TCN 메타 | — | 3000클래스 label_names |
| `ksl_lstm_model.pt` | BiLSTM | val 82.7% | TCN 없을 때 폴백 |
| `ksl_lstm_meta.pkl` | LSTM 메타 | — | |
| `ksl_dtw_knn_model.pkl` | DTW+KNN | — | 최종 폴백 |

## 스크립트

| 파일 | 역할 |
|------|------|
| `realtime_inference.py` | 실시간 추론 (canonical) |
| `ksl_tcn_model.py` | TCN 학습 스크립트 |
| `train_lstm.py` | LSTM 학습 스크립트 |
| `test_webcam.py` | 웹캠 연결 테스트 |

## 실행

```bash
conda activate sign_lang_env
python model/realtime_inference.py          # TCN 자동 선택
python model/realtime_inference.py --model model/ksl_lstm_model.pt
```

## 주의

- 가중치 파일(`.pt`, `.pkl`)은 직접 수정하지 않는다. 학습은 스크립트로만.
- `realtime_inference.py` 수정 후에는 `/sync-model` 로 검증 + 커밋.
- `TemporalBlock`은 반드시 named attribute 방식(`conv1/bn1/conv2/bn2`)으로 정의. `nn.Sequential` 사용 시 state_dict 로드 실패.
