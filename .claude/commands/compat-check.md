---
name: compat-check
description: 모델 가중치와 realtime_inference.py 아키텍처 호환성을 확인한다
---

# compat-check

새 모델 파일을 추가하거나 아키텍처를 수정한 뒤 호환성을 검증할 때 사용한다.

## Use When

- 새 `.pt` 모델 파일을 GitHub에서 가져온 뒤
- `TemporalBlock` 등 아키텍처 코드를 수정한 뒤
- state_dict 로드 오류가 발생했을 때

## Procedure

1. state_dict 키 확인
   ```bash
   python -c "
   import torch
   ckpt = torch.load('model/ksl_tcn_improved.pt', map_location='cpu', weights_only=False)
   keys = list(ckpt['model_state'].keys())[:10]
   print('config:', ckpt['config'])
   print('first keys:', keys)
   "
   ```

2. 모델 로드 테스트
   ```bash
   cd model && python -c "
   from realtime_inference import load_model
   fn, labels, mtype = load_model('auto')
   print('OK:', mtype, len(labels), 'classes')
   "
   ```

3. 키 구조 판단 기준
   - `tcn.0.conv1` 형태 → TemporalBlock을 named attribute 방식으로 정의해야 함
   - `tcn.0.net.0` 형태 → `nn.Sequential` 방식

## Output format

```
Scope:    model/ksl_tcn_improved.pt + model/realtime_inference.py
Keys:     tcn.0.conv1 / tcn.0.net.0 (확인된 구조)
Result:   OK / MISMATCH (이유)
Warnings: (있으면)
```

## Failure conditions

- `load_state_dict` 에서 unexpected/missing keys → TemporalBlock 구조 불일치
- `config['model']` 키 없음 → 구형 LSTM 모델, legacy 경로로 처리
