---
name: sync-model
description: model/realtime_inference.py를 루트 런처와 GitHub에 동기화한다
---

# sync-model

`model/realtime_inference.py`(canonical)를 수정한 뒤 루트 사본과 GitHub를 동기화할 때 사용한다.

## Use When

- `model/realtime_inference.py`를 수정한 뒤 커밋 + 푸시하려 할 때
- 루트 `realtime_inference.py`(런처)와 model/ 버전이 달라졌을 때

## Do Not Use

- 루트 `realtime_inference.py`만 수정했을 때 (항상 model/ 이 canonical)
- 모델 가중치(.pt/.pkl)만 바꿨을 때

## Procedure

1. 변경 내용 확인
   ```bash
   git diff model/realtime_inference.py
   ```
2. 모델 로드 테스트
   ```bash
   cd model && python -c "from realtime_inference import load_model; fn,l,t=load_model('auto'); print('OK:',t,len(l),'classes')"
   ```
3. 커밋 & 푸시
   ```bash
   git add model/realtime_inference.py
   git commit -m "Update realtime_inference.py: <변경 내용 한 줄>"
   git push origin main
   ```

## Output format

```
Scope:    model/realtime_inference.py
Commands: load_model test → OK / FAIL
Result:   pushed to main / blocked (reason)
Warnings: (있으면)
```

## Failure conditions

- 모델 로드 테스트 실패 → 커밋하지 않고 오류 원인 보고
- TemporalBlock 키 mismatch 감지 → CLAUDE.md 아키텍처 섹션 재확인
