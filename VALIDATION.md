# Validation

이 프로젝트의 최소 검증 계약.

## 기본 검증 명령

```bash
# 1. 모델 로드 테스트 (핵심 — 항상 통과해야 함)
cd model && python -c "
from realtime_inference import load_model
fn, labels, mtype = load_model('auto')
print('OK:', mtype, len(labels), 'classes')
"

# 2. 구조 정합성 체크
python tools/healthcheck.py
```

## 경로별 검증 라우팅

| 변경 경로 | 실행 명령 | 함께 확인할 문서 |
|-----------|-----------|-----------------|
| `model/realtime_inference.py` | 모델 로드 테스트 | `CLAUDE.md` → 아키텍처 주의사항 |
| `model/ksl_tcn_model.py` | 모델 로드 테스트 + TemporalBlock 키 확인 | `CLAUDE.md` → TemporalBlock 섹션 |
| `model/*.pt` / `model/*.pkl` | 모델 로드 테스트 | `CLAUDE.md` → 모델 우선순위 |
| `realtime_inference.py` (루트) | 런처 실행 테스트 | `model/realtime_inference.py` 와 sync 여부 |
| `tools/healthcheck.py` | `python tools/healthcheck.py` | `VALIDATION.md` |

## 중단 조건

아래 중 하나라도 실패하면 작업을 중단하고 알린다:

- `load_model('auto')` 에서 모델 로드 실패
- `TemporalBlock` state_dict 키 mismatch (`net.0` vs `conv1` 구조 충돌)
- 특징 추출 함수(`extract_improved_features_batch`)가 학습/추론 간 불일치

## 검증 보고 형식

```
Scope:    변경한 파일 목록
Commands: 실행한 명령과 결과
Warnings: 잠재적 위험 (있을 경우만)
```
