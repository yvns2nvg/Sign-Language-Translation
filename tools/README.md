# tools/

프로젝트 구조 정합성을 검증하는 helper 스크립트 폴더.
helper는 정책 source가 아니라 문서화된 정책을 검증하는 verifier다.

## 스크립트

| 파일 | 역할 |
|------|------|
| `healthcheck.py` | 필수 파일 존재, 중복 가중치, 런처 구조 확인 |

## 실행

```bash
python tools/healthcheck.py
```

## 검사 항목

1. `model/` 필수 파일 존재 여부 (가중치 + 스크립트)
2. 루트에 중복 가중치 파일 없는지
3. 루트 `realtime_inference.py` 가 런처인지 (전체 스크립트로 덮어써지지 않았는지)
