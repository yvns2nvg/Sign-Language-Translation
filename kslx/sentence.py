"""인식된 단어(gloss) 나열 -> 자연스러운 한국어 문장. 규칙 기반, 오프라인, 무료.

## 왜 규칙 기반인가 (배경)

한국수어(KSL)는 한국어를 그대로 옮긴 게 아니라 독립적인 문법을 가진 언어다.
한국어는 교착어(조사/어미로 문법관계를 표시)인데 KSL은 고립어에 가깝다 —
조사가 거의 없고('부터', '와/과' 정도만 존재), 주격/목적격/관형격 조사가
없다. 기본 어순은 SOV 지만 조사가 없어도 되니 어순이 더 자유롭고,
주제화(topicalization)로 어순이 바뀐다. 의문문 등 문장 종류는 단어가 아니라
비수지 신호(눈썹, 고개 등)로 표시된다 — 그래서 AI Hub SEN 데이터에도
"장애인복지카드"라는 단어 하나짜리 형태소에 "1형태소 의문사 없는 의문형"
같은 태그가 따로 붙어 있다 (단어 자체가 아니라 표정이 문장 종류를 정한다는 뜻).

즉 수어 인식이 뽑아내는 건 "조사·어미 없는 핵심어 나열"이고, 이걸 자연스러운
한국어 문장으로 바꾸는 건 기계번역의 gloss-to-text 문제다. AI Hub SEN 라벨은
형태소(단어) 나열만 있고 대응하는 완성 문장 텍스트가 없어 이 프로젝트
데이터만으로 학습 기반 gloss-to-text 모델을 만들 수는 없다 — 그래서 품사
기반 규칙(조사/어미 삽입)으로 접근한다. LLM 기반 접근보다 품질은 낮지만
인터넷/API 키 없이 완전히 오프라인으로 동작한다.

## 접근

각 단어를 `kiwipiepy`로 품사 태깅해서(우리 어휘는 이미 한국어 사전형이라
`가다`→가(VV)+다(EF), `학교`→학교(NNG) 처럼 잘 분석된다) 명사/용언을
구분하고:

- 문장의 마지막 용언(동사/형용사)을 술어로 삼는다 (KSL도 SOV 성향이라
  이미 대체로 맞는 순서다 — 어순을 임의로 재배열하지 않는다).
- 술어 앞 마지막 명사에 술어 종류에 따라 조사를 붙인다: 이동동사(가다/오다류)
  앞은 '에', 형용사 앞은 '이/가', 그 외 동사 앞은 '을/를'.
  맨 앞 명사는 주제 조사 '은/는'.
- 술어가 아예 없으면(명사만 나열됨 — AI Hub SEN에도 이런 단문이 있다)
  마지막 명사에 서술격조사 '이다'를 붙여 평서문으로 만든다.
- 종결어미는 해요체('-어요'/'-이에요')로 통일 — 조사/어미의 정확한 이형태
  (받침 유무에 따른 은/는, 이/가, 을/를, 이에요/예요 등) 선택은 전부
  `kiwi.join()`이 자동으로 처리한다.

이건 완전한 문법 분석기가 아니라 휴리스틱이다. 동사의 정확한 격틀(어떤
조사를 요구하는지)까지는 보지 않으므로 "을/를" 대신 다른 조사가 자연스러운
동사(예: 만나다 -> 을/를 대신 을 그대로 써도 되지만 '와/과 만나다'가 더
자연스러운 경우 등)에서는 문장이 다소 어색할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# 이동류 동사 — 뒤따르는 명사가 '을/를' 대신 '에'를 받는다 (장소/방향격).
MOVEMENT_VERB_SUFFIXES = (
    "가다", "오다", "다니다", "올라가다", "내려가다", "돌아가다", "돌아오다",
    "나가다", "들어가다", "들어오다", "떠나다", "도착하다", "출발하다",
    "등교하다", "퇴근하다", "출근하다", "여행하다", "이동하다",
)

# VV/VA/VX = 동사/형용사/보조용언 어간. XSV/XSA = 명사 뒤에 붙어 동사/형용사를
# 만드는 접미사("사랑+하다"의 "하"처럼 — kiwi가 흔한 단어는 VA 하나로 합쳐
# 분석하기도 하고("어색하다"->VA), 덜 흔한 조합은 "명사+XSV/XSA"로 쪼개기도
# 한다. 둘 다 잡아야 "OO하다" 계열 동사를 놓치지 않는다).
PREDICATE_TAG_PREFIXES = ("VV", "VA", "VX", "XSV", "XSA")
NOUN_TAGS = ("NNG", "NNP", "NP", "NR")


@lru_cache(maxsize=1)
def _get_kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


@dataclass
class WordInfo:
    gloss: str
    role: str          # "noun" | "predicate" | "adverb" | "other"
    toks: list         # kiwi Token 리스트
    head_tag: str = "NNG"   # role=="noun" 일 때 조사 결합에 쓸 실제 품사(NP 대명사 등)
    forced: bool = False    # role=="predicate" 인데 kiwi 분석에 용언 형태소가 안 잡혀
                            # "다"로 끝나는 것만 보고 강제로 동사 취급한 경우


def _classify(gloss: str) -> WordInfo:
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(gloss)
    tags = [t.tag for t in toks]

    if any(tag.startswith(PREDICATE_TAG_PREFIXES) for tag in tags):
        return WordInfo(gloss=gloss, role="predicate", toks=toks)

    if gloss.endswith("다") and len(gloss) >= 2:
        # kiwi가 다른 뜻(예: "배우다"->배우(NNG,배우/俳優)+이다)으로 오분석해도,
        # 우리 어휘는 AI Hub 사전형 표기라 "다"로 끝나면 사실상 항상 동사/형용사
        # 의도다. 문맥 없는 단어 하나만으로는 kiwi의 의미 중의성 해소보다
        # 이 표기 규칙이 더 믿을 만하다.
        return WordInfo(gloss=gloss, role="predicate", toks=toks, forced=True)

    if any(tag in NOUN_TAGS for tag in tags):
        head_tag = next(t.tag for t in toks if t.tag in NOUN_TAGS)
        return WordInfo(gloss=gloss, role="noun", toks=toks, head_tag=head_tag)

    if any(tag == "MAG" for tag in tags):
        return WordInfo(gloss=gloss, role="adverb", toks=toks)

    return WordInfo(gloss=gloss, role="other", toks=toks)


def _predicate_stem_toks(info: WordInfo) -> list:
    """어미를 뺀 나머지 형태소 — 새 어미를 붙이기 위한 어간.

    어미 태그는 EF(종결)만 있는 게 아니라 EC(연결, 예: "나쁘다"->나쁘/VA+다/EC)
    도 있다 — kiwi 가 문맥 없는 단어 하나를 분석할 때 종결어미 대신 연결어미로
    보는 경우가 있다. EF만 걸러내면 이 "다"가 어간에 남아 "나쁘다어요" 처럼
    깨지므로, E로 시작하는 어미 태그는 전부 제외한다.
    """
    if info.forced:
        return [(info.gloss[:-1], "VV")]
    return [(t.form, t.tag) for t in info.toks if not t.tag.startswith("E")]


def _is_adjective(info: WordInfo) -> bool:
    if info.forced:
        return False  # 강제 처리 케이스는 형용사인지 알 방법이 없어 동사로 취급
    # VA(형용사 어간) 뿐 아니라 XSA(명사+"하다"로 형용사를 만드는 접미사,
    # 예: "피곤하다"->피곤/NNG+하/XSA)도 형용사로 봐야 한다.
    return any(t.tag.startswith(("VA", "XSA")) for t in info.toks)


def compose_sentence(glosses: list[str]) -> str:
    """인식된 gloss 순서 그대로 받아 문장 하나를 만든다. 빈 리스트면 빈 문자열."""
    if not glosses:
        return ""
    kiwi = _get_kiwi()
    infos = [_classify(g) for g in glosses]

    predicate_idx = None
    for i in range(len(infos) - 1, -1, -1):
        if infos[i].role == "predicate":
            predicate_idx = i
            break

    noun_idxs_before_pred = [
        i for i, info in enumerate(infos)
        if info.role == "noun" and (predicate_idx is None or i < predicate_idx)
    ]
    last_noun_before_pred = max(noun_idxs_before_pred) if noun_idxs_before_pred else None

    parts: list[str] = []
    for i, info in enumerate(infos):
        g = info.gloss
        if info.role == "predicate" and i == predicate_idx:
            stem = _predicate_stem_toks(info)
            parts.append(kiwi.join(stem + [("어요", "EF")]))
        elif info.role == "noun":
            tag = info.head_tag
            if predicate_idx is None:
                # 술어 없음: 명사 나열 -> 마지막 명사에 서술격조사로 평서문 마무리
                if i == len(infos) - 1:
                    parts.append(kiwi.join([(g, tag), ("이", "VCP"), ("에요", "EF")]))
                elif i == 0:
                    parts.append(kiwi.join([(g, tag), ("는", "JX")]))
                else:
                    parts.append(g)
            elif i == last_noun_before_pred:
                # 술어 앞 마지막 명사 -> 술어 종류에 맞는 격조사.
                # 명사가 하나뿐이면(i==0 이기도 함) 주제가 아니라 이 격조사를 우선한다 —
                # "음료수는 마셔요"보다 "음료수를 마셔요"가 훨씬 자연스럽다.
                pred_gloss = infos[predicate_idx].gloss
                only_noun = len(noun_idxs_before_pred) == 1
                if _is_adjective(infos[predicate_idx]):
                    parts.append(kiwi.join([(g, tag), ("가", "JKS")]))
                elif pred_gloss.endswith(MOVEMENT_VERB_SUFFIXES):
                    if tag == "NP" and only_noun:
                        # 대명사가 이동동사 앞에 단독으로 오면 목적지가 아니라
                        # 주어다 ("나 빨리 가다" -> "나는 가요", "학교에 가요" X).
                        parts.append(kiwi.join([(g, tag), ("는", "JX")]))
                    else:
                        parts.append(kiwi.join([(g, tag), ("에", "JKB")]))
                else:
                    parts.append(kiwi.join([(g, tag), ("를", "JKO")]))
            elif i == 0:
                # 술어 앞에 명사가 둘 이상일 때만 맨 앞 명사를 주제로 삼는다.
                parts.append(kiwi.join([(g, tag), ("는", "JX")]))
            else:
                parts.append(g)
        else:
            parts.append(g)

    return " ".join(parts)


class SentenceBuilder:
    """단어가 하나씩 인식될 때마다 모았다가, 한동안(문장 사이) 손이 계속
    멈춰 있으면 지금까지 모은 단어들로 문장을 완성한다.

    kslx.stream.SegmentGate 는 "단어 하나 끝"을 감지하는 것이고, 이건 그
    위에서 "문장 하나 끝"(더 긴 정지)을 감지한다 — 문턱값을 다르게 둬서
    "단어 사이 짧은 쉼"과 "문장 사이 긴 쉼"을 구분한다.
    """

    def __init__(self, sentence_end_hold: int = 45):
        self.sentence_end_hold = sentence_end_hold
        self.words: list[str] = []
        self.idle_frames = 0

    def on_word(self, gloss: str) -> None:
        self.words.append(gloss)
        self.idle_frames = 0

    def tick_idle(self) -> bool:
        """매 프레임 idle(단어 인식 중이 아닌) 상태일 때 호출.
        반환값 True면 문장을 마무리할 시점이라는 뜻(단, words 가 있을 때만)."""
        if not self.words:
            return False
        self.idle_frames += 1
        return self.idle_frames >= self.sentence_end_hold

    def finalize(self) -> tuple[list[str], str]:
        words = self.words
        sentence = compose_sentence(words)
        self.words = []
        self.idle_frames = 0
        return words, sentence
