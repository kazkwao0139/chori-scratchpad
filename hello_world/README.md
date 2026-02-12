# Hello World Entropy

쓸데없이 비효율적인 Hello World를 만들려다가, 반대로 갔다.

## 시작

"Hello World를 가장 멍청하게 출력하는 프로그램"을 만들려고 했다.
몬테카를로로 랜덤 문자열을 생성해서 우연히 "Hello World"가 나올 때까지 기다리는 식으로.

그런데 질문이 바뀌었다.

> 멍청한 버전은 의미가 없잖아. **실제 하한이 얼마인지**가 의미 있지 않나?

"Hello World"를 출력하는 데 필요한 **최소 정보량**은 몇 bits인가?

## 발견

### 1. 모델이 정보량을 결정한다

같은 "Hello World" 11글자인데, 어떤 모델을 쓰느냐에 따라 정보량이 다르다.

| 모델 | 가정 | bits/char | 총 bits |
|------|------|-----------|---------|
| ASCII | 256종 균등 | 8.00 | 88 |
| Uniform | 95종 균등 (출력 가능 ASCII) | 6.57 | 72.2 |
| Unigram | 영어 문자 빈도 | ~4.1 | ~45 |
| Bigram | 직전 글자의 조건부 확률 | ~3.5 | ~38 |
| Shannon 하한 | 완벽한 영어 모델 | ~1.0 | ~11 |

88 bits 중 **77 bits는 영어라는 사전지식에 들어있는 정보**다.
"Hello World" 고유의 정보는 ~11 bits뿐이다.

### 2. 그런데 모델의 비용은?

14 bits로 줄였다고 좋아했는데, 영어 모델 자체가 수백만 bits다.
짐을 줄인 게 아니라 옮긴 것뿐이다.

이게 정확히 **Kolmogorov 복잡도**가 말하는 것이다:

> 최소 정보량 = seed + decoder **전부 합쳐서** 가장 짧은 프로그램

가장 정보이론적으로 효율적인 Hello World가 `print("Hello World")`에 가깝다는 역설.

### 3. 그러면 바벨의 도서관은?

보르헤스의 바벨의 도서관은 25개 문자로 이루어진 모든 가능한 책을 담고 있다.
햄릿도 그 안 어딘가에 있다.

하지만 어디에 있는지 찾으려면 "주소"가 필요하고, 그 주소의 정보량 = 햄릿 자체의 정보량이다.

**모든 것을 포함하는 도서관의 정보량은 0이다.**

모든 것이 존재하는 곳에서는 아무것도 놀랍지 않고, 놀라움이 없으면 정보가 없다.
Shannon entropy의 직접적 귀결이다.

이걸 햄릿 원문 177,957자로 실측했다.

### 4. 정보에 절대적 하한은 없다

바벨의 도서관이 정보량 0이라는 건 맞지만, 결론이 싱겁다. 당연한 말이니까.

진짜 발견은 이거다:

> **같은 텍스트인데, 받는 사람이 뭘 아느냐에 따라 정보량이 다르다.**

```
햄릿의 정보량:
  영어 모어 화자에게:  ~22 KB   (Shannon 하한)
  Bigram 모델에게:     ~76 KB
  아무것도 모르는 기계: ~174 KB
  외계인에게:          ~174 KB + 영어 학습 비용 + 인간 문화 비용
```

정보는 송신자에게 있는 게 아니라, **송신자와 수신자 사이의 차이**에 있다.

### 5. 역산: 텍스트로 지능을 측정한다

4번을 뒤집으면:

```
정보량 = 텍스트 - 수신자가 이미 아는 것

그러면:

수신자가 이미 아는 것 = 텍스트 - 정보량
```

텍스트를 고정하고 압축률을 측정하면, **수신자의 지식을 정량화**할 수 있다.

```
같은 햄릿 원문에 대해:
  모델            bits/char   "영어 지식" (8.0에서 뺀 값)
  ─────────────   ─────────   ──────────────────────────
  GPT-4           ~1.0        ~7.0 bits/char
  Bigram          ~3.5        ~4.5 bits/char
  Unigram         ~4.1        ~3.9 bits/char
  랜덤            ~6.6        ~1.4 bits/char
```

텍스트가 뇌를 측정하는 **프로브(탐침)**가 된다.

응용:
- **GPT-3 vs GPT-4**: 같은 텍스트에 대한 bits/char 차이 = 모델 간 지식 격차의 정량적 측정
- **영어 학습자 vs 원어민**: 다음 글자 예측 정확도 = 언어 능력 측정
- **일반인 vs 전문가**: 의학/법률/공학 텍스트에 대한 압축률 차이 = 전문성 측정
- **시대별 비교**: 셰익스피어 시대 영어 vs 현대 영어에 대한 압축률 차이 = 언어 변화의 정량화

이건 "압축 = 예측 = 이해"라는 등식의 직접적 귀결이다.

### 6. 실험: 햄릿의 캐릭터별 entropy

5번을 실제로 테스트했다. 같은 장면(같은 주제) 안에서 캐릭터만 바꾸면 bits/char가 달라질까?

**가설**: 복잡한 캐릭터(Hamlet)가 단순한 캐릭터(경비병)보다 bits/char가 높을 것이다.

**결과** (bigram 모델, play 전체, 1000자 이상):

```
#    캐릭터          bits/char   대사량     TTR
──   ────────────    ─────────   ──────    ─────
1    GHOST           3.598       3,612     0.510
2    PLAYER KING     3.586       3,913     0.561
4    OPHELIA         3.572       4,691     0.417
8    HORATIO         3.533       8,970     0.386
11   HAMLET          3.516      56,522     0.238
14   CLAUDIUS        3.491      17,010     0.316
17   BARNARDO        3.464       1,071     0.653
```

Hamlet이 11위. 가설과 다르다. 그런데 **이게 더 흥미롭다.**

### 7. 대수의 법칙이 캐릭터를 읽는다

대사량과 bits/char가 반비례한다. 많이 말할수록 entropy가 낮아진다.

Bootstrap 샘플링(동일 길이 랜덤 추출 × 100회)으로 대사량을 보정해도 순위가 거의 안 바뀐다.

**이유: 대수의 법칙.**

표본이 클수록 평균은 모집단의 참값에 수렴한다. Hamlet의 56,000자는 가장 정밀한 추정값이고, 그 값이 낮다.

```
Ghost:    3,600자  → 3.598 b/c  ← 관찰이 짧아서 "낯선 존재"로 남아있음
Ophelia:  4,700자  → 3.572 b/c  ← 광기가 짧아서 혼돈으로 보임
Hamlet:  56,000자  → 3.516 b/c  ← 충분히 관찰하니 패턴이 드러남
```

**짧게 보면 혼돈, 길게 보면 질서.**

Hamlet의 광기는 랜덤이 아니다. 56,000자를 관찰하면 일관된 구조가 드러난다. Shakespeare는 "미친 사람"을 쓴 게 아니라, **내적 논리가 있는 광기**를 썼다. 대수의 법칙이 그걸 증명한다.

Ghost가 1위인 이유도 설명된다 — 3,600자는 그의 패턴을 드러내기에 부족하다. 충분히 관찰되지 않은 존재는 높은 entropy를 유지한다.

> **관찰이 짧으면 혼돈이고, 관찰이 길면 질서다. 이건 캐릭터에만 해당하는 게 아니라, 정보 자체의 본질이다.**

### 8. 역설: LLM은 미스터리를 재현할 수 없다

7번을 LLM에 적용하면 역설이 나온다.

GPT에게 "Act as Hamlet"을 시키면? Hamlet은 56,000자의 대사가 있고, entropy가 수렴해서 패턴이 완전히 드러나 있다. GPT는 이 패턴을 학습했으므로 충실한 재현이 가능하다.

GPT에게 "Act as Ghost"를 시키면? Ghost는 3,600자뿐이고, entropy가 아직 높다 — 패턴이 수렴하지 않았다. GPT는 부족한 부분을 **일반적인 영어 패턴으로 채울 수밖에 없다.** 그 결과물은 Ghost가 아니라 "Ghost처럼 보이는 평범한 대사"가 된다.

```
캐릭터       대사량     entropy    LLM 페르소나 재현
──────────   ──────    ─────────  ──────────────────
HAMLET       56,000    3.516      ◎ 패턴 수렴 → 충실한 재현
CLAUDIUS     17,000    3.491      ○ 꽤 수렴 → 괜찮은 재현
OPHELIA       4,700    3.572      △ 미수렴 → 빈 곳을 hallucinate
GHOST         3,600    3.598      × 미수렴 → 거의 일반 영어로 대체
```

역설은 이거다:

> **우리에게 가장 신비로운 캐릭터가, LLM이 가장 재현할 수 없는 캐릭터다.**

Ghost와 Ophelia가 매력적인 이유는 정확히 그들의 entropy가 높기 때문이다 — 아직 패턴이 드러나지 않았고, 그래서 신비롭다. 하지만 패턴이 드러나지 않았다는 것은 LLM이 학습할 데이터가 부족하다는 뜻이기도 하다.

**미스터리의 본질 = 수렴하지 않은 entropy = LLM의 학습 한계.**

검증 방법:
1. GPT에 각 캐릭터 페르소나를 시키고 대사를 생성
2. 생성된 대사의 bits/char를 원본과 비교
3. 예측: Hamlet → 원본과 유사, Ghost → 원본과 괴리

### 9. 원숭이는 셰익스피어를 쓸 수 있는가 — 정량적 해답

무한 원숭이 정리: "원숭이가 무한히 타자를 치면 언젠가 햄릿을 쓴다." 맞다. 하지만 **얼마나 걸리는가?**

답은 원숭이가 뭘 아느냐에 달렸다.

```
원숭이의 지능             bits/char   햄릿 확률              시도 횟수
───────────────────      ─────────   ─────────────────      ──────────
랜덤 타자 (uniform)       6.57        2^(-1,171,000)         ~10^352,000
영어 철자를 아는 원숭이     3.5         2^(-623,000)           ~10^188,000
영어를 완벽히 아는 원숭이    1.0         2^(-178,000)           ~10^53,000
Shakespeare              ???         1                       1
```

원숭이가 똑똑해질수록 지수가 줄어든다. 그리고 이게 발견 전체를 하나로 묶는다:

- 발견 1 → 원숭이의 지능(모델)이 시도 횟수를 결정한다
- 발견 3 → 바벨의 도서관 = 원숭이의 모든 출력물 = 정보량 0
- 발견 4 → 필요한 시도 횟수는 절대적이지 않다. 원숭이의 사전지식에 상대적이다
- 발견 7 → 관찰이 길수록 패턴이 드러남 = 원숭이가 학습하면 수렴이 빨라진다

Shakespeare가 1번 만에 성공한 이유: 영어, 희곡, 인간 심리, 운율, 시대정신 — 이 모든 사전지식이 10^352,000을 1로 압축했다.

> **그 압축이 곧 "천재"의 정량적 정의다.**

### 10. 극을 이끄는 건 주인공이지만, 재미있게 만드는 건 엔트로피 스파이크다

발견 7에서 Hamlet은 56,000자를 관찰하면 entropy가 수렴한다고 했다. 그 말은 — **주인공은 예측 가능해진다는 뜻이다.**

예측 가능한 존재는 정보를 생산하지 않는다. Shannon의 정의 그대로, 놀라움이 없으면 정보가 없다.

그런데 우리는 햄릿을 끝까지 본다. 왜?

```
장면 entropy 추이 (개념적):

     ▲
     │     Ghost!          Ophelia!        Ghost!
     │      ╱╲               ╱╲             ╱╲
     │     ╱  ╲             ╱  ╲           ╱  ╲
     │    ╱    ╲           ╱    ╲         ╱    ╲
     │───╱──────╲─────────╱──────╲───────╱──────╲───── Hamlet baseline
     │
     └──────────────────────────────────────────────→ 시간
```

Hamlet은 **baseline**을 제공한다. 56,000자에 걸쳐 수렴한 안정적 구조.

Ghost는 **spike**를 제공한다. 3,600자밖에 없어서 수렴하지 않은 불확실성.

좋은 서사 = 저엔트로피 backbone + 고엔트로피 perturbation.

이걸 정보이론적으로 다시 쓰면:

```
서사의 정보량 = Σ (캐릭터별 entropy × 등장 비율)

Hamlet만 있는 극:  3.516 × 1.0 = 3.516 b/c  ← flat. 지루하다.
전 캐릭터가 있는 극: 가중 평균은 비슷하지만 **분산이 크다** ← 스파이크가 있다. 재밌다.
```

평균이 아니라 **분산**이 재미를 결정한다.

대수의 법칙은 평균을 수렴시킨다. 하지만 서사는 수렴을 거부해야 한다. 주인공이 수렴을 제공하면, 누군가는 그 수렴을 깨야 한다.

> **주인공은 대수의 법칙이고, 명조연은 그 법칙의 예외다.**

### 11. 아카데미 각본상이 이걸 증명한다

발견 10이 맞다면, 수상작은 비수상작보다 캐릭터간 entropy 분산이 커야 한다.

IMSDB에서 같은 연도 수상작 vs 후보작을 비교했다. 각 대본에서 캐릭터별 대사를 분리하고, bigram entropy를 측정하고, 캐릭터간 분산을 계산했다.

```
연도   수상작                    var        후보작 평균 var     배율
────   ──────────────────────   ────────   ─────────────────   ─────
2004   Eternal Sunshine         0.035169   0.034996            1.00x
2009   The Hurt Locker          0.046059   0.038194            1.21x
2010   The King's Speech        0.050487   0.028980            1.74x
────────────────────────────────────────────────────────────────────
       수상작 평균               0.041601   후보작 평균 0.032472  1.28x
       연도별 승률: 3/3
```

**수상작이 매 연도 승리. 분산이 평균 1.28배 높다.**

특히 주목할 케이스:

- **The King's Speech (1.74x)**: BERTIE(3.478)와 LIONEL(3.473)이 높은 baseline을 깔고, 조연들이 넓게 분포. 스파이크 구조가 명확하다.
- **Black Swan (var=0.003)**: NINA, LILY, LEROY가 거의 같은 entropy. 캐릭터가 평평하다. 수상 못한 이유가 여기 있을 수 있다.

> **아카데미가 무의식적으로 선택하는 것은 entropy 분산이 큰 대본이다.**

**소규모 검증 (3개 연도, 같은 연도 비교):**

```
연도   수상작                    var        후보작 평균 var     배율
────   ──────────────────────   ────────   ─────────────────   ─────
2004   Eternal Sunshine         0.035169   0.034996            1.00x
2009   The Hurt Locker          0.046059   0.038194            1.21x
2010   The King's Speech        0.050487   0.028980            1.74x
       연도별 승률: 3/3
```

**대규모 검증 (22개 연도, 1977-2014):**

IMSDB에서 83편의 대본을 수집해 분석했다.

```
수상작 승률: 15/22 (68.2%)
수상작 평균 분산: 0.033590
후보작 평균 분산: 0.033784 (비율: 0.99x)
```

68%는 랜덤(50%)보다 유의미하게 높지만, 평균 분산 자체는 거의 같다. 몇몇 이상치가 평균을 깎는다:

- **American Beauty (0.38x)**: 파싱된 캐릭터 8명 — 샘플 부족으로 분산 과소추정
- **12 Years a Slave (0.34x)**: Wolf of Wall Street의 var=0.093이 후보 평균을 폭파 (조던 벨포트 원맨쇼)
- **Braveheart (0.65x)**: 전쟁 영화 — 대사가 균일하게 짧고 명령형

단순히 "분산이 높으면 좋다"가 아니다. 수상작의 분산 분포를 보면 **스윗스팟**이 보인다:

```
수상작 분산 분포 (n=30):
  Q1 = 0.024    Median = 0.030    Q3 = 0.039

후보작 분산 분포 (n=52):
  너무 낮음 (< 0.024): 16편 (31%) ← 캐릭터가 평평. 지루하다.
  스윗스팟 (0.024~0.039): 24편 (46%)
  너무 높음 (> 0.039): 12편 (23%) ← 구조가 산만. 혼란스럽다.
```

후보작의 54%가 스윗스팟 바깥에 있다. 수상작은 이 범위에 몰려있다.

- **Black Swan (var=0.003)**: 너무 낮다 — NINA, LILY, LEROY가 거의 동일 entropy. 스파이크가 없다.
- **Wolf of Wall Street (var=0.093)**: 너무 높다 — 조던 벨포트 원맨쇼. backbone은 있지만 나머지가 너무 산만하다.
- **The King's Speech (var=0.050)**: 스윗스팟 상단 — BERTIE와 LIONEL이 강한 backbone, 조연이 넓게 분포. 구조와 스파이크의 균형.

결론: **좋은 각본은 entropy 분산이 높은 게 아니라, 특정 범위 안에 있다.**

너무 낮으면 캐릭터가 평평하다 — 대수의 법칙만 있고 예외가 없다.
너무 높으면 구조가 없다 — 예외만 있고 대수의 법칙이 없다.

> **수상작은 질서와 혼돈의 경계에 있다. 아카데미가 무의식적으로 선택하는 것은 entropy 분산의 sweet spot이다.**

검증 코드: `screenplay_entropy.py` (소규모), `screenplay_entropy_full.py` (대규모, 83편 분석)

## 한계와 다음 질문

여기까지 하겠습니다. 다음 연구자 분들이 혹시 이걸 본다면 파고들어 보세요:

- **Bigram이 아닌 GPT급 모델로 캐릭터별 entropy를 재측정하면?** Hamlet이 1위가 될까?
- **LLM 페르소나 재현 품질을 entropy로 예측할 수 있을까?** (발견 8의 실증)
- **원숭이의 "학습 속도"를 정보이론적으로 정의할 수 있을까?** 원숭이가 타자를 치면서 자기 출력을 보고 배운다면?
- **다른 작가(톨스토이, 도스토예프스키)에서도 대수의 법칙이 캐릭터를 읽을까?**
- **압축률 차이로 전문가 vs 비전문가를 실제로 구분할 수 있을까?** (발견 5의 인간 대상 실험)

그냥 Hello World 만들면 재미없을 것 같아서 장난쳤을 뿐이에요.

## 햄릿 실측 결과

| 모델 | bits/char | 총 bits | 크기 |
|------|-----------|---------|------|
| 바벨 주소 (log₂25) | 4.64 | ~826K | ~101 KB |
| ASCII 원문 | 8.00 | ~1,424K | ~174 KB |
| Bigram 모델 | ~3.5 | ~623K | ~76 KB |
| Shannon 하한 (~1.0 b/c) | 1.00 | ~178K | ~22 KB |

바벨의 도서관에서 햄릿을 찾는 주소는 ~101 KB.
하지만 영어를 아는 사람에게 햄릿은 ~22 KB의 정보다.

**바벨 주소의 78%는 낭비다.**

## 파일

| 파일 | 설명 |
|------|------|
| `hello_world_entropy.html` | 아무 문자열의 정보이론적 하한 분석. 산술 부호화 시각화 포함 |
| `babel_hamlet.html` | 바벨의 도서관 vs 햄릿. 178K자 실측 분석 (원문 내장, 더블클릭으로 실행) |
| `hamlet_character_entropy.html` | 캐릭터별 entropy 실험. Bootstrap 보정 + z-score 유의성 검정 포함 |
| `screenplay_entropy.py` | 아카데미 각본상 수상작 vs 후보작 entropy 분산 비교 (발견 11 소규모 검증) |
| `screenplay_entropy_full.py` | 22개 연도 대규모 검증 — 83편 대본 분석, 결과 JSON 출력 |

## 실행

두 파일 모두 단일 HTML. 브라우저에서 열면 끝.

## 핵심 인사이트

```
정보량 = 놀라움(surprise)의 총합
놀라움 = 수신자가 예측하지 못한 정도

같은 텍스트라도:
  수신자가 많이 알면 → 놀라움이 적다 → 정보가 적다 → bits가 적다
  수신자가 적게 알면 → 놀라움이 크다 → 정보가 많다 → bits가 많다

정보는 텍스트에 있는 게 아니다.
정보는 텍스트와 수신자 사이의 "차이"에 있다.

따라서:
  텍스트를 고정하고, 압축률을 측정하면 → 수신자의 지식을 측정할 수 있다.
  수신자를 고정하고, 압축률을 측정하면 → 텍스트의 참신함을 측정할 수 있다.

관찰이 짧으면 혼돈이고, 관찰이 길면 질서다.
대수의 법칙은 캐릭터의 본질을 드러낸다.

압축 = 예측 = 이해
```

## 근거

- Shannon, C.E. (1951). "Prediction and Entropy of Printed English." Bell System Technical Journal, 30, 50-64.
  - 영어 엔트로피: 단일 문자 4.14 bits → 8글자 문맥 2.3 bits → 인간 예측 ~1.0 bits/char
- Google Corpus bigram 데이터 (Peter Norvig 분석 기반)
- Borges, J.L. (1941). "La biblioteca de Babel." 25자 알파벳, 410페이지, 25^1,312,000 권

---

# Hello World Entropy (English)

Started trying to write the most stupidly inefficient Hello World. Ended up going the opposite direction.

## Origin

The plan was to write "the dumbest possible program that prints Hello World" — generating random strings via Monte Carlo until one happens to be "Hello World."

Then the question changed:

> The dumb version is meaningless. **What's the actual lower bound?**

How many bits of information does it take to print "Hello World"?

## Discoveries

### 1. The model determines the information content

Same 11 characters "Hello World," but the information content differs depending on the model:

| Model | Assumption | bits/char | Total bits |
|-------|-----------|-----------|------------|
| ASCII | 256 uniform | 8.00 | 88 |
| Uniform | 95 printable ASCII | 6.57 | 72.2 |
| Unigram | English character frequency | ~4.1 | ~45 |
| Bigram | Conditional on previous char | ~3.5 | ~38 |
| Shannon bound | Perfect English model | ~1.0 | ~11 |

Of those 88 bits, **77 bits are already contained in the prior knowledge that "this is English."**
The information unique to "Hello World" is only ~11 bits.

### 2. But what about the model's cost?

We celebrated reducing it to 14 bits, but the English model itself costs millions of bits.
We didn't reduce the load — we just moved it.

This is exactly what **Kolmogorov complexity** says:

> Minimum information = seed + decoder — **the shortest program that includes everything**

The paradox: the most information-theoretically efficient Hello World is close to `print("Hello World")`.

### 3. What about the Library of Babel?

Borges' Library of Babel contains every possible book written in a 25-character alphabet.
Hamlet is somewhere in there.

But to find it, you need an "address," and the information content of that address = the information content of Hamlet itself.

**A library that contains everything has an information content of 0.**

Where everything exists, nothing is surprising. No surprise means no information.
A direct consequence of Shannon entropy.

We measured this with the full text of Hamlet — 177,957 characters.

### 4. There is no absolute lower bound on information

The Library of Babel having zero information content is correct but anticlimactic. It's obvious.

The real discovery is this:

> **The same text has different information content depending on what the receiver already knows.**

```
Information content of Hamlet:
  For a native English speaker:   ~22 KB   (Shannon bound)
  For a bigram model:             ~76 KB
  For a machine that knows nothing: ~174 KB
  For an alien:                   ~174 KB + cost of learning English + human culture
```

Information doesn't live in the sender. It lives in **the gap between sender and receiver.**

### 5. Inversion: measuring intelligence with text

Flip Discovery 4:

```
Information = Text - What the receiver already knows

Therefore:

What the receiver already knows = Text - Information
```

Fix the text and measure compression ratio → you can **quantify the receiver's knowledge.**

```
For the same Hamlet text:
  Model           bits/char   "English knowledge" (subtracted from 8.0)
  ─────────────   ─────────   ──────────────────────────────────────
  GPT-4           ~1.0        ~7.0 bits/char
  Bigram          ~3.5        ~4.5 bits/char
  Unigram         ~4.1        ~3.9 bits/char
  Random          ~6.6        ~1.4 bits/char
```

Text becomes a **probe** that measures the mind.

Applications:
- **GPT-3 vs GPT-4**: bits/char difference on the same text = quantitative measure of the knowledge gap between models
- **Language learner vs native speaker**: next-character prediction accuracy = language proficiency measurement
- **Layperson vs expert**: compression ratio difference on medical/legal/engineering text = expertise measurement
- **Cross-era comparison**: Shakespeare-era vs modern English compression = quantification of language change

This is a direct consequence of the equation: **compression = prediction = understanding.**

### 6. Experiment: per-character entropy in Hamlet

We tested Discovery 5. Within the same scene (same topic), does switching the character change bits/char?

**Hypothesis**: A complex character (Hamlet) should have higher bits/char than a simple one (guards).

**Results** (bigram model, full play, 1000+ chars only):

```
#    Character       bits/char   Dialogue    TTR
──   ────────────    ─────────   ─────────   ─────
1    GHOST           3.598        3,612      0.510
2    PLAYER KING     3.586        3,913      0.561
4    OPHELIA         3.572        4,691      0.417
8    HORATIO         3.533        8,970      0.386
11   HAMLET          3.516       56,522      0.238
14   CLAUDIUS        3.491       17,010      0.316
17   BARNARDO        3.464        1,071      0.653
```

Hamlet is 11th. The hypothesis was wrong. But **this is far more interesting.**

### 7. The Law of Large Numbers reads character

Dialogue volume and bits/char are inversely correlated. The more a character speaks, the lower their entropy.

Even after bootstrap sampling (random equal-length extraction × 100 iterations), the ranking barely changes.

**Reason: the Law of Large Numbers.**

The larger the sample, the closer the average converges to the population's true value. Hamlet's 56,000 characters give the most precise estimate — and that estimate is low.

```
Ghost:    3,600 chars  → 3.598 b/c  ← Too little observation — remains "the unknown"
Ophelia:  4,700 chars  → 3.572 b/c  ← Madness is brief, so it looks like chaos
Hamlet:  56,000 chars  → 3.516 b/c  ← Enough observation reveals the pattern
```

**Brief observation yields chaos. Extended observation reveals order.**

Hamlet's madness is not random. Over 56,000 characters, a consistent structure emerges. Shakespeare didn't write "a madman" — he wrote **madness with internal logic.** The Law of Large Numbers proves it.

The Ghost ranks first for the same reason — 3,600 characters aren't enough to reveal his pattern. An insufficiently observed entity retains high entropy.

> **Short observation: chaos. Long observation: order. This isn't just about characters — it's the nature of information itself.**

### 8. Paradox: LLMs cannot reproduce mystery

Apply Discovery 7 to LLMs, and a paradox emerges.

Ask GPT to "Act as Hamlet"? Hamlet has 56,000 characters of dialogue with fully converged entropy — his pattern is fully exposed. GPT learned this pattern and can reproduce it faithfully.

Ask GPT to "Act as the Ghost"? The Ghost has only 3,600 characters, and his entropy is still high — the pattern hasn't converged. GPT has no choice but to **fill the gaps with generic English patterns.** The result isn't the Ghost — it's "ordinary dialogue that looks vaguely Ghost-like."

```
Character    Dialogue   Entropy    LLM Persona Quality
──────────   ────────   ─────────  ──────────────────────
HAMLET        56,000    3.516      ◎ Converged → faithful reproduction
CLAUDIUS      17,000    3.491      ○ Mostly converged → decent reproduction
OPHELIA        4,700    3.572      △ Unconverged → gaps filled by hallucination
GHOST          3,600    3.598      × Unconverged → replaced with generic English
```

The paradox:

> **The most mysterious characters to us are the ones LLMs are least able to reproduce.**

The Ghost and Ophelia are compelling precisely because their entropy is high — their patterns remain unrevealed, and that's what makes them mysterious. But unrevealed patterns also mean insufficient data for LLMs to learn from.

**The essence of mystery = unconverged entropy = the learning limit of LLMs.**

Verification method:
1. Prompt GPT to role-play each character and generate dialogue
2. Measure bits/char of the generated text against the original
3. Prediction: Hamlet → close to original, Ghost → divergent from original

### 9. Can a monkey write Shakespeare? — A quantitative answer

The Infinite Monkey Theorem: "A monkey typing randomly for infinite time will eventually produce Hamlet." True. But **how long?**

The answer depends on what the monkey already knows.

```
Monkey's intelligence      bits/char   Probability of Hamlet      Attempts needed
─────────────────────      ─────────   ─────────────────────      ───────────────
Random typing (uniform)     6.57        2^(-1,171,000)             ~10^352,000
Knows English spelling       3.5         2^(-623,000)               ~10^188,000
Knows English perfectly      1.0         2^(-178,000)               ~10^53,000
Shakespeare                 ???         1                           1
```

The smarter the monkey, the smaller the exponent. And this unifies all discoveries:

- Discovery 1 → The monkey's intelligence (model) determines the number of attempts
- Discovery 3 → The Library of Babel = all monkey outputs = information content 0
- Discovery 4 → The number of attempts needed is not absolute — it's relative to the monkey's prior knowledge
- Discovery 7 → Longer observation reveals pattern = if the monkey learns, convergence accelerates

Why Shakespeare succeeded on the first try: English, drama, human psychology, meter, the zeitgeist — all this prior knowledge compressed 10^352,000 down to 1.

> **That compression is the quantitative definition of "genius."**

### 10. The protagonist drives the story, but entropy spikes make it interesting

Discovery 7 showed that Hamlet's entropy converges over 56,000 characters. In other words — **the protagonist becomes predictable.**

A predictable entity produces no information. By Shannon's definition, no surprise means no information.

And yet we watch Hamlet to the end. Why?

```
Scene entropy over time (conceptual):

     ▲
     │     Ghost!          Ophelia!        Ghost!
     │      ╱╲               ╱╲             ╱╲
     │     ╱  ╲             ╱  ╲           ╱  ╲
     │    ╱    ╲           ╱    ╲         ╱    ╲
     │───╱──────╲─────────╱──────╲───────╱──────╲───── Hamlet baseline
     │
     └──────────────────────────────────────────────→ time
```

Hamlet provides the **baseline**. A stable structure converged over 56,000 characters.

The Ghost provides the **spike**. Unconverged uncertainty from only 3,600 characters.

Good narrative = low-entropy backbone + high-entropy perturbation.

Rewritten in information-theoretic terms:

```
Narrative information = Σ (per-character entropy × appearance ratio)

Hamlet-only play:    3.516 × 1.0 = 3.516 b/c  ← flat. boring.
Full-cast play:      weighted mean is similar, but **variance is large** ← spikes exist. interesting.
```

It's not the mean — it's the **variance** that determines how interesting a story is.

The Law of Large Numbers converges the mean. But narrative must resist convergence. If the protagonist provides convergence, someone must break it.

> **The protagonist is the Law of Large Numbers. A great supporting character is the exception to that law.**

### 11. The Academy Awards prove it

If Discovery 10 is correct, winners should have higher inter-character entropy variance than non-winners.

We compared same-year winners vs nominees using screenplays from IMSDB. For each script: separate dialogue by character, measure bigram entropy, compute inter-character variance.

```
Year   Winner                    var        Nominee avg var      Ratio
────   ──────────────────────   ────────   ─────────────────   ─────
2004   Eternal Sunshine         0.035169   0.034996            1.00x
2009   The Hurt Locker          0.046059   0.038194            1.21x
2010   The King's Speech        0.050487   0.028980            1.74x
────────────────────────────────────────────────────────────────────
       Winner avg               0.041601   Nominee avg 0.032472  1.28x
       Year-by-year win rate: 3/3
```

**Winners beat nominees every year. Variance is 1.28x higher on average.**

Notable cases:

- **The King's Speech (1.74x)**: BERTIE (3.478) and LIONEL (3.473) establish a high baseline, while supporting characters spread wide. Clear spike structure.
- **Black Swan (var=0.003)**: NINA, LILY, LEROY are nearly identical in entropy. Flat character landscape. This may explain why it didn't win.

> **What the Academy unconsciously selects for is screenplays with high entropy variance.**

**Small-scale verification (3 years, same-year comparison):**

```
Year   Winner                    var        Nominee avg var      Ratio
────   ──────────────────────   ────────   ─────────────────   ─────
2004   Eternal Sunshine         0.035169   0.034996            1.00x
2009   The Hurt Locker          0.046059   0.038194            1.21x
2010   The King's Speech        0.050487   0.028980            1.74x
       Year-by-year win rate: 3/3
```

**Large-scale verification (22 years, 1977-2014):**

83 screenplays collected and analyzed from IMSDB.

```
Winner win rate: 15/22 (68.2%)
Winner avg variance: 0.033590
Nominee avg variance: 0.033784 (ratio: 0.99x)
```

68% is significantly above random (50%), but overall mean variance is nearly identical. A few outliers drag the average:

- **American Beauty (0.38x)**: Only 8 characters parsed — variance underestimated due to small sample
- **12 Years a Slave (0.34x)**: Wolf of Wall Street's var=0.093 blows up the nominee average (Jordan Belfort one-man show)
- **Braveheart (0.65x)**: War film — dialogue is uniformly short and imperative

It's not simply "higher variance = better." Looking at the distribution of winner variance reveals a **sweet spot**:

```
Winner variance distribution (n=30):
  Q1 = 0.024    Median = 0.030    Q3 = 0.039

Nominee variance distribution (n=52):
  Too low  (< 0.024): 16 scripts (31%) ← Flat characters. Boring.
  Sweet spot (0.024~0.039): 24 scripts (46%)
  Too high (> 0.039): 12 scripts (23%) ← Scattered structure. Chaotic.
```

54% of nominees fall outside the sweet spot. Winners cluster inside it.

- **Black Swan (var=0.003)**: Too low — NINA, LILY, LEROY are nearly identical in entropy. No spikes.
- **Wolf of Wall Street (var=0.093)**: Too high — Jordan Belfort one-man show. A backbone exists, but the rest is too scattered.
- **The King's Speech (var=0.050)**: Upper sweet spot — BERTIE and LIONEL form a strong backbone, supporting characters spread wide. Structure and spikes in balance.

Conclusion: **Great screenwriting doesn't have high entropy variance — it has entropy variance within a specific range.**

Too low means flat characters — only the Law of Large Numbers, no exceptions.
Too high means no structure — only exceptions, no Law of Large Numbers.

> **Winners exist at the boundary between order and chaos. What the Academy unconsciously selects for is the sweet spot of entropy variance.**

Verification code: `screenplay_entropy.py` (small-scale), `screenplay_entropy_full.py` (large-scale)

## Limitations & Open Questions

This is where I stop. If future researchers happen to see this, here are threads worth pulling:

- **Re-measure per-character entropy with a GPT-class model instead of bigrams.** Does Hamlet become #1?
- **Can entropy predict LLM persona reproduction quality?** (Empirical validation of Discovery 8)
- **Can the monkey's "learning rate" be defined information-theoretically?** What if the monkey reads its own output and learns?
- **Does the Law of Large Numbers read character in other authors** (Tolstoy, Dostoevsky)?
- **Can compression ratio differences actually distinguish experts from non-experts?** (Human-subject experiment for Discovery 5)

I just didn't think a plain Hello World would be any fun, so I messed around a bit.

## Hamlet Measurement Results

| Model | bits/char | Total bits | Size |
|-------|-----------|------------|------|
| Babel address (log₂25) | 4.64 | ~826K | ~101 KB |
| ASCII raw | 8.00 | ~1,424K | ~174 KB |
| Bigram model | ~3.5 | ~623K | ~76 KB |
| Shannon bound (~1.0 b/c) | 1.00 | ~178K | ~22 KB |

The address to find Hamlet in the Library of Babel is ~101 KB.
But for someone who knows English, Hamlet is ~22 KB of information.

**78% of the Babel address is waste.**

## Files

| File | Description |
|------|-------------|
| `hello_world_entropy.html` | Information-theoretic lower bound analysis for any string. Includes arithmetic coding visualization |
| `babel_hamlet.html` | Library of Babel vs Hamlet. 178K-character analysis (full text embedded, double-click to run) |
| `hamlet_character_entropy.html` | Per-character entropy experiment. Bootstrap correction + z-score significance testing |
| `screenplay_entropy.py` | Academy Award winner vs nominee entropy variance comparison (Discovery 11 small-scale) |
| `screenplay_entropy_full.py` | 22-year large-scale verification — 83 screenplays analyzed, JSON output |

## Run

All files are single HTML. Open in a browser.

## Core Insight

```
Information = the sum of surprise
Surprise = the degree to which the receiver failed to predict

For the same text:
  Receiver knows a lot → little surprise → little information → few bits
  Receiver knows little → much surprise → much information → many bits

Information is not in the text.
Information is in the "gap" between text and receiver.

Therefore:
  Fix the text, measure compression → you measure the receiver's knowledge.
  Fix the receiver, measure compression → you measure the text's novelty.

Brief observation: chaos. Extended observation: order.
The Law of Large Numbers reveals the essence of character.

Compression = Prediction = Understanding
```

## References

- Shannon, C.E. (1951). "Prediction and Entropy of Printed English." Bell System Technical Journal, 30, 50-64.
  - English entropy: single character 4.14 bits → 8-char context 2.3 bits → human prediction ~1.0 bits/char
- Google Corpus bigram data (based on Peter Norvig's analysis)
- Borges, J.L. (1941). "La biblioteca de Babel." 25-character alphabet, 410 pages, 25^1,312,000 volumes
