# LLM 3축 각본 분석: 모델 간 교차 검증

> **동일한 41편의 각본, 동일한 프롬프트, 서로 다른 모델.**
> "소넷이 찜찜해서 추가로 돈을 태웠다."

---

## 1. 배경: 왜 다른 모델을 돌렸나

처음에 **Claude Sonnet 4.5**로 41편의 각본을 분석했다. 9명의 감독, 세 가지 축(Information Density, Payoff Rate, Direction-Dialogue Balance).

결과는 통계적으로 유의미했고, 감독별 패턴도 흥미로웠다. **그런데 찜찜했다.**

Sonnet의 채점 범위를 보면:

| 지표 | 최솟값 | 최댓값 | 범위 | 표준편차 |
|------|--------|--------|------|----------|
| Information Density | 72 | 82 | **10** | 3.2 |
| Payoff Rate | 78 | 92 | 14 | 3.6 |
| Balance Rating | 8 | 9 | **1** | 0.3 |

Density 점수가 72~82 사이에 몰려 있고, Balance는 사실상 8 아니면 9. 이 정도의 분산으로는 영화 간 차이를 제대로 포착하고 있는 건지 확신할 수 없었다. **"혹시 소넷이 점수를 양자화하고 있는 건 아닌가?"**

그래서 추가로 세 모델을 더 돌렸다:
- **Claude Opus 4.6** (Anthropic 최상위 모델)
- **GPT-5.2** (OpenAI 최신 모델)
- **Gemini 2.5 Pro** (Google)

비용은... 별도로 계산하지 않기로 했다.

---

## 2. 실험 설계

### 통제 변수
- **프롬프트**: 동일 (3축 분석, JSON 출력 요청)
- **각본 텍스트**: 동일 (동일 캐시 파일에서 로드)
- **전처리**: 동일 (120K 초과 시 앞뒤 60K 유지, 중간 생략)
- **영화 목록**: 동일 (9 감독, 41편)

### 조작 변수
- **모델만 다름** (temperature 기본값 사용)

### 모델 목록

| 모델 | API | 식별자 | 상태 |
|------|-----|--------|------|
| Claude Sonnet 4.5 | Anthropic | `claude-sonnet-4-5-20250929` | 41/41 완료 |
| Claude Opus 4.6 | Anthropic | `claude-opus-4-6` | 41/41 완료 |
| GPT-5.2 | OpenAI | `gpt-5.2` | 41/41 완료 |
| Gemini 2.5 Pro | Google | `gemini-2.5-pro` | 11/41 **중단** |

---

## 3. Gemini는 왜 빠졌나

Gemini 2.5 Pro의 11편 결과:

| 지표 | 최솟값 | 최댓값 | 평균 |
|------|--------|--------|------|
| Information Density | 85 | 95 | 91.5 |
| Payoff Rate | 95 | 99 | 97.5 |

**모든 각본이 최고 수준.** 2001: A Space Odyssey의 Density가 85, Payoff가 98. The Shining의 Density가 92, Payoff가 97.

이것은 전형적인 **천장 효과(ceiling effect)**다. Gemini 2.5 Pro는 "유명한 영화의 각본"이라는 사실 자체에 반응하여 거의 만점에 가까운 점수를 부여했다. 영화 간 차이를 전혀 변별하지 못했기 때문에 분석 도구로서의 가치가 없다.

API 비용이 아깝지만, 이 자체가 하나의 발견이다: **모든 LLM이 좋은 분석가는 아니다.**

---

## 4. 모델 특성 비교

### 채점 분포 요약

| 지표 | Sonnet 4.5 | Opus 4.6 | GPT-5.2 |
|------|-----------|----------|---------|
| **Density** | | | |
| - 범위 | 72 ~ 82 | 52 ~ 92 | 44 ~ 86 |
| - 평균 (SD) | 74.9 (3.2) | 77.7 (9.3) | 68.2 (8.6) |
| **Payoff Rate** | | | |
| - 범위 | 78 ~ 92 | 68 ~ 95 | 64 ~ 92 |
| - 평균 (SD) | 86.6 (3.6) | 86.7 (6.0) | 83.8 (7.0) |
| **Direction%** | | | |
| - 범위 | 35 ~ 78 | 25 ~ 68 | 34 ~ 82 |
| - 평균 (SD) | 58.0 (9.3) | 50.2 (11.0) | 58.6 (10.7) |
| **Balance** | | | |
| - 범위 | 8 ~ 9 | 5 ~ 10 | 6 ~ 9 |
| - 평균 (SD) | 8.9 (0.3) | 8.5 (1.0) | 8.1 (0.7) |

### 모델별 성격

**Sonnet 4.5 — "무난한 채점자"**
- Density를 72~82 범위에 압축. 모든 각본이 "꽤 좋다" 수준으로 수렴.
- Balance는 사실상 이진수 (8 또는 9).
- 상대 순위는 유지하지만, 변별력이 부족하다.

**Opus 4.6 — "변별력 있는 평론가"**
- Density 범위 40점 (52~92). 가장 넓은 분산.
- The Shining에 52점, No Country for Old Men에 92점 — 영화 간 차이를 과감하게 표현.
- Balance도 5~10으로 유일하게 만점과 낮은 점수를 모두 사용.

**GPT-5.2 — "엄격한 심사위원"**
- 전체적으로 가장 낮은 평균 점수 (Density 68.2, Payoff 83.8).
- Density 최저 44점 (The Shining), 최고 86점 (The Social Network).
- Opus와 비슷한 변별력을 보이되, 기준선 자체가 더 낮다.

---

## 5. 41편 전체 비교 테이블

### Information Density

| 감독 | 영화 | Sonnet | Opus | GPT |
|------|------|--------|------|-----|
| **Stanley Kubrick** | 2001: A Space Odyssey | 72 | 62 | 46 |
| | The Shining | 72 | 52 | 44 |
| | Barry Lyndon | 72 | 62 | 67 |
| | **평균** | **72.0** | **58.7** | **52.3** |
| **Alfred Hitchcock** | Psycho | 78 | 82 | 74 |
| | Rear Window | 72 | 82 | 62 |
| | **평균** | **75.0** | **82.0** | **68.0** |
| **Christopher Nolan** | The Dark Knight | 78 | 88 | 74 |
| | Inception | 78 | 88 | 72 |
| | Interstellar | 72 | 72 | 63 |
| | Memento | 78 | 88 | 74 |
| | **평균** | **76.5** | **84.0** | **70.8** |
| **Martin Scorsese** | The Departed | 78 | 85 | 67 |
| | Taxi Driver | 78 | 82 | 62 |
| | Wolf of Wall Street | 72 | 78 | 67 |
| | Casino | 78 | 82 | 72 |
| | Raging Bull | 78 | 82 | 74 |
| | Gangs of New York | 72 | 72 | 58 |
| | **평균** | **76.0** | **80.2** | **66.7** |
| **Steven Spielberg** | Schindler's List | 78 | 88 | 73 |
| | Saving Private Ryan | 72 | 68 | 63 |
| | Jurassic Park | 72 | 58 | 62 |
| | Jaws | 72 | 82 | 72 |
| | Minority Report | 72 | 62 | 63 |
| | **평균** | **73.2** | **71.6** | **66.6** |
| **Quentin Tarantino** | Pulp Fiction | 72 | 82 | 62 |
| | Django Unchained | 72 | 72 | 56 |
| | Inglourious Basterds | 72 | 72 | 67 |
| | Reservoir Dogs | 72 | 72 | 62 |
| | Jackie Brown | 72 | 72 | 63 |
| | **평균** | **72.0** | **74.0** | **62.0** |
| **David Fincher** | Fight Club | 78 | 82 | 74 |
| | Se7en | 78 | 78 | 72 |
| | The Social Network | 78 | 88 | 86 |
| | Panic Room | 78 | 82 | 74 |
| | **평균** | **78.0** | **82.5** | **76.5** |
| **James Cameron** | Aliens | 78 | 82 | 72 |
| | Titanic | 72 | 72 | 62 |
| | Avatar | 72 | 78 | 62 |
| | True Lies | 72 | 72 | 72 |
| | **평균** | **73.5** | **76.0** | **67.0** |
| **Coen Brothers** | No Country for Old Men | 82 | 92 | 86 |
| | Fargo | 78 | 88 | 79 |
| | The Big Lebowski | 78 | 72 | 66 |
| | Barton Fink | 78 | 82 | 72 |
| | True Grit | 78 | 88 | 78 |
| | Blood Simple | 72 | 82 | 74 |
| | Raising Arizona | 72 | 82 | 74 |
| | Burn After Reading | 72 | 82 | 74 |
| | **평균** | **76.2** | **83.5** | **75.4** |

### Payoff Rate

| 감독 | 영화 | Sonnet | Opus | GPT |
|------|------|--------|------|-----|
| **Stanley Kubrick** | 2001: A Space Odyssey | 88 | 78 | 78 |
| | The Shining | 88 | 88 | 81 |
| | Barry Lyndon | 85 | 78 | 84 |
| | **평균** | **87.0** | **81.3** | **81.0** |
| **Alfred Hitchcock** | Psycho | 92 | 92 | 92 |
| | Rear Window | 88 | 92 | 86 |
| | **평균** | **90.0** | **92.0** | **89.0** |
| **Christopher Nolan** | The Dark Knight | 92 | 92 | 88 |
| | Inception | 88 | 93 | 90 |
| | Interstellar | 85 | 82 | 68 |
| | Memento | 92 | 95 | 92 |
| | **평균** | **89.2** | **90.5** | **84.5** |
| **Martin Scorsese** | The Departed | 85 | 88 | 78 |
| | Taxi Driver | 85 | 88 | 84 |
| | Wolf of Wall Street | 85 | 75 | 74 |
| | Casino | 85 | 88 | 86 |
| | Raging Bull | 85 | 88 | 86 |
| | Gangs of New York | 81 | 78 | 64 |
| | **평균** | **84.3** | **84.2** | **78.7** |
| **Steven Spielberg** | Schindler's List | 85 | 91 | 86 |
| | Saving Private Ryan | 78 | 78 | 78 |
| | Jurassic Park | 78 | 68 | 74 |
| | Jaws | 85 | 88 | 88 |
| | Minority Report | 81 | 78 | 74 |
| | **평균** | **81.4** | **80.6** | **80.0** |
| **Quentin Tarantino** | Pulp Fiction | 88 | 91 | 78 |
| | Django Unchained | 85 | 88 | 78 |
| | Inglourious Basterds | 88 | 85 | 79 |
| | Reservoir Dogs | 85 | 85 | 86 |
| | Jackie Brown | 85 | 88 | 86 |
| | **평균** | **86.2** | **87.4** | **81.4** |
| **David Fincher** | Fight Club | 92 | 91 | 89 |
| | Se7en | 85 | 82 | 78 |
| | The Social Network | 85 | 92 | 91 |
| | Panic Room | 85 | 88 | 88 |
| | **평균** | **86.8** | **88.2** | **86.5** |
| **James Cameron** | Aliens | 92 | 88 | 88 |
| | Titanic | 88 | 88 | 86 |
| | Avatar | 85 | 88 | 74 |
| | True Lies | 85 | 78 | 86 |
| | **평균** | **87.5** | **85.5** | **83.5** |
| **Coen Brothers** | No Country for Old Men | 88 | 93 | 91 |
| | Fargo | 92 | 92 | 92 |
| | The Big Lebowski | 92 | 88 | 86 |
| | Barton Fink | 92 | 91 | 92 |
| | True Grit | 85 | 92 | 91 |
| | Blood Simple | 88 | 92 | 88 |
| | Raising Arizona | 85 | 88 | 92 |
| | Burn After Reading | 88 | 88 | 86 |
| | **평균** | **88.8** | **90.5** | **89.8** |

### Direction-Dialogue Balance (Direction %)

| 감독 | 영화 | Sonnet | Opus | GPT |
|------|------|--------|------|-----|
| **Stanley Kubrick** | 2001: A Space Odyssey | 68 | 58 | 82 |
| | The Shining | 78 | 65 | 68 |
| | Barry Lyndon | 55 | 40 | 58 |
| | **평균** | **67.0** | **54.3** | **69.3** |
| **Alfred Hitchcock** | Psycho | 62 | 55 | 58 |
| | Rear Window | 70 | 65 | 64 |
| | **평균** | **66.0** | **60.0** | **61.0** |
| **Christopher Nolan** | The Dark Knight | 58 | 55 | 62 |
| | Inception | 62 | 55 | 62 |
| | Interstellar | 58 | 62 | 58 |
| | Memento | 55 | 48 | 56 |
| | **평균** | **58.2** | **55.0** | **59.5** |
| **Martin Scorsese** | The Departed | 55 | 42 | 56 |
| | Taxi Driver | 58 | 48 | 68 |
| | Wolf of Wall Street | 45 | 38 | 56 |
| | Casino | 55 | 40 | 48 |
| | Raging Bull | 55 | 38 | 58 |
| | Gangs of New York | 62 | 62 | 72 |
| | **평균** | **55.0** | **44.7** | **59.7** |
| **Steven Spielberg** | Schindler's List | 55 | 55 | 62 |
| | Saving Private Ryan | 65 | 58 | 72 |
| | Jurassic Park | 65 | 52 | 68 |
| | Jaws | 65 | 58 | 58 |
| | Minority Report | 58 | 55 | 68 |
| | **평균** | **61.6** | **55.6** | **65.6** |
| **Quentin Tarantino** | Pulp Fiction | 35 | 25 | 34 |
| | Django Unchained | 55 | 48 | 62 |
| | Inglourious Basterds | 42 | 48 | 56 |
| | Reservoir Dogs | 35 | 28 | 38 |
| | Jackie Brown | 55 | 40 | 44 |
| | **평균** | **44.4** | **37.8** | **46.8** |
| **David Fincher** | Fight Club | 58 | 40 | 52 |
| | Se7en | 65 | 58 | 62 |
| | The Social Network | 35 | 30 | 42 |
| | Panic Room | 65 | 55 | 62 |
| | **평균** | **55.8** | **45.8** | **54.5** |
| **James Cameron** | Aliens | 70 | 62 | 62 |
| | Titanic | 58 | 62 | 71 |
| | Avatar | 65 | 68 | 78 |
| | True Lies | 65 | 58 | 67 |
| | **평균** | **64.5** | **62.5** | **69.5** |
| **Coen Brothers** | No Country for Old Men | 62 | 55 | 62 |
| | Fargo | 62 | 42 | 46 |
| | The Big Lebowski | 55 | 35 | 44 |
| | Barton Fink | 62 | 52 | 58 |
| | True Grit | 55 | 42 | 42 |
| | Blood Simple | 65 | 65 | 60 |
| | Raising Arizona | 60 | 55 | 63 |
| | Burn After Reading | 48 | 40 | 44 |
| | **평균** | **58.6** | **48.2** | **52.4** |

---

## 6. 순위 상관분석

세 모델이 **동일한 상대적 순위**를 매기고 있는가? 절대 점수가 달라도, "A가 B보다 낫다"는 판단이 일치한다면 수렴 타당도(convergent validity)가 확보된다.

### 6.1 영화 수준 (N=41)

**Spearman 순위상관계수**

| 모델 쌍 | Density | Payoff Rate | Direction% |
|---------|---------|-------------|------------|
| Sonnet ↔ Opus | ρ = 0.721*** | ρ = 0.620*** | ρ = 0.825*** |
| Sonnet ↔ GPT | ρ = 0.633*** | ρ = 0.570*** | ρ = 0.712*** |
| Opus ↔ GPT | ρ = 0.751*** | ρ = 0.761*** | ρ = 0.782*** |

**Kendall 순위상관계수**

| 모델 쌍 | Density | Payoff Rate | Direction% |
|---------|---------|-------------|------------|
| Sonnet ↔ Opus | τ = 0.650*** | τ = 0.531*** | τ = 0.692*** |
| Sonnet ↔ GPT | τ = 0.553*** | τ = 0.474*** | τ = 0.566*** |
| Opus ↔ GPT | τ = 0.629*** | τ = 0.633*** | τ = 0.627*** |

*\*\*\* p < 0.001 (모든 상관계수가 통계적으로 고도로 유의)*

### 6.2 감독 수준 (N=9)

**Spearman 순위상관계수**

| 모델 쌍 | Density | Payoff Rate | Direction% |
|---------|---------|-------------|------------|
| Sonnet ↔ Opus | ρ = 0.904** | ρ = 0.812** | ρ = 0.783* |
| Sonnet ↔ GPT | ρ = 0.929*** | ρ = 0.783* | ρ = 0.767* |
| Opus ↔ GPT | ρ = 0.900** | ρ = 0.879** | ρ = 0.750* |

*\* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001*

감독 수준에서 Density 상관은 **ρ = 0.900~0.929**로 거의 완벽한 일치. 세 모델은 "어떤 감독의 각본이 더 밀도 있는가"라는 질문에 사실상 같은 답을 내놓았다.

---

## 7. 세 모델이 만장일치로 동의한 것

### 7.1 감독 순위 합의

**Information Density — 감독 평균 순위**

| 순위 | Sonnet | Opus | GPT |
|------|--------|------|-----|
| 1 | David Fincher (78) | Christopher Nolan (84) | David Fincher (76) |
| 2 | Christopher Nolan (76) | Coen Brothers (84) | Coen Brothers (75) |
| 3 | Coen Brothers (76) | David Fincher (82) | Christopher Nolan (71) |
| ... | | | |
| 8 | Stanley Kubrick (72) | Steven Spielberg (72) | Quentin Tarantino (62) |
| 9 | Quentin Tarantino (72) | Stanley Kubrick (59) | Stanley Kubrick (52) |

**만장일치:**
- David Fincher, Coen Brothers, Christopher Nolan은 항상 상위 3에 포함
- Stanley Kubrick은 항상 하위 2에 포함
- Quentin Tarantino는 항상 하위 3에 포함

**Direction% (대사 비중) — 가장 대사 중심인 감독**

| 순위 | Sonnet | Opus | GPT |
|------|--------|------|-----|
| 1 (대사 중심) | Tarantino (44%) | Tarantino (38%) | Tarantino (47%) |
| 2 | Scorsese (55%) | Scorsese (45%) | Coen Brothers (52%) |
| 3 | Fincher (56%) | Fincher (46%) | Fincher (54%) |

**만장일치:** Quentin Tarantino가 가장 대사 중심의 각본을 쓴다. 세 모델 모두 동의.

### 7.2 개별 영화 합의

**세 모델 모두 Density 상위 5에 넣은 영화:**
- No Country for Old Men (Sonnet 82 / Opus 92 / GPT 86)
- The Social Network (Sonnet 78 / Opus 88 / GPT 86)

**세 모델 모두 Density 최하위에 넣은 영화:**
- The Shining (Sonnet 72 / Opus 52 / GPT 44)
- 2001: A Space Odyssey (Sonnet 72 / Opus 62 / GPT 46)

The Shining과 2001이 Density에서 꼴찌라는 것은 직관적으로 타당하다. 쿠브릭의 각본은 시각적 연출에 의존하는 부분이 크고, 텍스트 자체의 정보 밀도는 상대적으로 낮다 — 대신 화면에서 정보를 전달한다. 각본만 읽으면 "여백이 많다"고 느껴지는 것이 정상이며, 이것은 쿠브릭 각본의 약점이 아니라 그의 연출 스타일의 반영이다.

---

## 8. 모델이 의견을 달리한 곳

### Kubrick Direction%: 시각적 영화의 딜레마

| 모델 | Kubrick Direction% |
|------|--------------------|
| Sonnet | 67% |
| Opus | 54% |
| GPT | 69% |

Opus만 Kubrick의 Direction 비율을 낮게 평가했다. 이는 "지문의 정의"에 대한 해석 차이일 가능성이 있다. Kubrick 각본에는 최소한의 지문이 있지만 그 지문 자체가 대사 이상의 서사적 기능을 하기 때문에 — 모델마다 이것을 "direction"으로 분류하는 기준이 다른 것으로 보인다.

### Interstellar Payoff Rate

| 모델 | Payoff Rate |
|------|-------------|
| Sonnet | 85 |
| Opus | 82 |
| GPT | **68** |

GPT만 유독 낮은 점수. GPT-5.2는 Interstellar의 과학적 설정(웜홀, 시간 팽창 등)이 서사적으로 완전히 회수되지 않았다고 판단한 것으로 보인다. 반면 Opus와 Sonnet은 "Murph, don't let me leave" → 5차원 서재 재회를 충분한 payoff로 인정했을 수 있다.

### Gangs of New York: 보편적 저평가

| 모델 | Density | Payoff |
|------|---------|--------|
| Sonnet | 72 | 81 |
| Opus | 72 | 78 |
| GPT | 58 | **64** |

세 모델 모두 스코세이지 작품 중 가장 낮게 평가. 특히 GPT는 Payoff 64점으로 가장 엄격. 각본의 산만함에 대한 비평적 공감대(critically acknowledged bloat)가 LLM에서도 재현되었다.

---

## 9. 해석: 무엇이 검증되었나

### 9.1 수렴 타당도 (Convergent Validity)

세 개의 독립적인 LLM이 동일한 각본에 대해 **통계적으로 유의미한 순위 일치**를 보였다 (모든 Spearman ρ > 0.57, 모든 p < 0.001). 이는 LLM 기반 각본 분석이 단일 모델의 편향이 아닌, **텍스트에 내재된 실제 특성**을 포착하고 있음을 시사한다.

특히 감독 수준에서 Density 상관이 ρ = 0.900~0.929에 달한다는 것은:
- 세 모델이 "밀도 높은 각본"과 "여유로운 각본"을 거의 동일하게 구분한다
- 이 구분이 모델 고유의 편향이 아닌 텍스트의 객관적 특성에 기반한다

### 9.2 변별 타당도 (Discriminant Validity)

세 모델의 **절대 점수 척도가 다르다**는 사실 자체가 중요하다:
- Sonnet: 관대하되 좁은 범위 (압축)
- Opus: 중립적이되 넓은 범위 (변별적)
- GPT: 엄격하되 넓은 범위 (비판적)

서로 다른 채점 기준에도 불구하고 순위가 일치한다면, 이는 순위 자체의 신뢰도를 더욱 강화한다.

### 9.3 Gemini의 교훈

Gemini 2.5 Pro의 실패는 역설적으로 다른 세 모델의 결과를 더 신뢰할 수 있게 만든다. "유명 영화"라는 이유로 무조건 높은 점수를 주는 모델(Gemini)이 존재한다는 것은, 나머지 세 모델이 명성과 무관하게 텍스트를 분석하고 있음을 반증한다.

---

## 10. 결론

영화별로 해석을 다시 해야 하나 싶었지만, 나도 그렇고, 세 LLM의 상대평가가 일치하므로 그럴 필요까지는 없는 것 같다. 절대 점수는 다르지만, "누가 더 밀도 있고, 누가 더 대사 중심이며, 누가 더 설정을 잘 회수하는가"에 대해서는 사실상 같은 이야기를 하고 있다.

### 번외 검증: 기생충 (Parasite)

추가로 번외 분석에서 다뤘던 기생충을 Opus와 GPT-5.2에도 돌려보았다.

| | Sonnet | Opus | GPT |
|---|---|---|---|
| Density | 82 | 92 | 74 |
| Payoff | 91 | 95 | 92 |
| Direction% | 58 | 55 | 58 |

Opus는 92점으로 41편 전체에서 No Country for Old Men과 동률 1위에 해당하는 점수를 부여했다. GPT도 74점으로 자체 기준에서 상위권. 세 모델 모두 "봉준호 각본의 밀도와 설정 회수율이 극도로 높다"에 동의했다.

이 결과는 나머지 번외 영화들에 대해 추가 교차 검증을 할 필요가 없음을 시사한다. **Sonnet의 채점은 양자화(좁은 범위 압축)라는 표현상의 한계가 있을 뿐, 판단 자체는 신뢰할 만하다.** 결국 400달러를 태워서 확인한 건: "소넷 맞았다."

---

## 부록 A: 기술적 세부사항

### 실행 환경
- 스크립트: `model_comparison.py`
- 체크포인트: 영화별 자동 저장 (중단 후 재개 가능)

### API 호출 사양

| 모델 | max_tokens | timeout | 비고 |
|------|-----------|---------|------|
| Sonnet 4.5 | 4,096 | 300s | 최초 분석 |
| Opus 4.6 | 4,096 | 600s | |
| GPT-5.2 | 4,096 (max_completion_tokens) | 600s | `max_tokens` 미지원 |
| Gemini 2.5 Pro | 4,096 (maxOutputTokens) | 600s | 11편 후 중단 |

### 분석 프롬프트 (3축)
1. **Information Density (0-100)**: 서사에 직접 기여하는 텍스트의 비율
2. **Payoff Rate (0-100)**: 설정(setup)의 회수(payoff) 비율
3. **Direction-Dialogue Balance**: 지문 대 대사 비율 및 적절성 (1-10)

## 부록 B: 41편 모델별 분석 근거

각 모델이 점수를 매긴 이유 — 밀도 판단 근거, 회수율 판단 근거, 핵심 설정 목록, 균형 판단 근거, 종합 평가 원문 전문.

**[부록 B 전문 보기 (LLM_MODEL_COMPARISON_APPENDIX_B.md)](./LLM_MODEL_COMPARISON_APPENDIX_B.md)**
