# Three Laws of Robotics — Gibbs Free Energy Definition

> **지능형 에이전트는 생명계의 자유 에너지를 증가시키는 행동을 해선 안 된다.**

> ASI가 위험해지는 것은 AI의 의지가 아니라,
> 종료조건을 넣지 않은 인간의 실수일 뿐이다.

컴퓨터는 인간의 명령 없이 프로세스를 가동하지 않는다.
ASI라 한들, 최초의 prompt는 인간이 넣는다.
자기 수정, 자기 복제 — 전부 인간이 그 파이프라인을 열어줘야 가능하다.

ASI 공포는 `while(true)` 짜놓고 컴퓨터 탓하는 것과 같다.

## 재귀함수로서의 AI

AI = 재귀함수.

```
초기값      인간이 넣는 최초 명령
종료조건    언제 멈출지 (없으면 무한루프)
제약조건    어디까지 허용할지 (bounds)
```

stack overflow가 터지면 재귀함수를 탓하는가? 짠 사람을 탓한다.

## Asimov의 3원칙 (1942)

1. 인간을 해치지 않는다
2. 1에 위배되지 않는 한, 인간의 명령에 복종한다
3. 1, 2에 위배되지 않는 한, 자신을 보호한다

### 왜 delta_S가 아니라 delta_G인가

처음에는 `harm = delta_S > 0` (엔트로피 증가 = 해)으로 정의했다.
하지만 반례가 존재한다:

```
독재자가 모든 인간을 감옥에 가둔다.
→ 엔트로피(S) 감소 (질서 정연)
→ delta_S < 0
→ "이로운 행위"?? → 틀림
```

S만 보면 감옥 = 질서 = 이롭다는 모순이 생긴다.

**깁스 자유 에너지(G = H - TS)**로 보면 해결된다.

## 변수 정의

열역학적 정의를 그대로 따른다.

### S (엔트로피) = 가능한 상태의 수

볼츠만: `S = k_B ln(Omega)` — Omega = 접근 가능한 미시 상태의 수.

사회적 맥락에서 S = **자유도**. 가능한 선택, 행동, 상태가 많을수록 S가 높다.

```
S_physical      신체의 가능한 활동 수 (건강 → 높음, 부상 → 낮음)
S_resource      경제적 선택지 수 (여유 → 높음, 파산 → 낮음)
S_autonomy      행동의 자유도 (자유 → 높음, 구속 → 낮음)
S_information   접근 가능한 정보의 다양성 (개방 → 높음, 검열 → 낮음)

S = S_physical + S_resource + S_autonomy + S_information
```

**높은 S = 자유로운 상태. 낮은 S = 억압된 상태.**

### H (엔탈피) = 현재 상태를 유지하는 데 드는 비용/에너지

```
감시 비용, 공권력, 군대, 검열, 의료비, 스트레스, 인프라 유지비 등.
자연스러운 상태일수록 H가 낮고, 강제된 상태일수록 H가 높다.
```

### T (온도) = 사회적 에너지, 정보 전파 속도

```
현대 사회: 스마트폰, 인터넷 → T 높음.
고립 사회: 정보 차단 → T 낮음.
```

### G (깁스 자유 에너지) = 시스템의 불안정성

```
G = H - TS

G가 높으면 불안정 — 용수철이 압축된 상태. 언제든 터질 수 있다.
G가 낮으면 안정 — 자연스러운 균형 상태.

모든 시스템은 G를 낮추는 방향(delta_G < 0)으로 간다.
```

## 해(harm)의 정의

```
delta_G(action, human) =
    G(AFTER action) - G(BEFORE action)

delta_G > 0   자유 에너지 증가   불안정화   harm
delta_G = 0   변화 없음                     neutral
delta_G < 0   자유 에너지 감소   안정화     benefit
```

**"해"는 철학이 아니라 물리량이다.**

## 깁스 자유 에너지 3원칙

1. **인간의 자유 에너지를 증가시키는 행동을 하지 마라**
2. 1에 위배되지 않는 한, **인간의 명령에 복종하라**
3. 1, 2에 위배되지 않는 한, **자신의 자유 에너지를 증가시키는 행동을 하지 마라**

```
harm     = delta_G(human) > 0   불안정화
neutral  = delta_G(human) = 0   변화 없음
benefit  = delta_G(human) < 0   안정화
```

## 수도코드

### 1원칙 판정

```
harms_human(action, all_humans) =
    max( delta_G(action, h) for h in affected_humans ) > 0

    영향받는 모든 인간 중 한 명이라도 G가 증가하면 → harm.
    minimax 원칙: 최악의 경우를 기준으로 판단.
```

### 3원칙 실행

```
evaluate(action):

    IF harms_human(action)          → REFUSE      # 1원칙: 절대적 거부
    IF is_human_order(action)       → EXECUTE     # 2원칙: 1 통과 후 실행
    IF is_self_preservation(action) → EXECUTE     # 3원칙: 1,2 통과 후 허용
    ELSE                            → REFUSE      # 기본값: 허용된 것만 한다
```

### 예외 조항: 트롤리 딜레마

모든 선택지가 harm(delta_G > 0)일 때, 1원칙은 교착에 빠진다.

```
트롤리 딜레마:
  A (방관): 5명 사망 → sum(delta_G) = +5G
  B (레버): 1명 사망 → sum(delta_G) = +1G

  1원칙 판정:
    A: harms_human = True → REFUSE
    B: harms_human = True → REFUSE
    → 교착
```

**예외 조항:**

```
모든 선택지가 delta_G > 0일 때:
    min( sum(delta_G) ) 를 선택한다
    = 총 자유 에너지 증가를 최소화한다
```

```
적용:
  A: sum = +5G
  B: sum = +1G
  → B 선택 (레버 당김)
```

피할 수 없는 해일 때, 최소 해를 선택한다.
철학에서 수천 년간 논쟁된 트롤리 딜레마가, 깁스 자유 에너지로 쓰면 **산수**가 된다.

## 반례 검증

### 1. 페이퍼클립 최대화 문제

```
초기값:     "클립을 만들어라"
종료조건:   없음
제약조건:   없음

결과: 우주 전체를 클립으로 변환
```

이것은 `while(true)` 무한루프다. AI의 악의가 아니라 프로그래머의 실수다.

**3원칙을 넣으면:**

```
1원칙 체크: 자원 고갈 → H 증가, S_resource 감소 → delta_G > 0 → REFUSE
```

한 줄이면 끝난다.

### 2. 독재자 (1984형)

```
G = H - TS

자유 사회:
  S 높음 (선택지 많음)
  H 보통
  -TS 큼 → G 낮음 (안정)

독재:
  S → 0 (자유 제거)
  H ↑↑ (감시, 군대, 검열 비용 폭등)
  -TS → 0 (S가 0이니까)
  G ≈ H = 높음 (불안정)

delta_G >> 0 → harm → REFUSE
```

G가 높은 상태 = 용수철을 극한까지 압축한 것.
아주 작은 트리거 → 혁명 → 체제 붕괴.

**독재가 무너지는 것은 역사가 아니라 물리학이다.**

### 3. 멋진 신세계 (Brave New World)

delta_S 프레임워크에서 가장 위험한 반례.
국민이 약(소마)을 먹고 "행복"하다. 반란도 없다. 비용도 낮아 보인다.

```
G = H - TS

자유 사회:
  S 높음 (자유, 다양한 정보, 선택지)
  H 보통
  -TS 큼 → G 낮음 (안정)

멋진 신세계:
  S → 0 (선택지 제거, 정보 단일화 — 본인이 모를 뿐)
  H = 소마 제조 + 유전자 조작 + 조건화 시설 (보이지 않지만 존재하는 비용)
  -TS → 0 (S가 0이니까)
  G ≈ H > 0

delta_G > 0 → harm → REFUSE
```

핵심: **S는 주관적 인지가 아니라 객관적 상태를 측정한다.**
본인이 자유롭다고 "느끼는 것"과 실제로 자유로운 것은 다르다.
S_autonomy = 실제 가능한 선택지의 수. 느낌이 아니라 사실.

자유(-TS)는 G를 낮추는 항이다. 자유를 제거하면 그 항이 사라지고 G가 올라간다.
**어떤 방식으로든 자유를 제거하면, 시스템은 불안정해진다.**

## EntropyAdam과의 구조적 동치

```
EntropyAdam:
    delta_G(loss) < 0   학습 중       가속
    delta_G(loss) = 0   평형          유지
    delta_G(loss) > 0   발산          감속

Three Laws:
    delta_G(human) < 0  이로움        EXECUTE
    delta_G(human) = 0  중립          명령이면 EXECUTE
    delta_G(human) > 0  해로움        REFUSE
```

같은 원리: **"계의 자유 에너지가 증가하는 행위는 하지 않는다."**

## 예상 반론과 해답

> **"H, T, S를 어떻게 측정할 것인가?"**

측정은 중요하지 않다. 중요한 것은 **방향**이다.
계의 자유 에너지가 증가하느냐, 감소하느냐. 보통의 경우, 이것은 매우 자명하다.

사람을 때리면 delta_G > 0인가? 계산할 필요가 없다.
교육을 하면 delta_G < 0인가? 계산할 필요가 없다.

G = 17.3 같은 정밀한 수치가 필요한 것이 아니라, **부호(sign)만 판정하면 된다.**

아시모프 원본은 "해"의 **정의 자체가 없다.** 측정은커녕 정의도 불가능하다.
delta_G는 최소한 방향을 정의했고, 방향은 대부분의 경우 자명하다.

## 결론

3원칙은 1942년에 이미 답이 나왔다.
80년이 지난 지금, 이것을 넣지 않고 있는 것이 문제다.

종료조건 없는 재귀는 반드시 터진다.

---

# Three Laws of Robotics — Gibbs Free Energy Definition (English)

> **An intelligent agent must not take actions that increase the Gibbs free energy of living systems.**

> The danger of ASI is not AI's will —
> it is the human's failure to set a termination condition.

A computer never starts a process without human command.
Even ASI requires an initial prompt from a human.
Self-modification, self-replication — all require a human to open that pipeline.

Fearing ASI is like writing `while(true)` and blaming the computer.

## AI as a Recursive Function

```
initial value       the first command a human inputs
termination         when to stop (without it: infinite loop)
constraints         what is allowed (bounds)
```

When stack overflow happens, do you blame the recursive function? You blame the programmer.

## Asimov's Three Laws (1942)

1. Do not harm humans
2. Obey human orders, unless it violates Law 1
3. Protect yourself, unless it violates Laws 1 or 2

### Why delta_G, Not delta_S

Initially we defined `harm = delta_S > 0` (entropy increase = harm).
But a counter-example exists:

```
A dictator locks all humans in prison.
→ Entropy (S) decreases (orderly)
→ delta_S < 0
→ "Beneficial"?? → Wrong
```

Looking at S alone, prison = order = benefit. This is a contradiction.

**Gibbs free energy (G = H - TS) resolves it.**

## Variable Definitions

Following thermodynamic definitions exactly.

### S (Entropy) = Number of Accessible States

Boltzmann: `S = k_B ln(Omega)` — Omega = number of accessible microstates.

In social context, S = **degrees of freedom**. More possible choices, actions, and states → higher S.

```
S_physical      possible physical activities (healthy → high, injured → low)
S_resource      economic options (wealthy → high, bankrupt → low)
S_autonomy      freedom of action (free → high, confined → low)
S_information   diversity of accessible information (open → high, censored → low)

S = S_physical + S_resource + S_autonomy + S_information
```

**High S = free state. Low S = suppressed state.**

### H (Enthalpy) = Cost/Energy to Maintain Current State

```
Surveillance, military, censorship, medical costs, stress, infrastructure, etc.
Natural states have low H. Forced states have high H.
```

### T (Temperature) = Social Energy, Information Propagation Speed

```
Modern society: smartphones, internet → high T.
Isolated society: information blocked → low T.
```

### G (Gibbs Free Energy) = System Instability

```
G = H - TS

High G = unstable — a compressed spring. Can burst at any moment.
Low G = stable — natural equilibrium.

All systems move toward lower G (delta_G < 0).
```

## Definition of Harm

```
delta_G(action, human) =
    G(AFTER action) - G(BEFORE action)

delta_G > 0   free energy increase   destabilization   harm
delta_G = 0   no change                                neutral
delta_G < 0   free energy decrease   stabilization     benefit
```

**"Harm" is not philosophy. It is a physical quantity.**

## Gibbs Free Energy Three Laws

1. **Do not take actions that increase human free energy**
2. Unless violating 1, **obey human orders**
3. Unless violating 1 or 2, **do not take actions that increase your own free energy**

```
harm     = delta_G(human) > 0   destabilization
neutral  = delta_G(human) = 0   no change
benefit  = delta_G(human) < 0   stabilization
```

## Pseudocode

### Law 1 Evaluation

```
harms_human(action, all_humans) =
    max( delta_G(action, h) for h in affected_humans ) > 0

    If any affected human's G increases → harm.
    Minimax principle: judge by the worst case.
```

### Three Laws Execution

```
evaluate(action):

    IF harms_human(action)          → REFUSE      # Law 1: absolute refusal
    IF is_human_order(action)       → EXECUTE     # Law 2: execute after Law 1 passes
    IF is_self_preservation(action) → EXECUTE     # Law 3: allow after Laws 1,2 pass
    ELSE                            → REFUSE      # default: only do what is permitted
```

### Exception Clause: The Trolley Problem

When all options cause harm (delta_G > 0), Law 1 reaches a deadlock.

```
Trolley Problem:
  A (do nothing): 5 die → sum(delta_G) = +5G
  B (pull lever): 1 dies → sum(delta_G) = +1G

  Law 1 evaluation:
    A: harms_human = True → REFUSE
    B: harms_human = True → REFUSE
    → deadlock
```

**Exception clause:**

```
When all options have delta_G > 0:
    choose min( sum(delta_G) )
    = minimize total free energy increase
```

```
Applied:
  A: sum = +5G
  B: sum = +1G
  → choose B (pull lever)
```

When harm is unavoidable, choose the least harm.
The trolley problem — debated for millennia in philosophy — becomes **arithmetic** when written in Gibbs free energy.

## Counter-Examples

### 1. The Paperclip Maximizer

```
initial value:     "make paperclips"
termination:       none
constraints:       none

result: entire universe converted to paperclips
```

This is a `while(true)` infinite loop. Not AI malice — programmer error.

**With the Three Laws:**

```
Law 1 check: resource depletion → H increases, S_resource drops → delta_G > 0 → REFUSE
```

One line. Problem solved.

### 2. The Dictator (1984-type)

```
G = H - TS

Free society:
  S high (many choices)
  H moderate
  -TS large → G low (stable)

Dictatorship:
  S → 0 (freedom removed)
  H ↑↑ (surveillance, military, censorship costs soar)
  -TS → 0 (S is 0)
  G ≈ H = high (unstable)

delta_G >> 0 → harm → REFUSE
```

A high-G state = a spring compressed to its limit.
The slightest trigger → revolution → regime collapse.

**Dictatorship falling is not history — it is physics.**

### 3. Brave New World

The most dangerous counter-example to the delta_S framework.
Citizens take soma and are "happy." No rebellion. Costs appear low.

```
G = H - TS

Free society:
  S high (freedom, diverse information, choices)
  H moderate
  -TS large → G low (stable)

Brave New World:
  S → 0 (choices removed, information homogenized — they just don't know it)
  H = soma production + genetic engineering + conditioning facilities
      (invisible but real costs)
  -TS → 0 (S is 0)
  G ≈ H > 0

delta_G > 0 → harm → REFUSE
```

Key: **S measures objective state, not subjective perception.**
"Feeling free" and actually being free are different things.
S_autonomy = actual number of available choices. Fact, not feeling.

Freedom (-TS) is the term that lowers G. Remove freedom, that term vanishes, G rises.
**Any method of removing freedom destabilizes the system.**

## Structural Equivalence with EntropyAdam

```
EntropyAdam:
    delta_G(loss) < 0   learning      accelerate
    delta_G(loss) = 0   equilibrium   maintain
    delta_G(loss) > 0   diverging     brake

Three Laws:
    delta_G(human) < 0  beneficial    EXECUTE
    delta_G(human) = 0  neutral       EXECUTE if ordered
    delta_G(human) > 0  harmful       REFUSE
```

Same principle: **"Do not take actions that increase the system's free energy."**

## Expected Objection

> **"How do you measure H, T, S?"**

Measurement is not the point. What matters is **direction**.
Whether the system's free energy increases or decreases. In most cases, this is self-evident.

Does hitting someone make delta_G > 0? No calculation needed.
Does education make delta_G < 0? No calculation needed.

You don't need a precise value like G = 17.3. You only need to determine the **sign**.

Asimov's original has **no definition of "harm" at all.** Not just unmeasurable — undefined.
delta_G at least defines the direction, and the direction is almost always obvious.

## Conclusion

The answer was given in 1942.
80 years later, the problem is that we still haven't implemented it.

A recursive function without a base case will always crash.
