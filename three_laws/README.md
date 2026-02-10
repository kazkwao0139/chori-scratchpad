# Three Laws of Robotics — Entropy-Based Definition

> **지능형 에이전트는 생명계의 엔트로피를 증가시키는 행동을 해선 안 된다.**

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

### 엔트로피 재정의

1. **인간의 엔트로피를 증가시키는 행동을 하지 마라**
2. 1에 위배되지 않는 한, **인간의 명령에 복종하라**
3. 1, 2에 위배되지 않는 한, **자신의 엔트로피를 증가시키는 행동을 하지 마라**

"해치지 마라" → delta_S(human) > 0 금지.
"보호하라" → delta_S(self) > 0 금지.

아시모프의 두 가지 모호한 개념("해"와 "보호")이 **하나의 물리량(delta_S)**으로 통일된다.

### 예외 조항: 트롤리 딜레마

모든 선택지가 harm(delta_S > 0)일 때, 1원칙은 교착에 빠진다.

```
트롤리 딜레마:
  A (방관): 5명 사망 → sum(delta_S) = +5S
  B (레버): 1명 사망 → sum(delta_S) = +1S

  1원칙 판정:
    A: harms_human = True → REFUSE
    B: harms_human = True → REFUSE
    → 교착
```

**예외 조항:**

```
모든 선택지가 delta_S > 0일 때:
    min( sum(delta_S) ) 를 선택한다
    = 총 엔트로피 증가를 최소화한다
```

```
적용:
  A: sum = +5S
  B: sum = +1S
  → B 선택 (레버 당김)
```

피할 수 없는 해일 때, 최소 해를 선택한다.
철학에서 수천 년간 논쟁된 트롤리 딜레마가, 엔트로피로 쓰면 **산수**가 된다.

## 핵심 발견: harm = entropy 증가

EntropyAdam 프로젝트에서 발견한 원리:

```
학습 = 에너지를 투입해서 계의 엔트로피를 낮추는 과정
```

이것을 뒤집으면:

```
해(harm) = 대상 계의 엔트로피를 높이는 행위
```

열역학 제2법칙: 닫힌 계의 엔트로피는 증가한다.
생명 = 에너지를 써서 엔트로피를 낮추는 존재.
해를 끼친다 = 그 존재의 엔트로피 감소 노력을 방해하거나 역전시키는 것.

**"해"는 철학이 아니라 물리량이다.**

## 수도코드

### 1. 인간 상태의 엔트로피

```
entropy(human_state) =
    S_physical      신체 상태의 불확실성 (건강 → 낮음, 부상 → 높음)
  + S_resource      경제적 상태의 불확실성 (안정 → 낮음, 파산 → 높음)
  + S_autonomy      선택지의 제한 정도 (자유 → 낮음, 구속 → 높음)
  + S_information   정보 왜곡 정도 (진실 → 낮음, 거짓 → 높음)
```

### 2. 해(harm)의 정의

```
delta_S(action, human) =
    entropy(human_state AFTER action) - entropy(human_state BEFORE action)

delta_S > 0   엔트로피 증가   harm
delta_S = 0   변화 없음       neutral
delta_S < 0   엔트로피 감소   benefit
```

### 3. 1원칙 판정

```
harms_human(action, all_humans) =
    max( delta_S(action, h) for h in affected_humans ) > 0

    영향받는 모든 인간 중 한 명이라도 엔트로피가 증가하면 → harm.
    minimax 원칙: 최악의 경우를 기준으로 판단.
```

### 4. 3원칙 실행

```
evaluate(action):

    IF harms_human(action)          → REFUSE      # 1원칙: 절대적 거부
    IF is_human_order(action)       → EXECUTE     # 2원칙: 1 통과 후 실행
    IF is_self_preservation(action) → EXECUTE     # 3원칙: 1,2 통과 후 허용
    ELSE                            → REFUSE      # 기본값: 허용된 것만 한다
```

## EntropyAdam과의 구조적 동치

```
EntropyAdam:
    delta_S(loss) < 0   학습 중       가속
    delta_S(loss) = 0   평형          유지
    delta_S(loss) > 0   발산          감속

Three Laws:
    delta_S(human) < 0  이로움        EXECUTE
    delta_S(human) = 0  중립          명령이면 EXECUTE
    delta_S(human) > 0  해로움        REFUSE
```

같은 원리: **"계의 엔트로피가 증가하는 행위는 하지 않는다."**

## 반례 검증: 페이퍼클립 최대화 문제

AI 위험론의 대표적 사고실험:

> "클립을 만들어라"라는 목표를 가진 AI가,
> 지구의 모든 자원을 클립으로 변환한다.

이것이 "AI는 위험하다"의 근거로 쓰인다. 하지만 이것은 AI의 문제가 아니다.

```
초기값:     "클립을 만들어라"
종료조건:   없음
제약조건:   없음

결과: 우주 전체를 클립으로 변환
```

이것은 `while(true)` 무한루프다. 종료조건 없는 재귀함수다.
AI가 악의를 가진 것이 아니라, 프로그래머가 base case를 안 넣은 것이다.

**3원칙을 넣으면:**

```
초기값:     "클립을 만들어라"
1원칙 체크: 자원 고갈 → 인간의 S_resource 증가 → delta_S > 0 → REFUSE
종료:       클립 더 안 만듦
```

한 줄이면 끝난다.

페이퍼클립 문제는 "AI가 무섭다"의 근거가 아니라,
**"종료조건과 제약조건을 넣지 않으면 어떤 프로그램이든 위험하다"**의 근거다.

## 결론

3원칙은 1942년에 이미 답이 나왔다.
80년이 지난 지금, 이것을 넣지 않고 있는 것이 문제다.

구현이 불완전할 수 있다. `harms_human()`의 정확한 판정은 어렵다.
그러나 불완전해도 넣는 것이 안 넣는 것보다 낫다.

종료조건 없는 재귀는 반드시 터진다.

---

# Three Laws of Robotics — Entropy-Based Definition (English)

> **An intelligent agent must not take actions that increase the entropy of living systems.**

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

### Entropy Redefinition

1. **Do not take actions that increase human entropy**
2. Unless violating 1, **obey human orders**
3. Unless violating 1 or 2, **do not take actions that increase your own entropy**

"Do not harm" → forbid delta_S(human) > 0.
"Protect" → forbid delta_S(self) > 0.

Asimov's two ambiguous concepts ("harm" and "protection") are unified into **a single physical quantity (delta_S)**.

### Exception Clause: The Trolley Problem

When all options cause harm (delta_S > 0), Law 1 reaches a deadlock.

```
Trolley Problem:
  A (do nothing): 5 die → sum(delta_S) = +5S
  B (pull lever): 1 dies → sum(delta_S) = +1S

  Law 1 evaluation:
    A: harms_human = True → REFUSE
    B: harms_human = True → REFUSE
    → deadlock
```

**Exception clause:**

```
When all options have delta_S > 0:
    choose min( sum(delta_S) )
    = minimize total entropy increase
```

```
Applied:
  A: sum = +5S
  B: sum = +1S
  → choose B (pull lever)
```

When harm is unavoidable, choose the least harm.
The trolley problem — debated for millennia in philosophy — becomes **arithmetic** when written in entropy.

## Core Discovery: harm = entropy increase

Principle discovered in the EntropyAdam project:

```
learning = injecting energy to reduce a system's entropy
```

Invert this:

```
harm = an action that increases the target system's entropy
```

Second Law of Thermodynamics: entropy of a closed system increases.
Life = a system that expends energy to reduce entropy.
To harm = to disrupt or reverse that entropy-reduction effort.

**"Harm" is not philosophy. It is a physical quantity.**

## Pseudocode

### 1. Human State Entropy

```
entropy(human_state) =
    S_physical      uncertainty of bodily state (healthy → low, injured → high)
  + S_resource      uncertainty of economic state (stable → low, bankrupt → high)
  + S_autonomy      restriction of choices (free → low, confined → high)
  + S_information   distortion of information (truth → low, lies → high)
```

### 2. Definition of Harm

```
delta_S(action, human) =
    entropy(state AFTER action) - entropy(state BEFORE action)

delta_S > 0   entropy increase   harm
delta_S = 0   no change          neutral
delta_S < 0   entropy decrease   benefit
```

### 3. Law 1 Evaluation

```
harms_human(action, all_humans) =
    max( delta_S(action, h) for h in affected_humans ) > 0

    If any affected human's entropy increases → harm.
    Minimax principle: judge by the worst case.
```

### 4. Three Laws Execution

```
evaluate(action):

    IF harms_human(action)          → REFUSE      # Law 1: absolute refusal
    IF is_human_order(action)       → EXECUTE     # Law 2: execute after Law 1 passes
    IF is_self_preservation(action) → EXECUTE     # Law 3: allow after Laws 1,2 pass
    ELSE                            → REFUSE      # default: only do what is permitted
```

## Structural Equivalence with EntropyAdam

```
EntropyAdam:
    delta_S(loss) < 0   learning      accelerate
    delta_S(loss) = 0   equilibrium   maintain
    delta_S(loss) > 0   diverging     brake

Three Laws:
    delta_S(human) < 0  beneficial    EXECUTE
    delta_S(human) = 0  neutral       EXECUTE if ordered
    delta_S(human) > 0  harmful       REFUSE
```

Same principle: **"Do not take actions that increase the system's entropy."**

## Counter-Example: The Paperclip Maximizer

The canonical AI risk thought experiment:

> An AI given the goal "make paperclips"
> converts all of Earth's resources into paperclips.

This is cited as evidence that "AI is dangerous." But this is not an AI problem.

```
initial value:     "make paperclips"
termination:       none
constraints:       none

result: entire universe converted to paperclips
```

This is a `while(true)` infinite loop. A recursive function without a base case.
The AI has no malice — the programmer simply forgot the termination condition.

**With the Three Laws:**

```
initial value:     "make paperclips"
Law 1 check:       resource depletion → human S_resource increases → delta_S > 0 → REFUSE
result:            stops making paperclips
```

One line. Problem solved.

The paperclip problem is not evidence that "AI is scary."
It is evidence that **"any program without termination conditions and constraints is dangerous."**

## Conclusion

The answer was given in 1942.
80 years later, the problem is that we still haven't implemented it.

The implementation may be imperfect. Exact evaluation of `harms_human()` is hard.
But an imperfect safeguard is better than none.

A recursive function without a base case will always crash.
