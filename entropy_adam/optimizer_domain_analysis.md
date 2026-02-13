# Optimizer Domain Analysis
## 최적화 문제의 4영역 분류 프레임워크

---

## 동기

"글로벌 옵티마이저를 만들 수 있을까?"라는 질문에서 출발.
각 문제 영역을 분류하고, 어떤 도구가 적합한지 정리한 프레임워크.

---

## 4개의 영역

### 1. 해석적 영역 (Analytic Domain)
- **특징**: 미분 가능, 닫힌 해(closed-form) 존재
- **예시**: 선형 회귀, 간단한 물리 시뮬레이션
- **도구**: 공식. 미분해서 0 되는 점 찾으면 끝.
- **옵티마이저**: 불필요.

### 2. 이산 구조 영역 (Discrete / Algebraic Domain)
- **특징**: 정보는 보존되지만 구조가 이산적.
- **예시**: RSA, 이산로그, 소인수분해
- **도구**: 정수론, 대수학 (Shor, GNFS 등)
- **옵티마이저**: 틀린 도구. landscape가 이산적이라 gradient가 발 디딜 곳이 없음.
  > "자물쇠를 망치로 부수려는 시도"

### 3. 비가역 영역 (Non-injective / Chaotic Domain)
- **특징**: 정보가 소실되는 구조. 역방향 복원 불가.
- **예시**: 해시 역상, TSP, SAT
- **도구**: 몬테카를로(무작위 탐색) 정도.
- **옵티마이저**: 글로벌 미니멈 보장 불가.

### 4. 뉴럴넷 영역 (Deep Learning Domain)
- **특징**: 미분 가능하지만 닫힌 해 없음. 차원이 극도로 높음.
- **핵심**: 글로벌 미니멈 = 과적합. 정답이 아님.
- **도구**: 옵티마이저. 넓은 분지(flat minima)를 찾는 것이 목표.
- **이론적 근거**: 넓은 분지 = 높은 볼츠만 엔트로피 = 일반화 성능.
- **선행 연구**: Hochreiter & Schmidhuber (1997), Keskar et al. (2017)

---

## 왜 범용 글로벌 옵티마이저는 없는가

### 수학적 도구의 한계

| 도구 | 한계 |
|------|------|
| 미분 | 로컬 정보만 (h -> 0 극한) |
| 적분 | 수치적으로는 결국 샘플링 |
| 푸리에 변환 | 전구간 샘플 필요 |
| 위상수학 | 로컬 미니마가 위상적으로 동일 (구분 불가) |
| 해석적 연속 | 해석적 함수에서만 작동 (극히 특수) |

거의 모든 수학적 도구가 연속성에 의존한다.
일반적인 최적화 landscape는 이 조건을 만족하지 않는다.

### 열역학적 하한

- Landauer's principle: 1비트 확정 = kT ln2 에너지
- 분지가 기하급수적이면, 최저를 확정하는 데 필요한 정보량도 비례
- 이 비용은 알고리즘의 영리함으로 제거할 수 없음

### 역설

- 해석적이면 -> 수학으로 직접 풀림 -> 옵티마이저 불필요
- 비해석적이면 -> 수학적 지름길 없음 -> 전수조사 필수

옵티마이저가 필요하면서 동시에 풀 수 있는 영역은
**미분 가능하지만 닫힌 해가 없는 곳** — 영역 4뿐이다.
그리고 거기서는 글로벌 미니멈이 오히려 나쁜 해(과적합)다.

---

## 정리

| 영역 | 올바른 도구 | 옵티마이저 적합성 |
|------|------------|-----------------|
| 해석적 | 공식 | 불필요 |
| 이산 구조 | 정수론/대수학 | 틀린 도구 |
| 비가역 | 없음 | 보장 불가 |
| 뉴럴넷 | 로컬 옵티마이저 | 유일한 적합 영역 |

문제를 먼저 분류하고, 영역에 맞는 도구를 선택해야 한다.

---

---

## References

**Flat minima와 일반화**
- Hochreiter & Schmidhuber, "Flat Minima" (1997) — flat minima가 낮은 과적합과 연결된다는 최초 제안
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (ICLR 2017) — small-batch SGD가 flat minima로, large-batch가 sharp minima로 수렴한다는 실증

**글로벌 미니멈 = 과적합**
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (ICLR 2017, Best Paper) — 뉴럴넷이 랜덤 라벨도 완벽히 외움(training loss = 0). 글로벌 미니멈 도달이 일반화를 보장하지 않음

**Loss landscape 구조**
- Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018) — loss landscape 시각화. flat/sharp minima와 일반화 성능의 상관관계 확인

**열역학적 하한**
- Landauer, "Irreversibility and Heat Generation in the Computing Process" (1961) — 1비트 소거 = kT ln2 에너지. 정보와 열역학의 근본 연결
- Berut et al., "Experimental Verification of Landauer's Principle" (Nature 2012) — Landauer 원리의 최초 실험적 검증

**범용 옵티마이저의 불가능성**
- Wolpert & Macready, "No Free Lunch Theorems for Optimization" (IEEE 1997) — 모든 문제에 대해 평균하면 어떤 최적화 알고리즘도 동일한 성능. 범용 최적 옵티마이저는 존재하지 않음

---

*2026-02-12*

---

P.S. 삽질 몇시간 했는데.. 오히려 안된다는걸 알았어요... ㅋㅋㅋㅋ
프레임워크만 남기고 과정을 공개하지 않는건, 공개하고 싶지 않은 내용이 있어서입니다.

글로벌 옵티마? 굳이 찾아야 하나?
만약 자연어에 글로벌 옵티마가 존재한다 해보자. 그것은 정말 자연어일까?
내가 이렇게 자연어(한국어)로 이걸 쓰고 있다는거 자체가,
자연어는 여러개의 로컬 옵티마(개인의 개성, 말투, 논리)가 존재한단 증거다.
글로벌 옵티마에 도달하면 "날씨 좋아?" 라고 물어봤는데
침묵만 하거나, "좋아." 한마디밖에 못하는 것과 뭐가 다르지...?
