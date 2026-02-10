# EntropyAdam

> loss trajectory를 "문자열"처럼 읽는 옵티마이저.
> **Worst case = Adam.**

Shannon bigram entropy로 loss의 예측 가능성을 실시간 측정해서,
수렴 중이면 가속하고, 불안정하면 Adam처럼 신중하게 간다.

```
H 높음 (예측 불가) → multiplier = 1.0 → 순수 Adam. 손해 제로.
H 낮음 (예측 가능) → multiplier > 1.0 → Adam + 가속. 이득만 존재.
```

Adam이 10년간 왕좌를 지킨 이유는 "이기기 어려워서"가 아니라 "지지 않아서"다.
EntropyAdam은 그 "지지 않는 성질"을 그대로 포함한 상위호환이다.

나는 가능성과 경향성만 확인했다. 실제 대규모 검증은 내 몫이 아니다.

## 발견 경위

셰익스피어 *Hamlet*의 캐릭터별 엔트로피를 측정하는 실험에서 출발했다.

| 캐릭터 | 대사량 | bits/char | 수렴 여부 |
|--------|--------|-----------|----------|
| Hamlet | 56,000 chars | 3.516 (11위) | **수렴** |
| Ghost | 3,612 chars | 3.598 (1위) | 미수렴 |

**대수의 법칙**: 관찰이 길면 엔트로피가 수렴한다 = 패턴이 예측 가능해진다.

이 원리를 뒤집었다:
- **"loss trajectory가 예측 가능한가?"** 를 bigram entropy로 측정
- 예측 가능하면 → 수렴 중 → **가속**
- 예측 불가능하면 → 탐색 중 → Adam 기본 속도

## 알고리즘

```
매 스텝:
  1. loss 기록
  2. loss 변화량(delta)을 5단계로 양자화: 급하강/완하강/정체/완상승/급상승
  3. 양자화된 시퀀스의 bigram 전이행렬 구축
  4. 조건부 엔트로피 H(state_t | state_{t-1}) 계산
  5. H 낮으면 → LR multiplier ↑ (최대 5x)
     H 높으면 → LR multiplier → 1.0x (순수 Adam)
  6. 비대칭 EMA: 가속은 느리게(alpha=0.05), 감속은 빠르게(alpha=0.3)
```

### 핵심 수식

```
H_norm = H(state_t | state_{t-1}) / log₂(5)

multiplier = 1 + (max_boost - 1) × (1 - H_norm)²

lr_effective = lr_base × multiplier
```

### CV(변동계수)와의 차이

```
loss = [2.1, 2.0, 2.1, 2.0, 2.1, 2.0]
  → CV 높음 → 기존 방식: 감속
  → bigram H 낮음 → EntropyAdam: 가속 (패턴이 있으니까)

loss = [2.1, 1.8, 2.3, 1.9, 2.4, 1.7]
  → CV 높음 → 기존 방식: 감속
  → bigram H 높음 → EntropyAdam: 감속 (진짜 랜덤이니까)
```

CV는 "얼마나 흔들리나"만 본다. 엔트로피는 "흔들림에 패턴이 있는가"를 본다.

## 벤치마크

### 1. 토이 함수 (numpy)

| 함수 | Adam | EntropyAdam | 배수 |
|------|------|-------------|------|
| Rosenbrock 10D | 97,080 | 3,416 | **28.4x** |
| Rosenbrock 30D | 327,633 | 11,705 | **28.0x** |
| Rastrigin 10D | 81.59 | 81.59 | 1.0x (동률) |
| Ackley 10D | 8.24 | 8.24 | 1.0x (동률) |

- Rosenbrock (좁은 계곡, 명확한 하강): **28x 도륙**
- Rastrigin/Ackley (local minima): 동률 = 가속하면 안 되는 곳에서 안전하게 fallback

### 2. LLM (Tiny Transformer on Shakespeare, CUDA RTX 4070 Ti)

| Batch Size | Adam val | EntropyAdam val | 개선 |
|-----------|---------|----------------|------|
| 32 | 2.1174 | 2.0874 | **1.4%** |
| 64 | 2.0755 | 1.9930 | **4.0%** |
| 128 | 1.9956 | 1.9396 | **2.8%** |
| 512 | 1.9241 | 1.8644 | **3.1%** |

**전승. 지는 케이스 없음.**

### 3. Homogeneous vs Heterogeneous 데이터

| 데이터 | Adam val | EntropyAdam val | 개선 |
|--------|---------|----------------|------|
| Hamlet 대사만 (57K, homogeneous) | 1.9334 | 1.9237 | 0.5% |
| Shakespeare 전체 (1M, heterogeneous) | 2.0755 | 1.9930 | 4.0% |

Hamlet only에서 train loss 차이는 5.2%로 더 크지만, 57K chars에 446K params → 오버피팅.
homogeneous 데이터에서 가속은 더 잘 걸리지만, 데이터가 충분해야 일반화된다.

## 노이즈의 본질

실험 중 발견한 것:

> **노이즈 = 이질적 분포의 혼합.**

미니배치의 loss가 출렁이는 이유는 샘플링 분산이 아니라,
한 배치 안에 "햄릿의 대사"와 "고스트의 대사"와 "지문"이 섞여 있기 때문이다.

배치가 커지면 → 대수의 법칙 → 매 배치가 "평균적 영어"에 수렴 → 노이즈 감소 → 엔트로피 하락 → 가속 가능.

## 스케일링 전망

```
현재 실험                     실제 LLM 학습
──────────────────────────────────────────────
batch: ~8K tokens             batch: 1M+ tokens
steps: 2,000                  steps: 100K+
data: 1M chars                data: 조 단위 tokens
model: 446K params            model: 수십억 params
multiplier: 1.1~1.3x          multiplier: ? (4~5x 가능)
개선: 1~4%                    개선: ?
```

배치 사이즈가 대수의 법칙을 따르므로, 실제 LLM 학습에서는 loss trajectory의 entropy가 현저히 낮아질 것으로 예상.
현재 1.1x에서 4% 개선이면, 4x 가속 시 수십% 이상의 학습 효율 개선 가능.

## 사용법

### numpy (토이 함수)

```python
from entropy_optimizer import EntropyAdam, rosenbrock

opt = EntropyAdam(rosenbrock, x0, lr=0.01, max_boost=5.0)
best = opt.run(steps=500)
```

### PyTorch (LLM)

```python
from entropy_adam_torch import EntropyAdam

optimizer = EntropyAdam(model.parameters(), lr=3e-4, max_boost=5.0)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.report_loss(loss)   # 이것만 추가
    optimizer.step()
    optimizer.zero_grad()
```

## 파일

| 파일 | 설명 |
|------|------|
| `entropy_optimizer.py` | numpy 구현 + 토이 함수 벤치마크 |
| `entropy_adam_torch.py` | PyTorch 구현 + LLM 비교 스크립트 |

## 원리 한 줄 요약

> "관찰이 짧으면 혼돈이고, 관찰이 길면 질서다."
> loss가 예측 가능해지는 순간을 포착해서 밟는다.

---

# EntropyAdam (English)

> An optimizer that reads loss trajectory like a string.
> **Worst case = Adam.**

Measures predictability of loss via Shannon bigram entropy.
Accelerates when converging, stays cautious (Adam) when unstable.

```
High H (unpredictable) → multiplier = 1.0 → pure Adam. Zero downside.
Low H  (predictable)   → multiplier > 1.0 → Adam + acceleration. Only upside.
```

Adam held the throne for a decade not because it's hard to beat, but because it never loses.
EntropyAdam is a strict superset — it inherits "never loses" and adds "sometimes wins."

I only verified the possibility and the trend. Large-scale validation is not my job.

## Origin

From a character entropy experiment on Shakespeare's *Hamlet*:
- Hamlet (56K chars): entropy **converged** to 3.516 bits/char — consistent madness
- Ghost (3.6K chars): entropy **unconverged** at 3.598 — not enough data

**Law of Large Numbers**: longer observation → entropy converges → pattern becomes predictable.

Flipped this into an optimizer signal:
- Loss trajectory predictable → converging → **accelerate**
- Loss trajectory unpredictable → exploring → stay at Adam speed

## Algorithm

```
Each step:
  1. Record loss
  2. Quantize loss delta into 5 bins: big_drop/small_drop/flat/small_rise/big_rise
  3. Build bigram transition matrix of quantized states
  4. Compute conditional entropy H(state_t | state_{t-1})
  5. Low H → LR multiplier ↑ (up to 5x)
     High H → multiplier → 1.0x (pure Adam)
  6. Asymmetric EMA: accelerate slowly (0.05), brake fast (0.3)
```

### Key Difference from CV

CV measures "how much does it shake." Entropy measures "is there a pattern in the shaking."

Oscillating loss `[2.1, 2.0, 2.1, 2.0]` has high CV but low entropy — it's predictable. EntropyAdam correctly accelerates. CV-based methods incorrectly slow down.

## Results

### Toy Functions
- Rosenbrock 10D: **28x** faster than Adam
- Rastrigin/Ackley: tied (safe fallback when acceleration is wrong)

### LLM (Tiny Transformer, RTX 4070 Ti)

| Batch Size | Improvement |
|-----------|-------------|
| 32 | 1.4% |
| 64 | 4.0% |
| 128 | 2.8% |
| 512 | 3.1% |

**Undefeated across all conditions.**

### On Noise

> Noise = mixture of heterogeneous distributions.

Mini-batch loss fluctuates not from sampling variance, but because each batch mixes different patterns (Hamlet's lines + Ghost's lines + stage directions). Larger batches → Law of Large Numbers → each batch converges to "average English" → less noise → lower entropy → more acceleration.

## Scaling

At LLM scale (batch 1M+ tokens, 100K+ steps), loss trajectories should be near-deterministic per batch due to LLN. The entropy signal would be much stronger, potentially pushing multipliers from current 1.1x to 4-5x range.

> "Brief observation: chaos. Extended observation: order."
> EntropyAdam catches the moment loss becomes predictable, and floors it.
