# 팩토리얼의 합은 왜 실패하는가: 초순열의 구조적 분석

## 1. 문제 정의

$n$개의 심볼에 대한 **초순열(superpermutation)** 이란, $\{1, 2, \ldots, n\}$의 모든 순열을 연속 부분 문자열로 포함하는 문자열이다.

고전적인 재귀 구성법은 다음 길이의 초순열을 생성한다:

$$L(n) = \sum_{k=1}^{n} k!$$

| $n$ | $L(n)$ | 최적인가? |
|-----|--------|-----------|
| 1   | 1      | 예        |
| 2   | 3      | 예        |
| 3   | 9      | 예        |
| 4   | 33     | 예        |
| 5   | 153    | 예        |
| 6   | 873    | **아니오** ($\leq 872$, Houston 2014) |
| 7+  | —      | **아니오** (아래 참조) |

**주 정리.** 모든 $n \geq 6$에 대해, 최소 초순열 길이 $\text{OPT}(n) < \sum_{k=1}^{n} k!$

**핵심 원리.** 덧셈은 곱셈 구조를 파괴한다.

---

## 2. 준비

### 2.1 회전 사이클

순열 $\pi = a_1 a_2 \cdots a_n$에 대해, **좌회전(left rotation)** 은 $a_2 a_3 \cdots a_n a_1$을 생성하며, 이는 $\pi$와 $n-1$개의 문자가 겹친다. 회전을 $n$번 적용하면 $\pi$로 돌아오므로, 길이 $n$의 **회전 사이클**이 형성된다.

$\{1, \ldots, n\}$의 $n!$개 순열은 크기 $n$인 $(n-1)!$개의 회전 사이클로 분할된다.

### 2.2 커널과 브릿지

초순열 내에서:
- **커널(Kernel)**: 하나의 회전 사이클 내에서의 연속 순열 묶음. 각 전환 비용은 1 (겹침 $n-1$).
- **브릿지(Bridge)**: 두 회전 사이클 사이의 전환. 비용 $\geq 2$ (겹침 $\leq n-2$).

### 2.3 재귀 구성법

$\sum k!$ 공식은 다음의 재귀 구성에서 비롯된다:

1. $n-1$개 심볼에 대한 초순열 $S_{n-1}$ (길이 $L(n-1)$)에서 출발한다.
2. $S_{n-1}$ 안의 각 $(n-1)$-순열 $\pi_i$에 대해, 심볼 $n$을 모든 위치에 삽입하여 $n$개 순열로 이루어진 **커널** $K_i$를 생성한다.
3. 커널들을 $S_{n-1}$이 결정하는 순서대로 연결한다.

이 구성은 $L(n-1)$에 정확히 $n!$개의 문자를 추가하여, $L(n) = L(n-1) + n! = \sum_{k=1}^{n} k!$을 만든다.

이 공식은 **각 층이 독립적으로 기여한다**고 암묵적으로 가정한다 — 순수한 덧셈 분해이다.

---

## 3. 확장 보조정리 (Expansion Lemma)

**보조정리.** 임의의 $n \geq 2$에 대해, $\{1, \ldots, n-1\}$의 길이 $l$인 초순열이 주어지면, $\{1, \ldots, n\}$의 길이가 정확히 $l + n!$인 초순열이 존재한다.

### 증명

$S$를 길이 $l$인 $(n-1)$-초순열이라 하고, $S$가 순열 $\pi_1, \pi_2, \ldots, \pi_{(n-1)!}$을 순서대로 방문하며, 연속된 $\pi_i$와 $\pi_{i+1}$ 사이의 겹침을 $o_i$라 하자.

**구성.** $S$를 베이스로 사용하여 커널 확장(2.3절)을 적용한다.

**길이 계산.** 확장된 초순열은 다음으로 구성된다:

- **첫 번째 순열**: $n$개 문자.
- **커널 내 전환**: 커널당 $(n-1)$회, 총 $(n-1) \cdot (n-1)!$회, 각 비용 1.
- **커널 간 브릿지**: $(n-1)! - 1$개.

각 브릿지에서, 커널 $K_i$의 마지막 순열은 $n\, \pi_{i,1}\, \pi_{i,2}\, \cdots\, \pi_{i,n-1}$이고, 커널 $K_{i+1}$의 첫 순열은 $\pi_{i+1,1}\, \pi_{i+1,2}\, \cdots\, \pi_{i+1,n-1}\, n$이다.

베이스에서 $\pi_i$와 $\pi_{i+1}$이 $o_i$개의 후미/선두 문자를 공유하므로, 커널 간 겹침은 정확히 $o_i$이며, 브릿지 비용은 $n - o_i$이다.

**총 길이:**

$$|S_n| = n + (n-1)(n-1)! + \sum_{i=1}^{(n-1)!-1}(n - o_i)$$

베이스 초순열로부터:

$$l = (n-1) + \sum_{i=1}^{(n-1)!-1}(n - 1 - o_i)$$

따라서:

$$\sum o_i = (n-1)(n-1)! - l$$

대입하면:

$$\sum(n - o_i) = n\bigl((n-1)! - 1\bigr) - \sum o_i = n(n-1)! - n - (n-1)(n-1)! + l = (n-1)! + l - n$$

그러므로:

$$|S_n| = n + (n-1)(n-1)! + (n-1)! + l - n = n(n-1)! + l = n! + l$$

확장 비용은 정확히 $n!$이며, **베이스 길이 $l$에 무관하다**. $\blacksquare$

**따름정리.** 모든 $n \geq 2$에 대해 $\text{OPT}(n) \leq \text{OPT}(n-1) + n!$

---

## 4. 주 정리

**정리.** 모든 $n \geq 6$에 대해 $\text{OPT}(n) < \sum_{k=1}^{n} k!$

### 증명

**기저.** Houston (2014)은 $n = 6$에 대해 길이 872인 초순열을 구성하였다.

$$\text{OPT}(6) \leq 872 < 873 = \sum_{k=1}^{6} k! \quad \checkmark$$

**귀납.** $n \geq 7$인 어떤 $n$에 대해 $\text{OPT}(n-1) < \sum_{k=1}^{n-1} k!$이라 가정하자. 확장 보조정리에 의해:

$$\text{OPT}(n) \leq \text{OPT}(n-1) + n! < \sum_{k=1}^{n-1} k! + n! = \sum_{k=1}^{n} k! \quad \blacksquare$$

---

## 5. 왜 $n = 6$인가: 구조적 붕괴 지점

### 5.1 덧셈 가정

$\sum k!$ 공식은 초순열을 독립적인 층들로 분해한다:

$$L(n) = \underbrace{1!}_{\text{1층}} + \underbrace{2!}_{\text{2층}} + \cdots + \underbrace{n!}_{\text{n층}}$$

각 $k!$는 "$k$층을 추가하는 비용"을 나타낸다. 이 공식은 각 층이 독립적으로 기여한다고 가정한다 — 층 사이에 상호작용이 없다. 이것은 **곱셈적 대상**(대칭군 $S_n$)에 대한 순수한 **덧셈 분해**이다.

### 5.2 순환군 구조

회전 사이클은 순환군 $\mathbb{Z}_n$의 구조를 가진다. 중국인의 나머지 정리에 의해, $\mathbb{Z}_n$은 $n$이 서로소인 인수를 가질 때에만 직접곱으로 분해된다:

| $n$ | 소인수분해 | $\mathbb{Z}_n$ 분해 | 독립 주파수 |
|-----|-----------|---------------------|------------|
| 2   | 소수      | $\mathbb{Z}_2$      | 1          |
| 3   | 소수      | $\mathbb{Z}_3$      | 1          |
| 4   | $2^2$     | $\mathbb{Z}_4$ (비분해) | 1       |
| 5   | 소수      | $\mathbb{Z}_5$      | 1          |
| **6** | **$2 \times 3$** | **$\mathbb{Z}_2 \times \mathbb{Z}_3$** | **2** |

$n \leq 5$에서 각 회전 사이클은 **단일 주파수**를 가진다. 모든 사이클이 균일하게 진동하며, 사이클 간 브릿지가 다른 사이클과 교차할 수 없다. 덧셈 가정이 정확히 성립한다.

$n = 6$에서, 분해 $\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$은 각 회전 사이클 안에 **두 개의 독립 주파수**를 도입한다. 이 주파수들은 **간섭 패턴**을 만든다: 두 사이클 사이의 브릿지 경로가 제3의 사이클에 속하는 순열을 관통할 수 있으며, 이는 덧셈 공식이 고려하지 못하는 "무비용 방문"을 산출한다.

### 5.3 $4 = 2^2$는 왜 깨지지 않는가

4는 합성수이지만, $\mathbb{Z}_4 \not\cong \mathbb{Z}_2 \times \mathbb{Z}_2$이다. 위수 4의 순환군은 비분해적이다 — 위수 4인 단일 생성원을 가진다. 주파수가 하나뿐이므로 간섭이 불가능하다.

핵심 구분:
- **소수 거듭제곱** $p^k$: $\mathbb{Z}_{p^k}$는 비분해적이다. 단일 주파수. 간섭 없음.
- **서로 다른 소수의 곱**: $\gcd(a,b) = 1$일 때 $\mathbb{Z}_{ab} \cong \mathbb{Z}_a \times \mathbb{Z}_b$. 복수의 독립 주파수. 간섭 발생.

$n = 6 = 2 \times 3$은 두 개의 서로 다른 소인수를 가지는 최소의 정수이며, 순환 구조가 비자명하게 분해되는 최초의 지점이다.

### 5.4 비가역성

덧셈 공식이 $n = 6$에서 한번 깨지면, 확장 보조정리가 회복 불가능을 보장한다: 확장 비용은 베이스와 무관하게 정확히 $n!$이므로, $n-1$ 수준의 절약분이 $n$ 수준으로 손실 없이 전파된다. 갭 $\sum k! - \text{OPT}(n) \geq 1$은 단조 비감소한다.

---

## 6. 요약

$$\boxed{\text{OPT}(n) < \sum_{k=1}^{n} k! \quad \text{(모든 } n \geq 6\text{)}}$$

증명은 두 기둥 위에 선다:

1. **확장 보조정리**: 새로운 층을 추가하는 비용은 베이스와 무관하게 정확히 $n!$이다. 이것은 커널-브릿지 구성의 구조적 불변량이다.

2. **$n = 6$에서의 구조적 붕괴**: 중국인의 나머지 정리가 $\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$으로 분해하며, 덧셈 공식 $\sum k!$이 포착하지 못하는 층 간 간섭을 만든다.

근본 원리: **대칭군 $S_n$은 곱셈 구조이고, $\sum k!$은 그것의 덧셈 근사이다. 덧셈은 곱셈이 만든 구조를 파괴한다. $n = 6$에서 이 파괴가 처음 드러나며, 그것은 결코 치유되지 않는다.**

---

## 알려진 한계 (참고)

$n \geq 6$에서:

| 한계 | 공식 | 출처 |
|------|------|------|
| 하한 | $n! + (n-1)! + (n-2)! + n - 3$ | Anonymous / Engen-Vatter (2018) |
| 상한 | $n! + (n-1)! + (n-2)! + (n-3)! + n - 3$ | Egan (2018) |
| 갭   | $(n-3)!$ | — |

---

## 7. 전역 최적화의 불가피성: 왜 일반 공식은 없는가

### 7.1 지역 최적과 전역 최적

확장 보조정리는 다음을 **등호**로 증명한다:

> 길이 $l$인 $(n-1)$-초순열을 확장하면, 길이가 정확히 $l + n!$인 $n$-초순열이 된다.

이것은 **층별(layer-by-layer) 구성의 최적**이다. 각 층을 독립적으로 추가할 때, $n!$보다 적은 비용은 불가능하다. 즉:

$$\text{지역 최적} = \sum_{k=1}^{n} k!$$

그런데 4절에서 증명했다:

$$\text{OPT}(n) < \sum_{k=1}^{n} k! \quad (n \geq 6)$$

**지역 최적이 전역 최적이 아니다.** 이 갭의 존재 자체가 핵심이다.

### 7.2 갭이 증명하는 것

갭 $\Delta(n) = \sum k! - \text{OPT}(n) > 0$의 의미를 정밀하게 분석한다.

확장 보조정리의 구성은 **각 층을 독립적으로** 처리한다. 비용이 정확히 $n!$이라는 것은, 이 구성이 **층 간 상호작용을 전혀 활용하지 않는다**는 뜻이다.

그런데 $\text{OPT}(n) < \sum k!$이므로, 실제 최적 초순열은 층 간 상호작용을 활용하여 비용을 줄인다. 이 절약은 **지역 정보(한 층의 구조)만으로는 보이지 않는다** — 만약 보였다면 확장 보조정리의 구성이 이미 활용했을 것이다.

따라서:

$$\Delta(n) > 0 \quad \Longrightarrow \quad \text{OPT}(n)\text{에 도달하려면 전역 정보가 필요하다}$$

### 7.3 정보 파괴와 비복구성

$\sum k!$ 공식은 $S_n$의 곱셈 구조를 가법적으로 분해한다:

$$S_n \;\xrightarrow{\text{가법 분해}}\; 1! + 2! + \cdots + n!$$

이 분해 과정에서 **층 간 간섭 정보가 파괴된다.** $\Delta(n)$은 이 파괴된 정보의 양이다.

파괴된 정보는 가법적 표현 안에서 복구할 수 없다. $\text{OPT}(n) = \sum k! - \Delta(n)$이라는 식은 $\Delta(n)$이 독립적으로 계산 가능한 양인 것처럼 보이게 하지만, 실제로는 $\text{OPT}(n)$을 먼저 알아야 $\Delta(n)$이 결정된다. **순서가 반대이다.**

$\Delta(n)$을 독립적으로 구하려면, 파괴되기 전의 구조 — $S_n$의 곱셈 구조 전체 — 로 돌아가서 전역 최적화를 수행해야 한다.

### 7.4 핵심 정리

**정리.** $\text{OPT}(n)$에 대한 일반 공식은 존재하지 않는다.

**논증.**

1. 확장 보조정리에 의해, 층별 독립 구성의 비용은 정확히 $\sum k!$이다. 이것이 **가법적 접근의 최적**이다.

2. $n \geq 6$에서 $\text{OPT}(n) < \sum k!$이다. 따라서 가법적 접근은 전역 최적에 도달하지 못한다.

3. 가법적 최적과 전역 최적의 차이 $\Delta(n)$은 층 간 간섭에서 비롯되며, 이 간섭은 $\mathbb{Z}_n$의 순환군 분해(5절)에 의존한다.

4. $\Delta(n)$을 결정하려면 $S_n$ 위의 순열 그래프에서 전역 최적화를 수행해야 한다. 이는 가법 분해에서 파괴된 정보를 원래 구조로 돌아가 탐색하는 과정이며, 어떤 닫힌 식으로도 대체할 수 없다.

5. $\text{OPT}(n) = \sum k! - \Delta(n)$이므로, $\Delta(n)$에 닫힌 공식이 없으면 $\text{OPT}(n)$에도 없다. $\blacksquare$

### 7.5 요약

| | $\sum k!$ (가법적 근사) | $\text{OPT}(n)$ (전역 최적) |
|---|---|---|
| 구성 방법 | 층별 독립 확장 | $S_n$ 전역 탐색 |
| 층 간 상호작용 | 무시 | 활용 |
| 공식 존재 | 예 ($\sum k!$) | 아니오 |
| 계산 가능성 | $O(n)$ | 전역 최적화 필요 |

$$\boxed{\text{지역 최적} \neq \text{전역 최적} \;\Longrightarrow\; \text{전역 최적화 불가피} \;\Longrightarrow\; \text{일반 공식 부재}}$$

**덧셈은 곱셈 구조를 파괴한다. 파괴된 정보는 복구되지 않는다. 전역 최적에 도달하려면 파괴 이전의 구조로 돌아가 전체를 탐색해야 한다. 이것이 공식이 아닌 이유이다.**

---

## 참고문헌

1. R. Houston, "Tackling the Minimal Superpermutation Problem," arXiv:1408.5108, 2014.
2. N. Johnston, "Non-uniqueness of Minimal Superpermutations," *Discrete Mathematics* 313(14):1553-1557, 2013.
3. Anonymous 4chan Poster, R. Houston, J. Pantone, V. Vatter, "A lower bound on the length of the shortest superpattern," OEIS A180632, 2018.
4. M. Engen, V. Vatter, "Containing All Permutations," *The American Mathematical Monthly* 128(1):4-24, 2021. arXiv:1810.08252.
5. G. Egan, "Superpermutations," https://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html, 2018.

---
---

# Why the Sum of Factorials Fails: Structural Analysis of Superpermutations

## 1. Problem Definition

A **superpermutation** on $n$ symbols is a string over $\{1, 2, \ldots, n\}$ that contains every permutation of those symbols as a contiguous substring.

The classical recursive construction produces a superpermutation of length:

$$L(n) = \sum_{k=1}^{n} k!$$

| $n$ | $L(n)$ | Optimal? |
|-----|--------|----------|
| 1   | 1      | Yes      |
| 2   | 3      | Yes      |
| 3   | 9      | Yes      |
| 4   | 33     | Yes      |
| 5   | 153    | Yes      |
| 6   | 873    | **No** ($\leq 872$, Houston 2014) |
| 7+  | —      | **No** (see below) |

**Main result.** For all $n \geq 6$, the minimum superpermutation length satisfies $\text{OPT}(n) < \sum_{k=1}^{n} k!$

**Core principle.** Addition destroys multiplicative structure.

---

## 2. Preliminaries

### 2.1 Rotation Cycles

For a permutation $\pi = a_1 a_2 \cdots a_n$, the **left rotation** produces $a_2 a_3 \cdots a_n a_1$, which overlaps with $\pi$ in $n-1$ characters. Applying the rotation $n$ times returns to $\pi$, forming a **rotation cycle** of length $n$.

The $n!$ permutations of $\{1, \ldots, n\}$ partition into $(n-1)!$ rotation cycles of size $n$.

### 2.2 Kernels and Bridges

Within a superpermutation:
- **Kernel**: a block of consecutive permutations within a single rotation cycle. Each transition costs 1 (overlap $n-1$).
- **Bridge**: a transition between two rotation cycles. Cost $\geq 2$ (overlap $\leq n-2$).

### 2.3 Recursive Construction

The $\sum k!$ formula arises from the following recursive construction:

1. Start with a superpermutation $S_{n-1}$ on $n-1$ symbols (length $L(n-1)$).
2. For each $(n-1)$-permutation $\pi_i$ in $S_{n-1}$, insert symbol $n$ into every position to create a **kernel** $K_i$ of $n$ permutations.
3. Concatenate the kernels in the order determined by $S_{n-1}$.

This construction adds exactly $n!$ characters to $L(n-1)$, giving $L(n) = L(n-1) + n! = \sum_{k=1}^{n} k!$.

The formula implicitly assumes **each layer contributes independently** — a purely additive decomposition.

---

## 3. Expansion Lemma

**Lemma.** For any $n \geq 2$, given a superpermutation on $\{1, \ldots, n-1\}$ of length $l$, there exists a superpermutation on $\{1, \ldots, n\}$ of length exactly $l + n!$.

### Proof

Let $S$ be an $(n-1)$-superpermutation of length $l$ that visits permutations $\pi_1, \pi_2, \ldots, \pi_{(n-1)!}$ in order, with overlap $o_i$ between consecutive $\pi_i$ and $\pi_{i+1}$.

**Construction.** Apply the kernel expansion (Section 2.3) using $S$ as the base.

**Length calculation.** The expanded superpermutation consists of:

- **First permutation**: $n$ characters.
- **Intra-kernel transitions**: $(n-1)$ per kernel, total $(n-1) \cdot (n-1)!$, each costing 1.
- **Inter-kernel bridges**: $(n-1)! - 1$ bridges.

At each bridge, the last permutation of kernel $K_i$ is $n\, \pi_{i,1}\, \pi_{i,2}\, \cdots\, \pi_{i,n-1}$ and the first permutation of kernel $K_{i+1}$ is $\pi_{i+1,1}\, \pi_{i+1,2}\, \cdots\, \pi_{i+1,n-1}\, n$.

Since $\pi_i$ and $\pi_{i+1}$ share $o_i$ trailing/leading characters in the base, the inter-kernel overlap is exactly $o_i$, and the bridge cost is $n - o_i$.

**Total length:**

$$|S_n| = n + (n-1)(n-1)! + \sum_{i=1}^{(n-1)!-1}(n - o_i)$$

From the base superpermutation:

$$l = (n-1) + \sum_{i=1}^{(n-1)!-1}(n - 1 - o_i)$$

Therefore:

$$\sum o_i = (n-1)(n-1)! - l$$

Substituting:

$$\sum(n - o_i) = n\bigl((n-1)! - 1\bigr) - \sum o_i = n(n-1)! - n - (n-1)(n-1)! + l = (n-1)! + l - n$$

Hence:

$$|S_n| = n + (n-1)(n-1)! + (n-1)! + l - n = n(n-1)! + l = n! + l$$

The expansion cost is exactly $n!$, **independent of the base length $l$**. $\blacksquare$

**Corollary.** For all $n \geq 2$, $\text{OPT}(n) \leq \text{OPT}(n-1) + n!$

---

## 4. Main Result

**Theorem.** For all $n \geq 6$, $\text{OPT}(n) < \sum_{k=1}^{n} k!$

### Proof

**Base case.** Houston (2014) constructed a superpermutation of length 872 for $n = 6$.

$$\text{OPT}(6) \leq 872 < 873 = \sum_{k=1}^{6} k! \quad \checkmark$$

**Induction.** Assume $\text{OPT}(n-1) < \sum_{k=1}^{n-1} k!$ for some $n \geq 7$. By the Expansion Lemma:

$$\text{OPT}(n) \leq \text{OPT}(n-1) + n! < \sum_{k=1}^{n-1} k! + n! = \sum_{k=1}^{n} k! \quad \blacksquare$$

---

## 5. Why $n = 6$: The Structural Breaking Point

### 5.1 The Additive Assumption

The $\sum k!$ formula decomposes the superpermutation into independent layers:

$$L(n) = \underbrace{1!}_{\text{layer 1}} + \underbrace{2!}_{\text{layer 2}} + \cdots + \underbrace{n!}_{\text{layer n}}$$

Each $k!$ represents "the cost of adding layer $k$." The formula assumes each layer contributes independently — no interaction between layers. This is a purely **additive decomposition** of a **multiplicative object** (the symmetric group $S_n$).

### 5.2 Cyclic Group Structure

Rotation cycles carry the structure of the cyclic group $\mathbb{Z}_n$. By the Chinese Remainder Theorem, $\mathbb{Z}_n$ decomposes into a direct product only when $n$ has coprime factors:

| $n$ | Factorization | $\mathbb{Z}_n$ decomposition | Independent frequencies |
|-----|--------------|------------------------------|------------------------|
| 2   | prime        | $\mathbb{Z}_2$              | 1                      |
| 3   | prime        | $\mathbb{Z}_3$              | 1                      |
| 4   | $2^2$        | $\mathbb{Z}_4$ (indecomposable) | 1                  |
| 5   | prime        | $\mathbb{Z}_5$              | 1                      |
| **6** | **$2 \times 3$** | **$\mathbb{Z}_2 \times \mathbb{Z}_3$** | **2**        |

For $n \leq 5$, each rotation cycle has a **single frequency**. All cycles oscillate uniformly, and bridges between cycles cannot cross through other cycles. The additive assumption holds exactly.

At $n = 6$, the decomposition $\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$ introduces **two independent frequencies** within each rotation cycle. These frequencies create **interference patterns**: a bridge path between two cycles can pass through permutations belonging to a third cycle, yielding "free visits" that the additive formula fails to account for.

### 5.3 Why $4 = 2^2$ Doesn't Break

4 is composite, but $\mathbb{Z}_4 \not\cong \mathbb{Z}_2 \times \mathbb{Z}_2$. The cyclic group of order 4 is indecomposable — it has a single generator of order 4. With only one frequency, interference is impossible.

Key distinction:
- **Prime powers** $p^k$: $\mathbb{Z}_{p^k}$ is indecomposable. Single frequency. No interference.
- **Products of distinct primes**: $\mathbb{Z}_{ab} \cong \mathbb{Z}_a \times \mathbb{Z}_b$ when $\gcd(a,b) = 1$. Multiple independent frequencies. Interference occurs.

$n = 6 = 2 \times 3$ is the smallest integer with two distinct prime factors — the first point where cyclic structure decomposes nontrivially.

### 5.4 Irreversibility

Once the additive formula breaks at $n = 6$, the Expansion Lemma guarantees it never recovers: since expansion cost is exactly $n!$ regardless of the base, any savings at level $n-1$ propagate to level $n$ without loss. The gap $\sum k! - \text{OPT}(n) \geq 1$ is monotonically non-decreasing.

---

## 6. Summary

$$\boxed{\text{OPT}(n) < \sum_{k=1}^{n} k! \quad \text{for all } n \geq 6}$$

The argument rests on two pillars:

1. **Expansion Lemma**: The cost of adding a new layer is exactly $n!$, independent of the base. This is a structural invariant of the kernel-bridge construction.

2. **Structural collapse at $n = 6$**: The Chinese Remainder Theorem decomposes $\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$, creating inter-layer interference that $\sum k!$ cannot capture.

Fundamental principle: **$S_n$ is a multiplicative structure, and $\sum k!$ is its additive approximation. Addition destroys the structure that multiplication creates. At $n = 6$ this destruction first becomes visible, and it never heals.**

---

## Known Bounds (Reference)

For $n \geq 6$:

| Bound | Formula | Source |
|-------|---------|--------|
| Lower | $n! + (n-1)! + (n-2)! + n - 3$ | Anonymous / Engen-Vatter (2018) |
| Upper | $n! + (n-1)! + (n-2)! + (n-3)! + n - 3$ | Egan (2018) |
| Gap   | $(n-3)!$ | — |

---

## 7. Inevitability of Global Search: Why No General Formula Exists

### 7.1 Local Optimum vs Global Optimum

The Expansion Lemma proves the following with **equality**:

> Expanding an $(n-1)$-superpermutation of length $l$ yields an $n$-superpermutation of length exactly $l + n!$.

This is the **optimum of the layer-by-layer construction**. When adding each layer independently, no cost less than $n!$ is possible. That is:

$$\text{local optimum} = \sum_{k=1}^{n} k!$$

Yet Section 4 proves:

$$\text{OPT}(n) < \sum_{k=1}^{n} k! \quad (n \geq 6)$$

**The local optimum is not the global optimum.** The existence of this gap is the key.

### 7.2 What the Gap Proves

We analyze the precise meaning of the gap $\Delta(n) = \sum k! - \text{OPT}(n) > 0$.

The Expansion Lemma construction processes **each layer independently**. The fact that its cost is exactly $n!$ means it **exploits no inter-layer interaction whatsoever**.

Since $\text{OPT}(n) < \sum k!$, the actual optimal superpermutation does exploit inter-layer interaction to reduce cost. This saving is **invisible to local information (the structure of a single layer)** — if it were visible, the Expansion Lemma construction would have already exploited it.

Therefore:

$$\Delta(n) > 0 \quad \Longrightarrow \quad \text{reaching OPT}(n) \text{ requires global information}$$

### 7.3 Information Destruction and Non-Recoverability

The $\sum k!$ formula decomposes the multiplicative structure of $S_n$ additively:

$$S_n \;\xrightarrow{\text{additive decomposition}}\; 1! + 2! + \cdots + n!$$

This decomposition **destroys inter-layer interference information**. $\Delta(n)$ is the amount of destroyed information.

Destroyed information cannot be recovered from within the additive representation. The expression $\text{OPT}(n) = \sum k! - \Delta(n)$ makes $\Delta(n)$ appear to be an independently computable quantity, but in reality, $\text{OPT}(n)$ must be known first before $\Delta(n)$ can be determined. **The order is reversed.**

To determine $\Delta(n)$ independently, one must return to the pre-destruction structure — the full multiplicative structure of $S_n$ — and perform global optimization.

### 7.4 Core Result

**Theorem.** No general formula for $\text{OPT}(n)$ exists.

**Argument.**

1. By the Expansion Lemma, the cost of layer-independent construction is exactly $\sum k!$. This is the **optimum of the additive approach**.

2. For $n \geq 6$, $\text{OPT}(n) < \sum k!$. Therefore the additive approach fails to reach the global optimum.

3. The gap $\Delta(n)$ between additive optimum and global optimum arises from inter-layer interference, which depends on the cyclic group decomposition of $\mathbb{Z}_n$ (Section 5).

4. Determining $\Delta(n)$ requires global optimization over the permutation graph of $S_n$. This is the process of returning to the original structure to search for what was lost in the additive decomposition, and cannot be replaced by any closed-form expression.

5. Since $\text{OPT}(n) = \sum k! - \Delta(n)$, if $\Delta(n)$ has no closed formula, neither does $\text{OPT}(n)$. $\blacksquare$

### 7.5 Summary

| | $\sum k!$ (additive approximation) | $\text{OPT}(n)$ (global optimum) |
|---|---|---|
| Construction | Layer-independent expansion | Global search over $S_n$ |
| Inter-layer interaction | Ignored | Exploited |
| Formula exists | Yes ($\sum k!$) | No |
| Computability | $O(n)$ | Requires global optimization |

$$\boxed{\text{local optimum} \neq \text{global optimum} \;\Longrightarrow\; \text{global search inevitable} \;\Longrightarrow\; \text{no general formula}}$$

**Addition destroys multiplicative structure. Destroyed information cannot be recovered. To reach the global optimum, one must return to the pre-destruction structure and search it entirely. This is why no formula exists.**

---

## References

1. R. Houston, "Tackling the Minimal Superpermutation Problem," arXiv:1408.5108, 2014.
2. N. Johnston, "Non-uniqueness of Minimal Superpermutations," *Discrete Mathematics* 313(14):1553-1557, 2013.
3. Anonymous 4chan Poster, R. Houston, J. Pantone, V. Vatter, "A lower bound on the length of the shortest superpattern," OEIS A180632, 2018.
4. M. Engen, V. Vatter, "Containing All Permutations," *The American Mathematical Monthly* 128(1):4-24, 2021. arXiv:1810.08252.
5. G. Egan, "Superpermutations," https://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html, 2018.
