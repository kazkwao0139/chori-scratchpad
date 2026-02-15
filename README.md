# chori-scratchpad

> Hirameki를 정리해두는 곳.

새벽에 갑자기 떠오른 아이디어를 코드로 옮겨보는 공간입니다.
장난처럼 시작했지만, 검증이 필요한 것들이 많고, 언젠가 다른 프로젝트나 솔루션의 씨앗이 될 수도 있는 것들을 모아둡니다.

## Projects

| 폴더 | 한 줄 요약 |
|------|-----------|
| [`viral/`](./viral/) | 넷플릭스 추천 알고리즘을 뒤집어서 가짜뉴스 확산을 탐지하고 멈출 수 있는지 실험 |
| [`todo_gravity/`](./todo_gravity/) | 키스트로크 타이밍의 \|z-score\|로 투두 중요도를 자동 추정 |
| [`hello_world/`](./hello_world/) | 압축 = 예측 = 이해 — 텍스트의 압축률로 수신자의 지식을 역산한다 |
| [`entropy_adam/`](./entropy_adam/) | loss trajectory의 Shannon bigram entropy로 학습률을 적응 — Adam 전승 |
| [`three_laws/`](./three_laws/) | entropy_adam에서 이어지는 사고실험 — AI 윤리란 무엇일까? harm = delta_S > 0 |
| [`lorenz/`](./lorenz/) | 나비효과 멍때리기. |
| [`haruhi/`](./haruhi/) | 옵티마이저의 전역 탐색이 쓸모없다는걸 알아내고선 하루히 문제가 갑자기 떠올라서 메모 |
| [`ballistic_cache/`](./ballistic_cache/) | RK4 탄도 시뮬레이션을 캐싱해서 O(1) 조회하려는 시도. 근데 별 쓸모가 없더라 |

---

## 왜 자꾸 열역학 제2법칙이랑 대수의 법칙이 나오냐면

> 1. 무질서해지는 건 막을 수 없다.
> 2. 많이 하면 평균에 수렴하는 것도 막을 수 없다.
> 3. 둘 다 우주의 전제조건이라 자꾸 갖다 쓰게 된다.

친구가 도대체 왜 그렇게 그 두개에 집착하냐고 물어봐서 대문에 박는다.

좋아서 집착하는 건 아니고, 애증에 가깝다.

이유는 별거 없다. 우주가 시작된 이후로 v1.0이고, 우주가 끝날 때까지 패치 예정이 없다. 근데 활용할 곳은 무궁무진하다. 그래서 자꾸 꺼내 쓰게 된다.

- **열역학 제2법칙** — 우주는 질서에서 무질서로 흐른다. 여기저기서 썼다.
- **대수의 법칙** — 숫자가 커지면 예측 가능해진다. 여기저기서 썼다.

둘 다 증명이 어렵지 않진 않고. 꽤 어렵다 사실. 근데 무시하면 큰일난다.

> **경고: 밑에 가독성 테러. 수학에 자신있는 사람만 내려볼 것. 3줄요약만 알아도 상관없음.**

### 열역학 제2법칙

먼저 제1법칙. 에너지는 생기지도, 사라지지도 않는다. 형태만 바뀐다:

$$dU = \delta Q - \delta W$$

내부 에너지 변화 $dU$ 는 받은 열 $\delta Q$ 에서 한 일 $\delta W$ 를 뺀 것이다. 에너지 보존. 이건 대전제다.

제2법칙은 여기에 방향을 붙인다. 에너지는 보존되지만, 아무 방향으로나 흐르지는 않는다.

엔트로피를 세는 가장 근본적인 공식은 볼츠만이 만들었다. 계가 취할 수 있는 미시 상태 수를 $\Omega$ 라 하면:

$$S = k_B \ln \Omega$$

$\Omega$ 가 크다는 건 가능한 배치가 많다는 뜻이고, 곧 무질서하다는 뜻이다.

열역학적으로는 클라우지우스 부등식이 핵심이다. 온도 $T$ 에서 열 $\delta Q$ 를 받을 때:

$$dS \geq \frac{\delta Q}{T}$$

비가역 과정이면 등호가 성립하지 않는다. 고립계에서는 $\delta Q = 0$ 이니까 $dS \geq 0$ 이 되고, 엔트로피는 절대 줄어들지 않는다.

실제로 쓸 때는 자유 에너지 쪽이 편하다. 등온등적 조건에서는 헬름홀츠 자유 에너지:

$$F = U - TS$$

등온등압 조건에서는 깁스 자유 에너지:

$$G = H - TS$$

자발적 과정은 자유 에너지가 감소하는 방향으로 진행된다. $\Delta G < 0$ 이면 알아서 일어난다.

그리고 란다우어의 원리. 정보를 지우는 건 공짜가 아니다. 1비트를 지울 때 최소한 이만큼의 에너지가 열로 방출된다:

$$E \geq k_B T \ln 2$$

이게 정보 엔트로피와 열역학 엔트로피를 잇는 다리다.

결국 전부 같은 소리다. 세상은 엔트로피가 증가하는 방향으로 굴러간다.

### WLLN (약한 대수의 법칙)

$X_1, X_2, \ldots$ 가 iid이고 $E[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2 < \infty$ 라 하자. 표본 평균

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$$

에 대해 보이고 싶은 것은 확률 수렴(convergence in probability)이다:

$$\bar{X}_n \xrightarrow{P} \mu$$

이게 정확히 뭐냐면, 임의의 $\epsilon > 0$ 과 $\delta > 0$ 에 대해 충분히 큰 $N$ 이 존재해서, 모든 $n \geq N$ 에서 $P(|\bar{X}_n - \mu| > \epsilon) < \delta$ 가 성립한다는 것이다. 동치인 표현으로:

$$\forall \epsilon > 0, \quad \lim_{n \to \infty} P(|\bar{X}_n - \mu| > \epsilon) = 0$$

$n$ 이 커지면 표본 평균이 $\mu$ 에서 $\epsilon$ 이상 벗어날 확률이 0으로 간다는 뜻이다.

증명에 체비셰프 부등식을 쓸 건데, 이것부터 유도하자. 출발점은 마르코프 부등식이다. 음이 아닌 확률변수 $Z \geq 0$ 과 $a > 0$ 에 대해:

$$E[Z] = \int_0^\infty z \, dF(z) \geq \int_a^\infty z \, dF(z) \geq a \int_a^\infty dF(z) = a \cdot P(Z \geq a)$$

정리하면:

$$P(Z \geq a) \leq \frac{E[Z]}{a}$$

이게 마르코프 부등식이다. 여기서 $Z = (Y - E[Y])^2$, $a = \epsilon^2$ 을 넣으면:

$$P((Y - E[Y])^2 \geq \epsilon^2) \leq \frac{E[(Y - E[Y])^2]}{\epsilon^2}$$

좌변은 $P(|Y - E[Y]| \geq \epsilon)$ 이고 우변의 분자는 $\text{Var}(Y)$ 이다:

$$P(|Y - E[Y]| \geq \epsilon) \leq \frac{\text{Var}(Y)}{\epsilon^2}$$

이게 체비셰프 부등식이다. 이제 본론으로 간다.

$X_i$ 가 독립이므로 표본 평균의 분산은:

$$\text{Var}(\bar{X}_n) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{\sigma^2}{n}$$

체비셰프 부등식에 $Y = \bar{X}_n$ 을 넣으면, 임의의 $\epsilon > 0$ 에 대해:

$$P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}$$

$n \to \infty$ 이면 우변이 0으로 간다. $\blacksquare$

### SLLN (강한 대수의 법칙)

WLLN은 "각 $n$ 에서 벗어날 확률이 줄어든다"는 것이고, SLLN은 "경로 자체가 수렴한다"는 것이다. almost sure convergence는 convergence in probability를 함의하지만 역은 성립하지 않는다. 더 강한 결과다.

$X_1, X_2, \ldots$ 가 iid이고 $E[X_i] = \mu$, $E[X_i^4] < \infty$ 일 때 보이고 싶은 것은:

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

4차 모멘트를 쓴다. $Y_i = X_i - \mu$ 로 놓으면 $E[Y_i] = 0$, $E[Y_i^2] = \sigma^2$, $E[Y_i^4] = \kappa < \infty$ 이다.

$$E\left[\left(\sum_{i=1}^n Y_i\right)^4\right] = \sum_{i,j,k,l} E[Y_i Y_j Y_k Y_l]$$

을 전개하면, $Y_i$ 가 독립이고 $E[Y_i] = 0$ 이므로 어떤 인덱스가 홀수 번 등장하는 항은 전부 0이 된다. 살아남는 경우는 두 가지뿐이다. 네 인덱스가 모두 같은 경우 ($i = j = k = l$)는 $n$ 개의 항이 있고 각각 $E[Y_i^4] = \kappa$ 를 기여한다. 두 쌍으로 나뉘는 경우 ($i = j \neq k = l$ 등)는 $\binom{n}{2}$ 가지 인덱스 조합에 $\binom{4}{2} = 6$ 가지 자리 배치가 있어서 총 $3n(n-1)$ 개의 항이 되고, 각각 $E[Y_i^2] E[Y_k^2] = \sigma^4$ 를 기여한다.

따라서:

$$E\left[\left(\sum_{i=1}^n Y_i\right)^4\right] = n\kappa + 3n(n-1)\sigma^4$$

$$E[\bar{Y}_n^4] = \frac{n\kappa + 3n(n-1)\sigma^4}{n^4} \leq \frac{\kappa + 3\sigma^4}{n^2}$$

마르코프 부등식에 의해:

$$P(|\bar{X}_n - \mu| > \epsilon) = P(\bar{Y}_n^4 > \epsilon^4) \leq \frac{E[\bar{Y}_n^4]}{\epsilon^4} \leq \frac{\kappa + 3\sigma^4}{n^2 \epsilon^4}$$

$C = \frac{\kappa + 3\sigma^4}{\epsilon^4}$ 로 놓으면:

$$\sum_{n=1}^{\infty} P(|\bar{X}_n - \mu| > \epsilon) \leq C \sum_{n=1}^{\infty} \frac{1}{n^2} < \infty$$

여기서 보렐-칸텔리 보조정리를 쓴다. 사건들의 확률 합이 유한하면 그 사건이 무한히 자주 일어날 확률은 0이다.

$$P(|\bar{X}_n - \mu| > \epsilon \text{ i.o.}) = 0$$

이것이 모든 $\epsilon > 0$ 에 대해 성립하므로, $\epsilon = 1/m$ $(m = 1, 2, \ldots)$ 을 취하면:

$$P(\bar{X}_n \not\to \mu) = P\left(\bigcup_{m=1}^{\infty} \{|\bar{X}_n - \mu| > 1/m \text{ i.o.}\}\right) \leq \sum_{m=1}^{\infty} 0 = 0$$

$$\therefore \quad P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1 \quad \blacksquare$$

---

ps. 이 스크래치패드만 해도 WLLN으로만 되는 게 있고, SLLN이 필요한 게 있다. 뭔지는 안 쓰겠다(...)

ps2. 원래는 접기 문법으로 테러방지를 하려했으나... LaTeX이 깨지는 문제가 있어서... 스크래치패드인데 뭐 어때?

---

# chori-scratchpad

> A place to organize Hirameki.

A space where I turn ideas that pop into my head at 3 AM into code.
What started as a joke often needs verification, and some of these may one day become seeds for other projects or solutions.

## Projects

| Folder | One-liner |
|--------|-----------|
| [`viral/`](./viral/) | Flipping Netflix's recommendation algorithm to detect and stop fake news propagation |
| [`todo_gravity/`](./todo_gravity/) | Auto-estimating todo priority from the \|z-score\| of keystroke timing |
| [`hello_world/`](./hello_world/) | Compression = prediction = understanding — reverse-engineering a recipient's knowledge from text compression ratio |
| [`entropy_adam/`](./entropy_adam/) | Adapting learning rate via Shannon bigram entropy of loss trajectory — beats Adam |
| [`three_laws/`](./three_laws/) | A thought experiment continuing from entropy_adam — what is AI ethics? harm = delta_S > 0 |
| [`lorenz/`](./lorenz/) | Staring at the butterfly effect. |
| [`haruhi/`](./haruhi/) | Realized global search in optimizers is useless, then the Haruhi problem suddenly came to mind |
| [`ballistic_cache/`](./ballistic_cache/) | Attempting to cache RK4 ballistic simulations for O(1) lookup. Turned out to be pretty useless |

---

## Why the Second Law of Thermodynamics and the Law of Large Numbers keep showing up

> 1. Disorder is unstoppable.
> 2. With enough samples, convergence to the mean is also unstoppable.
> 3. Both are preconditions of the universe, so I keep reaching for them.

A buddy asked me why I'm so obsessed with these two. So I'm putting this on the front page.

It's not that I love them. It's closer to a love-hate relationship.

The reason is simple. They've been v1.0 since the universe began, and there are no patches planned until the universe ends. But the use cases are endless. So I keep reaching for them.

- **Second Law of Thermodynamics** — The universe flows from order to disorder. Used all over the place.
- **Law of Large Numbers** — As numbers grow, things become predictable. Used all over the place.

The proofs aren't exactly easy. They're actually pretty hard. But ignore them at your own risk.

> **Warning: readability terrorism below. Only scroll down if you're confident in math. The 3-line summary above is all you need.**

### Second Law of Thermodynamics

First, the First Law. Energy is neither created nor destroyed. It only changes form:

$$dU = \delta Q - \delta W$$

The change in internal energy $dU$ equals the heat received $\delta Q$ minus the work done $\delta W$. Conservation of energy. This is the ground rule.

The Second Law adds direction. Energy is conserved, but it doesn't flow in just any direction.

The most fundamental formula for counting entropy was given by Boltzmann. Let $\Omega$ be the number of microstates available to the system:

$$S = k_B \ln \Omega$$

A large $\Omega$ means many possible configurations, which means disorder.

Thermodynamically, the Clausius inequality is the core. When receiving heat $\delta Q$ at temperature $T$:

$$dS \geq \frac{\delta Q}{T}$$

Equality fails for irreversible processes. In an isolated system $\delta Q = 0$, so $dS \geq 0$. Entropy never decreases.

In practice, free energy is more convenient. Under isothermal-isochoric conditions, the Helmholtz free energy:

$$F = U - TS$$

Under isothermal-isobaric conditions, the Gibbs free energy:

$$G = H - TS$$

Spontaneous processes proceed in the direction of decreasing free energy. If $\Delta G < 0$, it happens on its own.

Then there's Landauer's principle. Erasing information isn't free. Erasing one bit dissipates at least this much energy as heat:

$$E \geq k_B T \ln 2$$

This is the bridge connecting information entropy and thermodynamic entropy.

It all says the same thing. The world rolls in the direction of increasing entropy.

### WLLN (Weak Law of Large Numbers)

Let $X_1, X_2, \ldots$ be iid with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$. For the sample mean

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$$

what we want to show is convergence in probability:

$$\bar{X}_n \xrightarrow{P} \mu$$

What this means precisely: for any $\epsilon > 0$ and $\delta > 0$, there exists a sufficiently large $N$ such that for all $n \geq N$, $P(|\bar{X}_n - \mu| > \epsilon) < \delta$. Equivalently:

$$\forall \epsilon > 0, \quad \lim_{n \to \infty} P(|\bar{X}_n - \mu| > \epsilon) = 0$$

As $n$ grows, the probability that the sample mean deviates from $\mu$ by more than $\epsilon$ goes to 0.

The proof uses Chebyshev's inequality, so let's derive that first. The starting point is Markov's inequality. For a non-negative random variable $Z \geq 0$ and $a > 0$:

$$E[Z] = \int_0^\infty z \, dF(z) \geq \int_a^\infty z \, dF(z) \geq a \int_a^\infty dF(z) = a \cdot P(Z \geq a)$$

Rearranging:

$$P(Z \geq a) \leq \frac{E[Z]}{a}$$

This is Markov's inequality. Substituting $Z = (Y - E[Y])^2$ and $a = \epsilon^2$:

$$P((Y - E[Y])^2 \geq \epsilon^2) \leq \frac{E[(Y - E[Y])^2]}{\epsilon^2}$$

The left side is $P(|Y - E[Y]| \geq \epsilon)$ and the numerator on the right is $\text{Var}(Y)$:

$$P(|Y - E[Y]| \geq \epsilon) \leq \frac{\text{Var}(Y)}{\epsilon^2}$$

This is Chebyshev's inequality. Now for the main proof.

Since the $X_i$ are independent, the variance of the sample mean is:

$$\text{Var}(\bar{X}_n) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{\sigma^2}{n}$$

Plugging $Y = \bar{X}_n$ into Chebyshev's inequality, for any $\epsilon > 0$:

$$P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}$$

As $n \to \infty$, the right side goes to 0. $\blacksquare$

### SLLN (Strong Law of Large Numbers)

WLLN says "the probability of deviating decreases at each $n$." SLLN says "the path itself converges." Almost sure convergence implies convergence in probability, but the converse does not hold. This is a stronger result.

Let $X_1, X_2, \ldots$ be iid with $E[X_i] = \mu$ and $E[X_i^4] < \infty$. What we want to show is:

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

We use the fourth moment. Let $Y_i = X_i - \mu$, so $E[Y_i] = 0$, $E[Y_i^2] = \sigma^2$, and $E[Y_i^4] = \kappa < \infty$.

Expanding:

$$E\left[\left(\sum_{i=1}^n Y_i\right)^4\right] = \sum_{i,j,k,l} E[Y_i Y_j Y_k Y_l]$$

Since the $Y_i$ are independent and $E[Y_i] = 0$, any term where some index appears an odd number of times vanishes. Only two types of terms survive. When all four indices are equal ($i = j = k = l$), there are $n$ such terms, each contributing $E[Y_i^4] = \kappa$. When the indices split into two pairs ($i = j \neq k = l$, etc.), there are $\binom{n}{2}$ index combinations with $\binom{4}{2} = 6$ position arrangements, giving $3n(n-1)$ terms in total, each contributing $E[Y_i^2] E[Y_k^2] = \sigma^4$.

Therefore:

$$E\left[\left(\sum_{i=1}^n Y_i\right)^4\right] = n\kappa + 3n(n-1)\sigma^4$$

$$E[\bar{Y}_n^4] = \frac{n\kappa + 3n(n-1)\sigma^4}{n^4} \leq \frac{\kappa + 3\sigma^4}{n^2}$$

By Markov's inequality:

$$P(|\bar{X}_n - \mu| > \epsilon) = P(\bar{Y}_n^4 > \epsilon^4) \leq \frac{E[\bar{Y}_n^4]}{\epsilon^4} \leq \frac{\kappa + 3\sigma^4}{n^2 \epsilon^4}$$

Setting $C = \frac{\kappa + 3\sigma^4}{\epsilon^4}$:

$$\sum_{n=1}^{\infty} P(|\bar{X}_n - \mu| > \epsilon) \leq C \sum_{n=1}^{\infty} \frac{1}{n^2} < \infty$$

Now we invoke the Borel-Cantelli lemma: if the sum of probabilities of events is finite, the probability that they occur infinitely often is 0.

$$P(|\bar{X}_n - \mu| > \epsilon \text{ i.o.}) = 0$$

Since this holds for all $\epsilon > 0$, taking $\epsilon = 1/m$ for $m = 1, 2, \ldots$:

$$P(\bar{X}_n \not\to \mu) = P\left(\bigcup_{m=1}^{\infty} \{|\bar{X}_n - \mu| > 1/m \text{ i.o.}\}\right) \leq \sum_{m=1}^{\infty} 0 = 0$$

$$\therefore \quad P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1 \quad \blacksquare$$

---

ps. Even within this scratchpad alone, some things only need WLLN, while others require SLLN. I won't say which(...)

ps2. I originally tried to use collapsible sections to prevent math terrorism, but... LaTeX breaks inside them, so... it's a scratchpad, who cares?
