"""
Entropy-Adaptive Optimizer
===========================

loss trajectory를 "문자열"로 읽는다.
bigram entropy H(state_t | state_{t-1})로 예측 가능성 측정.

H 낮음 → loss가 예측 가능 → 수렴 중 → 가속
H 높음 → loss가 예측 불가 → 탐색 중 → Adam처럼 신중

vs ITAS의 CV(변동계수):
  loss = [2.1, 2.0, 2.1, 2.0]  → CV 높음(감속) but H 낮음(가속해야 맞음)
  loss = [2.1, 1.8, 2.3, 1.9]  → CV 높음(감속) and H 높음(감속 맞음)
  CV는 이 둘을 구분 못 함. 엔트로피는 구분함.

감속: 비대칭 EMA (가속은 느리게, 감속은 빠르게)

원리: "관찰이 짧으면 혼돈이고, 관찰이 길면 질서다"
      — Hamlet Character Entropy Experiment (2026)

Author: 쵸리 (Chori)
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ─────────────────────────────────────────────
# Entropy Engine
# ─────────────────────────────────────────────

N_BINS = 5  # 급하강 / 완하강 / 정체 / 완상승 / 급상승


def quantize_delta(delta, thresholds):
    """loss 변화량 → 이산 상태 (0..N_BINS-1)"""
    for i, t in enumerate(thresholds):
        if delta < t:
            return i
    return len(thresholds)


def bigram_entropy(states, n_bins=N_BINS):
    """
    H(state_t | state_{t-1})

    transition matrix로부터 조건부 Shannon entropy 계산.
    높을수록 예측 불가, 낮을수록 예측 가능.
    """
    if len(states) < 2:
        return np.log2(n_bins)

    T = np.zeros((n_bins, n_bins))
    for i in range(len(states) - 1):
        s_from = min(states[i], n_bins - 1)
        s_to = min(states[i + 1], n_bins - 1)
        T[s_from][s_to] += 1

    total = np.sum(T)
    if total == 0:
        return np.log2(n_bins)

    H = 0.0
    for x in range(n_bins):
        row_sum = np.sum(T[x])
        if row_sum == 0:
            continue
        p_x = row_sum / total
        for y in range(n_bins):
            if T[x][y] > 0:
                p_y_given_x = T[x][y] / row_sum
                H -= p_x * p_y_given_x * np.log2(p_y_given_x)

    return H


# ─────────────────────────────────────────────
# Adam (Vanilla Baseline)
# ─────────────────────────────────────────────

class Adam:
    """Pure Adam. 비교 기준."""

    def __init__(self, func, x0, lr=0.01):
        self.func = func
        self.x = x0.copy()
        self.lr = lr
        self.dim = len(x0)
        self.m = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.t = 0
        self.best_x = x0.copy()
        self.best_loss = func(x0)
        self.history = [self.best_loss]

    def _grad(self):
        base = self.func(self.x)
        g = np.zeros(self.dim)
        for i in range(self.dim):
            xp = self.x.copy()
            xp[i] += 1e-7
            g[i] = (self.func(xp) - base) / 1e-7
        return g

    def step(self):
        self.t += 1
        g = self._grad()
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)
        self.x -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        loss = self.func(self.x)
        self.history.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_x = self.x.copy()
        return loss

    def run(self, steps=500):
        for _ in range(steps):
            self.step()
        return self.best_loss


# ─────────────────────────────────────────────
# EntropyAdam
# ─────────────────────────────────────────────

class EntropyAdam:
    """
    Adam + Shannon bigram entropy 가속기.

    loss trajectory를 양자화 → bigram 전이행렬 → 조건부 엔트로피
    H 낮으면 가속, H 높으면 Adam 기본.
    비대칭 EMA: 가속은 느리게(신중), 감속은 빠르게(안전).
    """

    def __init__(self, func, x0, lr=0.01,
                 window=50, max_boost=5.0,
                 accel_alpha=0.05, brake_alpha=0.3):
        self.func = func
        self.x = x0.copy()
        self.lr = lr
        self.dim = len(x0)

        # Adam internals
        self.m = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.t = 0

        # Entropy config
        self.window = window
        self.max_boost = max_boost
        self.accel_alpha = accel_alpha
        self.brake_alpha = brake_alpha

        # Tracking
        self.best_x = x0.copy()
        self.best_loss = func(x0)
        self.history = [self.best_loss]
        self.deltas = []
        self.states = []
        self.H_history = []
        self.mult_history = []
        self.current_mult = 1.0

    def _grad(self):
        base = self.func(self.x)
        g = np.zeros(self.dim)
        for i in range(self.dim):
            xp = self.x.copy()
            xp[i] += 1e-7
            g[i] = (self.func(xp) - base) / 1e-7
        return g

    def _update_entropy(self, loss):
        """새 loss로 엔트로피 상태 갱신"""
        if len(self.history) < 1:
            return

        prev = self.history[-1]
        delta = (loss - prev) / (abs(prev) + 1e-10)
        self.deltas.append(delta)

        if len(self.deltas) > self.window:
            self.deltas = self.deltas[-self.window:]

        # 적응형 threshold (percentile 기반)
        if len(self.deltas) >= 10:
            ds = np.array(self.deltas)
            thr = np.percentile(ds, [20, 40, 60, 80])
            state = quantize_delta(delta, thr)
        else:
            state = quantize_delta(delta, np.array([-0.01, -0.001, 0.001, 0.01]))

        self.states.append(state)
        if len(self.states) > self.window:
            self.states = self.states[-self.window:]

    def _get_mult(self):
        """엔트로피 → LR 배수"""
        H_max = np.log2(N_BINS)

        if len(self.states) < 10:
            self.H_history.append(H_max)
            self.mult_history.append(1.0)
            return 1.0

        H = bigram_entropy(self.states)
        H_norm = min(H / H_max, 1.0)

        # (1 - H_norm)^2: 수렴할수록 급가속
        raw = 1.0 + (self.max_boost - 1.0) * (1.0 - H_norm) ** 2

        # 비대칭 EMA: 가속은 천천히, 감속은 빠르게
        if raw > self.current_mult:
            alpha = self.accel_alpha
        else:
            alpha = self.brake_alpha

        self.current_mult = (1 - alpha) * self.current_mult + alpha * raw

        self.H_history.append(H)
        self.mult_history.append(self.current_mult)
        return self.current_mult

    def step(self):
        self.t += 1
        g = self._grad()

        # Adam
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)

        # Entropy-adaptive LR
        mult = self._get_mult()

        self.x -= self.lr * mult * mh / (np.sqrt(vh) + 1e-8)

        loss = self.func(self.x)
        self._update_entropy(loss)
        self.history.append(loss)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_x = self.x.copy()

        return loss

    def run(self, steps=500):
        for _ in range(steps):
            self.step()
        return self.best_loss


# ─────────────────────────────────────────────
# Benchmark Functions
# ─────────────────────────────────────────────

def rosenbrock(x):
    """바나나 계곡. global min = 0 at (1,1,...,1)"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def rastrigin(x):
    """local minima 지옥. global min = 0 at (0,...,0)"""
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    """평평한 고원 + local minima. global min = 0 at (0,...,0)"""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


# ─────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────

FUNCS = {
    'Rosenbrock': (rosenbrock, (-5, 10)),
    'Rastrigin':  (rastrigin,  (-5.12, 5.12)),
    'Ackley':     (ackley,     (-5, 5)),
}


def run_benchmark(dims=(10, 30), n_steps=500, n_seeds=5):
    print("=" * 70)
    print("  EntropyAdam vs Adam")
    print("  loss trajectory의 Shannon bigram entropy → 적응형 학습률")
    print("=" * 70)

    results = {}

    for fname, (func, (lo, hi)) in FUNCS.items():
        for dim in dims:
            key = f"{fname} {dim}D"
            adam_losses = []
            ent_losses = []

            for seed in range(n_seeds):
                np.random.seed(seed * 100 + 42)
                x0 = np.random.uniform(lo, hi, dim)

                adam_losses.append(Adam(func, x0.copy(), lr=0.01).run(n_steps))
                ent_losses.append(EntropyAdam(func, x0.copy(), lr=0.01).run(n_steps))

            am, astd = np.mean(adam_losses), np.std(adam_losses)
            em, estd = np.mean(ent_losses), np.std(ent_losses)
            ratio = am / (em + 1e-20)

            results[key] = {'adam': (am, astd), 'entropy': (em, estd), 'ratio': ratio}

            winner = "EntropyAdam" if em < am else "Adam"
            print(f"\n  {key}:")
            print(f"    Adam:        {am:>14.4f} +/- {astd:.4f}")
            print(f"    EntropyAdam: {em:>14.4f} +/- {estd:.4f}")
            print(f"    ratio: {ratio:.1f}x  -->  {winner}")

    return results


def plot_detail(func, fname, dim=10, n_steps=500, seed=42):
    """단일 run 상세: loss curve + entropy signal + LR multiplier"""
    if not HAS_PLT:
        print("  (matplotlib 없음, 플롯 스킵)")
        return

    np.random.seed(seed)
    lo, hi = FUNCS[fname][1]
    x0 = np.random.uniform(lo, hi, dim)

    opt_a = Adam(func, x0.copy(), lr=0.01)
    opt_e = EntropyAdam(func, x0.copy(), lr=0.01)
    opt_a.run(n_steps)
    opt_e.run(n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1. Loss
    axes[0].semilogy(opt_a.history, 'b-', alpha=0.7, label=f'Adam (best={opt_a.best_loss:.4f})')
    axes[0].semilogy(opt_e.history, 'r-', alpha=0.7, label=f'EntropyAdam (best={opt_e.best_loss:.4f})')
    axes[0].set_ylabel('Loss (log)')
    axes[0].legend()
    axes[0].set_title(f'{fname} {dim}D')
    axes[0].grid(True, alpha=0.3)

    # 2. Entropy
    H_max = np.log2(N_BINS)
    axes[1].plot(opt_e.H_history, 'g-', alpha=0.7, label='H(state_t | state_{t-1})')
    axes[1].axhline(y=H_max, color='gray', ls='--', alpha=0.5, label=f'H_max = {H_max:.2f}')
    axes[1].axhline(y=0, color='gray', ls='--', alpha=0.3)
    axes[1].set_ylabel('Bigram Entropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Multiplier
    axes[2].plot(opt_e.mult_history, color='orange', alpha=0.7, label='LR multiplier')
    axes[2].axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='Adam baseline (1.0x)')
    axes[2].set_ylabel('LR Multiplier')
    axes[2].set_xlabel('Step')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'entropy_vs_adam_{fname.lower()}_{dim}d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_benchmark()

    print("\n" + "=" * 70)
    print("  Detail Plots")
    print("=" * 70)
    for fname, (func, _) in FUNCS.items():
        plot_detail(func, fname)

    # Summary
    print("\n" + "=" * 70)
    wins = sum(1 for r in results.values() if r['ratio'] > 1.0)
    total = len(results)
    ratios = [r['ratio'] for r in results.values()]
    print(f"  EntropyAdam wins: {wins}/{total}")
    print(f"  avg ratio: {np.mean(ratios):.1f}x")
    print(f"  max ratio: {np.max(ratios):.1f}x")
    print("=" * 70)
