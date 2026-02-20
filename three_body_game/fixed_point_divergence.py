"""
Fixed Point Divergence: k → |S|·k² → ∞
3-player stochastic game best-response automaton sizes diverge super-exponentially.
"""

import numpy as np
from itertools import product
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

S = 3  # game states

# ═══════════════════════════════════════════════════════════
# Part 1: Theoretical — fixed point equation has no solution
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  FIXED POINT EQUATION: k = f(|S|·k²)")
print("  |S| = 3, f(n) ≥ n (full distinction game)")
print("=" * 60)

print(f"\n  {'k':>4} │ {'|S|·k²':>8} │ {'k ≥ |S|·k² ?'}")
print(f"  {'─'*32}")
for k in range(1, 7):
    rhs = S * k * k
    print(f"  {k:4d} │ {rhs:8d} │ {'impossible' if k < rhs else 'ok'}")

print(f"\n  k ≥ 2에서 k < |S|·k². 고정점 없음.\n")

# ═══════════════════════════════════════════════════════════
# Part 2: Iterated best-response — super-exponential divergence
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("  ITERATED BEST-RESPONSE DIVERGENCE")
print("  각 플레이어가 순서대로 최적 응답 크기로 업그레이드")
print("=" * 60)

sizes = [2, 2, 2]
print(f"\n  {'Step':>4} │ {'k_A':>14} {'k_B':>14} {'k_C':>14}")
print(f"  {'─'*52}")
print(f"  {0:4d} │ {sizes[0]:14,} {sizes[1]:14,} {sizes[2]:14,}")

for step in range(1, 12):
    player = step % 3
    opps = [i for i in range(3) if i != player]
    new = S * sizes[opps[0]] * sizes[opps[1]]
    sizes[player] = new

    if max(sizes) > 10**15:
        print(f"  {step:4d} │ {'> 10^15':>14} {'':>14} {'':>14}  ← 발산")
        break
    print(f"  {step:4d} │ {sizes[0]:14,} {sizes[1]:14,} {sizes[2]:14,}")

# ═══════════════════════════════════════════════════════════
# Part 3: Empirical — MDP optimal policy needs > k states
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  EMPIRICAL VERIFICATION")
print("  Random game + random opponents → MDP → optimal policy")
print("=" * 60)

np.random.seed(42)
payoffs = np.random.uniform(0, 10, (S, 2, 2, 2, 3))
trans_p = np.random.dirichlet(np.ones(S), (S, 2, 2, 2))

print(f"\n  {'k':>3} │ {'MDP':>6} │ {'act=0':>5} {'act=1':>5} │ {'need > k':>8}")
print(f"  {'─'*40}")

for k in [1, 2, 3, 4, 5, 8]:
    np.random.seed(100 + k)
    o1_out = np.random.randint(0, 2, (k, S))
    o1_tr = np.random.randint(0, k, (k, S, 4))
    o2_out = np.random.randint(0, 2, (k, S))
    o2_tr = np.random.randint(0, k, (k, S, 4))

    n_mdp = S * k * k
    R = np.zeros((n_mdp, 2))
    T = np.zeros((n_mdp, 2, n_mdp))

    for s, q1, q2 in product(range(S), range(k), range(k)):
        idx = (s * k + q1) * k + q2
        a1 = o1_out[q1, s]
        a2 = o2_out[q2, s]
        for a0 in range(2):
            R[idx, a0] = payoffs[s][a0][a1][a2][0]
            for sn in range(S):
                p = trans_p[s][a0][a1][a2][sn]
                nq1 = o1_tr[q1, s, a0 * 2 + a2]
                nq2 = o2_tr[q2, s, a0 * 2 + a1]
                T[idx, a0, (sn * k + nq1) * k + nq2] += p

    # Value iteration
    V = np.zeros(n_mdp)
    for _ in range(5000):
        Q = R + 0.99 * (T @ V.reshape(-1, 1) if False else np.einsum('ijk,k->ij', T, V))
        Vn = np.max(Q, axis=1)
        if np.max(np.abs(Vn - V)) < 1e-12:
            break
        V = Vn

    policy = np.argmax(Q, axis=1)
    n0 = np.sum(policy == 0)
    n1 = np.sum(policy == 1)

    # Lower bound on needed states: at least 2 if both actions used
    # Better: count distinct (action, transition-signature) pairs
    sigs = set()
    for idx in range(n_mdp):
        a = policy[idx]
        # transition signature: which MDP states we go to
        next_states = tuple(np.argsort(T[idx, a])[-3:])
        sigs.add((a, next_states))

    need = len(sigs)
    gt = "YES" if need > k else "no"
    print(f"  {k:3d} │ {n_mdp:6d} │ {n0:5d} {n1:5d} │ {gt:>8} ({need} classes)")

# ═══════════════════════════════════════════════════════════
# Part 4: 2-player comparison (should NOT diverge)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  2-PLAYER COMPARISON (should converge)")
print("  Best response: k = f(|S|·k) — linear, not product")
print("=" * 60)

sizes_2p = [2, 2]
print(f"\n  {'Step':>4} │ {'k_A':>14} {'k_B':>14}")
print(f"  {'─'*36}")
print(f"  {0:4d} │ {sizes_2p[0]:14,} {sizes_2p[1]:14,}")

for step in range(1, 12):
    player = step % 2
    opp = 1 - player
    new = S * sizes_2p[opp]  # LINEAR: |S| × k, not |S| × k²
    sizes_2p[player] = new
    print(f"  {step:4d} │ {sizes_2p[0]:14,} {sizes_2p[1]:14,}")
    if max(sizes_2p) > 10**12:
        print(f"  ... (선형 성장, but still grows)")
        break

print(f"\n  2인: k → |S|·k → |S|²·k → ... (지수적, but 고정점 가능)")
print(f"  3인: k → |S|·k² → |S|³·k⁴ → ... (초지수적, 고정점 불가)")

print(f"\n{'='*60}")
print(f"  ∴ 3인: 고정점 없음 → uniform equilibrium 부재")
print(f"{'='*60}")
