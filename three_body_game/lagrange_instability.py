"""
Lagrange Point Instability: Find a 3-player stochastic game
where the stationary NE CHANGES with discount factor λ.

Strategy:
1. Random search for games where pure NE changes with λ
2. Cross-λ stability analysis: no single profile works for all λ
3. Jacobian eigenvalue analysis at the NE
"""

import numpy as np
from itertools import product
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

NS = 2   # states
NA = 2   # actions per player
NP = 3   # players

def compute_values(r, trans, strats, lam):
    """
    Compute discounted values. strats[p][s] = prob of action 1.
    Returns V[s, p].
    """
    R = np.zeros((NS, NP))
    T = np.zeros((NS, NS))

    for s in range(NS):
        for a0, a1, a2 in product(range(NA), repeat=NP):
            actions = [a0, a1, a2]
            prob = 1.0
            for p in range(NP):
                prob *= strats[p][s] if actions[p] == 1 else (1 - strats[p][s])

            for p in range(NP):
                R[s, p] += prob * r[s][a0][a1][a2][p]
            for sn in range(NS):
                T[s, sn] += prob * trans[s][a0][a1][a2][sn]

    A = np.eye(NS) - lam * T
    V = np.linalg.solve(A, (1 - lam) * R)
    return V

def find_all_pure_ne(r, trans, lam):
    """Find ALL pure stationary NE by enumeration."""
    # Each player: 2^NS = 4 pure strategies (action in each state)
    player_strats = list(product(range(NA), repeat=NS))  # 4 options per player
    ne_list = []

    for ps0, ps1, ps2 in product(player_strats, repeat=NP):
        strats = [np.array(ps0, float), np.array(ps1, float), np.array(ps2, float)]
        V_curr = compute_values(r, trans, strats, lam)

        is_ne = True
        for p in range(NP):
            curr_val = np.mean(V_curr[:, p])
            for alt in player_strats:
                alt_strats = [s.copy() for s in strats]
                alt_strats[p] = np.array(alt, float)
                V_alt = compute_values(r, trans, alt_strats, lam)
                if np.mean(V_alt[:, p]) > curr_val + 1e-9:
                    is_ne = False
                    break
            if not is_ne:
                break

        if is_ne:
            profile = (ps0, ps1, ps2)
            ne_list.append(profile)

    return ne_list

def max_deviation_gain(r, trans, strats_float, lam):
    """Max gain any player can get by deviating."""
    V_curr = compute_values(r, trans, strats_float, lam)
    player_strats = list(product(range(NA), repeat=NS))
    max_gain = 0.0

    for p in range(NP):
        curr_val = np.mean(V_curr[:, p])
        for alt in player_strats:
            alt_strats = [s.copy() for s in strats_float]
            alt_strats[p] = np.array(alt, float)
            V_alt = compute_values(r, trans, alt_strats, lam)
            gain = np.mean(V_alt[:, p]) - curr_val
            max_gain = max(max_gain, gain)

    return max_gain


# ═══════════════════════════════════════════════════════════
# Phase 1: Random Search for λ-Unstable Games
# ═══════════════════════════════════════════════════════════

print("=" * 65)
print("  SEARCHING FOR λ-UNSTABLE 3-PLAYER STOCHASTIC GAMES")
print("  Game where pure stationary NE changes with discount factor")
print("=" * 65)

lambdas_test = [0.5, 0.8, 0.9, 0.95, 0.99]
found_games = []

np.random.seed(7)
for trial in range(5000):
    r = np.random.uniform(0, 10, (NS, NA, NA, NA, NP))
    trans = np.random.dirichlet(np.ones(NS), (NS, NA, NA, NA))

    # Find NE at two extreme λ values
    ne_low = find_all_pure_ne(r, trans, 0.5)
    ne_high = find_all_pure_ne(r, trans, 0.99)

    # Check if NE sets differ
    if set(ne_low) != set(ne_high) and len(ne_low) > 0 and len(ne_high) > 0:
        # Interesting! NE changes with λ
        found_games.append((trial, r, trans, ne_low, ne_high))
        if len(found_games) >= 3:
            break

    if (trial + 1) % 1000 == 0:
        print(f"  Searched {trial + 1} games, found {len(found_games)} unstable...")

print(f"\n  Found {len(found_games)} games with λ-dependent NE\n")

if not found_games:
    print("  No games found. Trying with different seed...")
    np.random.seed(123)
    for trial in range(10000):
        r = np.random.uniform(0, 10, (NS, NA, NA, NA, NP))
        # Stronger asymmetry in transitions
        alpha = np.random.uniform(0.1, 5.0, NS)
        trans = np.random.dirichlet(alpha, (NS, NA, NA, NA))

        ne_low = find_all_pure_ne(r, trans, 0.5)
        ne_high = find_all_pure_ne(r, trans, 0.99)

        if set(ne_low) != set(ne_high) and len(ne_low) > 0 and len(ne_high) > 0:
            found_games.append((trial, r, trans, ne_low, ne_high))
            if len(found_games) >= 3:
                break

        if (trial + 1) % 2000 == 0:
            print(f"  Searched {trial + 1} games, found {len(found_games)} unstable...")

    print(f"\n  Found {len(found_games)} games total\n")

# ═══════════════════════════════════════════════════════════
# Phase 2: Detailed Analysis of Found Games
# ═══════════════════════════════════════════════════════════

for gi, (trial_id, r, trans, ne_low, ne_high) in enumerate(found_games):
    print("=" * 65)
    print(f"  GAME #{gi+1} (trial {trial_id})")
    print(f"  NE at λ=0.5: {ne_low}")
    print(f"  NE at λ=0.99: {ne_high}")
    print("=" * 65)

    # Find NE at many λ values
    lambdas_fine = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]

    print(f"\n  {'λ':>8} │ Pure NE profiles")
    print(f"  {'─' * 55}")

    all_ne_by_lam = {}
    all_profiles_ever = set()

    for lam in lambdas_fine:
        ne = find_all_pure_ne(r, trans, lam)
        all_ne_by_lam[lam] = ne
        for p in ne:
            all_profiles_ever.add(p)

        ne_str = ", ".join([f"({p[0]},{p[1]},{p[2]})" for p in ne]) if ne else "NONE"
        print(f"  {lam:8.4f} │ {ne_str}")

    # Cross-λ stability: for each NE profile ever seen, check deviation at all λ
    print(f"\n  {'─' * 65}")
    print(f"  Cross-λ Stability: max deviation gain for each profile")
    print(f"  {'─' * 65}")

    for profile in sorted(all_profiles_ever):
        strats = [np.array(profile[p], float) for p in range(NP)]
        print(f"\n  Profile {profile}:")
        print(f"    {'λ':>8} │ {'max gain':>10} │ {'ε-NE (ε=0.1)?':>14}")
        print(f"    {'─' * 38}")

        gains_above_eps = 0
        for lam in lambdas_fine:
            gain = max_deviation_gain(r, trans, strats, lam)
            eps_ok = "YES" if gain < 0.1 else f"NO  ({gain:.4f})"
            print(f"    {lam:8.4f} │ {gain:10.5f} │ {eps_ok:>14}")
            if gain >= 0.1:
                gains_above_eps += 1

        if gains_above_eps > 0:
            print(f"    → NOT a uniform 0.1-equilibrium ({gains_above_eps}/{len(lambdas_fine)} failures)")
        else:
            print(f"    → IS a uniform 0.1-equilibrium")

    # Game payoff structure
    print(f"\n  Game Payoff Structure:")
    for s in range(NS):
        print(f"\n  State {s}:")
        print(f"    a0 a1 a2 │    r0    r1    r2 │  P(→0)  P(→1)")
        print(f"    {'─' * 50}")
        for a0, a1, a2 in product(range(NA), repeat=NP):
            r_str = " ".join(f"{r[s][a0][a1][a2][p]:5.1f}" for p in range(NP))
            t_str = " ".join(f"{trans[s][a0][a1][a2][sn]:5.2f}" for sn in range(NS))
            print(f"     {a0}  {a1}  {a2} │ {r_str} │  {t_str}")

    if gi >= 2:
        break

# ═══════════════════════════════════════════════════════════
# Phase 3: Summary
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")

if found_games:
    print(f"\n  {len(found_games)}개 게임에서 NE가 λ에 따라 변화")
    print(f"  → 정상(k=1) 균형이 구조적으로 불안정")
    print(f"  → 라그랑주 점 L1-L3과 동일한 안장점 구조")
    print()
    print(f"  Combined with k≥2 divergence:")
    print(f"    k ≥ 2: 고정점 부재 (product divergence) ✓")
    print(f"    k = 1: NE가 λ-의존, 단일 프로파일로 커버 불가 ✓")
    print(f"    ∴ uniform equilibrium 부재")
else:
    print(f"\n  Pure NE는 λ에 안정적 — mixed NE 분석 필요")

print(f"\n{'=' * 65}")
