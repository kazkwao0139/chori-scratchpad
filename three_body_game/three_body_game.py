"""
THREE-BODY PROBLEM OF GAME THEORY
3-Player Stochastic Game: Uniform Equilibrium Exploration

Neyman's Open Problem: Does every stochastic game with finitely many
players, states, and actions have a uniform equilibrium?

Approach:
  1. Minimal case: 3 players × 2 states × 2 actions
  2. "Rotating Crown" game: cyclic dominance structure
  3. Check stationary equilibria (k=1 automata)
  4. Check discount factor dependence (λ → 1)
  5. Check if k=2 automata can break k=1 equilibria (arms race)
"""

import numpy as np
from itertools import product
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

np.set_printoptions(precision=4, suppress=True)


# ═══════════════════════════════════════════════════════════════
# Game Definition: "Rotating Crown"
# ═══════════════════════════════════════════════════════════════
# State 0: Player 0 has the crown (positional advantage)
# State 1: Player 2 has the crown (rotated)
#
# Core: crown-holder extracts rents, but two others can
# coordinate to flip the state. Unstable coalition dynamics.

def make_game():
    """Returns payoffs[s][a0][a1][a2] and trans[s][a0][a1][a2]."""
    payoffs = np.zeros((2, 2, 2, 2, 3))
    trans = np.zeros((2, 2, 2, 2))  # P(next_state = 1)

    # State 0: Player 0 advantaged
    P0 = {
        (0,0,0): (3, 3, 3),   # peace
        (1,0,0): (5, 1, 2),   # p0 exploits
        (0,1,0): (2, 5, 1),   # p1 exploits
        (0,0,1): (1, 2, 5),   # p2 exploits
        (1,1,0): (4, 4, 0),   # p0+p1 vs p2
        (1,0,1): (4, 0, 4),   # p0+p2 vs p1
        (0,1,1): (0, 4, 4),   # p1+p2 vs p0 → flip trigger
        (1,1,1): (1, 1, 1),   # all-out war
    }
    T0 = {
        (0,0,0): 0.1,  # peace → stable
        (1,0,0): 0.2,  # p0 exploits → mostly stays
        (0,1,0): 0.3,  # p1 pushes
        (0,0,1): 0.4,  # p2 pushes harder
        (1,1,0): 0.2,  # p0 allied → stable
        (1,0,1): 0.5,  # coin flip
        (0,1,1): 0.8,  # p1+p2 → crown flips!
        (1,1,1): 0.6,  # chaos
    }

    for a, r in P0.items():
        payoffs[0][a[0]][a[1]][a[2]] = r
    for a, t in T0.items():
        trans[0][a[0]][a[1]][a[2]] = t

    # State 1: Player 2 has crown (rotate 0→1→2→0)
    for a0, a1, a2 in product(range(2), repeat=3):
        orig = payoffs[0][a2][a0][a1]        # rotated actions
        payoffs[1][a0][a1][a2] = [orig[2], orig[0], orig[1]]  # rotated payoffs
        trans[1][a0][a1][a2] = 1.0 - trans[0][a2][a0][a1]     # mirror

    return payoffs, trans


# ═══════════════════════════════════════════════════════════════
# Analytical Tools
# ═══════════════════════════════════════════════════════════════

def stationary_payoff(payoffs, trans, strats):
    """Average payoff for pure stationary strategies.
    strats[i] = (action_in_s0, action_in_s1)"""
    a = [[strats[i][s] for i in range(3)] for s in range(2)]
    r = [payoffs[s][a[s][0]][a[s][1]][a[s][2]] for s in range(2)]
    p01 = trans[0][a[0][0]][a[0][1]][a[0][2]]
    p10 = 1.0 - trans[1][a[1][0]][a[1][1]][a[1][2]]
    denom = p01 + p10
    pi0 = p10 / denom if denom > 1e-10 else 0.5
    return pi0 * r[0] + (1 - pi0) * r[1]


def discounted_payoff(payoffs, trans, strats, lam):
    """Discounted payoff starting from state 0."""
    a = [[strats[i][s] for i in range(3)] for s in range(2)]
    r = np.array([payoffs[s][a[s][0]][a[s][1]][a[s][2]] for s in range(2)])
    p01 = trans[0][a[0][0]][a[0][1]][a[0][2]]
    p11 = trans[1][a[1][0]][a[1][1]][a[1][2]]
    P = np.array([[1 - p01, p01], [1 - p11, p11]])
    v = (1 - lam) * np.linalg.solve(np.eye(2) - lam * P, r)
    return v[0]  # from state 0


# ═══════════════════════════════════════════════════════════════
# Finite State Automaton
# ═══════════════════════════════════════════════════════════════

class FSA:
    def __init__(self, k):
        self.k = k
        self.output = np.zeros((k, 2), dtype=int)       # [q][gs] → action
        self.trans = np.zeros((k, 2, 4), dtype=int)      # [q][gs][obs] → next_q

    @staticmethod
    def from_stationary(strategy):
        f = FSA(1)
        f.output[0, 0] = strategy[0]
        f.output[0, 1] = strategy[1]
        return f


def automata_payoff(payoffs, trans_p, automata):
    """Analytical average payoff for 3 automata via Markov chain."""
    k = [a.k for a in automata]
    n = k[0] * k[1] * k[2] * 2

    def enc(q0, q1, q2, gs):
        return ((q0 * k[1] + q1) * k[2] + q2) * 2 + gs

    P = np.zeros((n, n))
    R = np.zeros((n, 3))

    for q0, q1, q2, gs in product(range(k[0]), range(k[1]), range(k[2]), range(2)):
        idx = enc(q0, q1, q2, gs)
        acts = [automata[i].output[q0 if i == 0 else (q1 if i == 1 else q2), gs]
                for i in range(3)]
        R[idx] = payoffs[gs][acts[0]][acts[1]][acts[2]]

        obs = [acts[1]*2+acts[2], acts[0]*2+acts[2], acts[0]*2+acts[1]]
        nq = [automata[i].trans[
                  [q0, q1, q2][i], gs, obs[i]] for i in range(3)]

        p1 = trans_p[gs][acts[0]][acts[1]][acts[2]]
        P[idx, enc(nq[0], nq[1], nq[2], 0)] += 1 - p1
        P[idx, enc(nq[0], nq[1], nq[2], 1)] += p1

    # Stationary distribution
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
        if np.any(pi < -1e-8):
            raise ValueError
    except (np.linalg.LinAlgError, ValueError):
        evals, evecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(evals - 1.0))
        pi = np.abs(np.real(evecs[:, idx]))
        pi /= pi.sum()

    return pi @ R


# ═══════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════

def main():
    payoffs, trans = make_game()

    sep = "=" * 70
    print(sep)
    print("  THREE-BODY PROBLEM OF GAME THEORY")
    print("  3 players × 2 states × 2 actions: 'Rotating Crown'")
    print(sep)

    # ─── Game Structure ───
    for s in range(2):
        crown = "P0" if s == 0 else "P2"
        print(f"\n  State {s} (crown: {crown}):")
        print(f"  {'a0':>3} {'a1':>3} {'a2':>3} │ {'r0':>5} {'r1':>5} {'r2':>5} │ {'P(→1)':>6}")
        print(f"  {'─'*42}")
        for a0, a1, a2 in product(range(2), repeat=3):
            r = payoffs[s][a0][a1][a2]
            p = trans[s][a0][a1][a2]
            print(f"  {a0:3d} {a1:3d} {a2:3d} │ {r[0]:5.1f} {r[1]:5.1f} {r[2]:5.1f} │ {p:6.2f}")

    # ─── Phase 1: Stationary Equilibria ───
    print(f"\n{sep}")
    print("  Phase 1: Pure Stationary Nash Equilibria (k=1 automata)")
    print(sep)

    strats = list(product(range(2), repeat=2))  # 4 per player

    # Find equilibria & compute deviation gains for all profiles
    all_profiles = []
    for s0, s1, s2 in product(strats, repeat=3):
        prof = [s0, s1, s2]
        pay = stationary_payoff(payoffs, trans, prof)
        max_gain = 0
        for player in range(3):
            for alt in strats:
                if alt == prof[player]:
                    continue
                dp = list(prof)
                dp[player] = alt
                dv = stationary_payoff(payoffs, trans, dp)
                max_gain = max(max_gain, dv[player] - pay[player])
        all_profiles.append((prof, pay, max_gain))

    all_profiles.sort(key=lambda x: x[2])
    equilibria = [(p, r) for p, r, g in all_profiles if g < 1e-10]

    print(f"\n  Total profiles: 64")
    print(f"  Pure stationary Nash equilibria: {len(equilibria)}")

    if equilibria:
        for prof, pay in equilibria:
            print(f"    {prof} → ({pay[0]:.3f}, {pay[1]:.3f}, {pay[2]:.3f})")
    else:
        print(f"\n  No pure stationary equilibria! Top 5 closest:")
        for prof, pay, gain in all_profiles[:5]:
            print(f"    {prof} → ({pay[0]:.3f},{pay[1]:.3f},{pay[2]:.3f})"
                  f"  max_gain={gain:.4f}")

    # ─── Phase 2: Discount Factor Dependence ───
    print(f"\n{sep}")
    print("  Phase 2: Equilibria by Discount Factor λ")
    print("  (Does the equilibrium set converge as λ → 1?)")
    print(sep)

    lambdas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
    print(f"\n  {'λ':>8} │ {'#Eq':>4} │ {'Equilibrium payoffs'}")
    print(f"  {'─'*60}")

    for lam in lambdas:
        eq_list = []
        for s0, s1, s2 in product(strats, repeat=3):
            prof = [s0, s1, s2]
            pay = discounted_payoff(payoffs, trans, prof, lam)
            is_eq = True
            for player in range(3):
                for alt in strats:
                    if alt == prof[player]:
                        continue
                    dp = list(prof)
                    dp[player] = alt
                    dv = discounted_payoff(payoffs, trans, dp, lam)
                    if dv[player] > pay[player] + 1e-10:
                        is_eq = False
                        break
                if not is_eq:
                    break
            if is_eq:
                eq_list.append((prof, pay))

        pay_strs = "; ".join(f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" for _, p in eq_list[:3])
        suffix = "..." if len(eq_list) > 3 else ""
        print(f"  {lam:8.4f} │ {len(eq_list):4d} │ {pay_strs}{suffix}")

    # ─── Phase 3: k=2 Automata Deviations ───
    print(f"\n{sep}")
    print("  Phase 3: k=2 Automata vs k=1 Equilibria (Arms Race)")
    print(sep)

    # Use top-5 most stable profiles as test candidates
    test_profiles = all_profiles[:5]

    print(f"\n  Testing 50,000 random k=2 automata per player per profile...")

    for prof, pay, base_gain in test_profiles[:3]:
        print(f"\n  Profile {prof}")
        print(f"  Base payoff: ({pay[0]:.3f}, {pay[1]:.3f}, {pay[2]:.3f}), "
              f"k=1 max_gain={base_gain:.4f}")

        base_auto = [FSA.from_stationary(s) for s in prof]

        for player in range(3):
            best_gain = 0
            best_auto = None
            n_positive = 0

            for _ in range(50000):
                dev = FSA(2)
                dev.output = np.random.randint(0, 2, (2, 2))
                dev.trans = np.random.randint(0, 2, (2, 2, 4))

                test_a = list(base_auto)
                test_a[player] = dev

                try:
                    dp = automata_payoff(payoffs, trans, test_a)
                    g = dp[player] - pay[player]
                    if g > 1e-10:
                        n_positive += 1
                    if g > best_gain:
                        best_gain = g
                        best_auto = (dev.output.copy(), dev.trans.copy())
                except:
                    continue

            print(f"    P{player}: best k=2 gain = {best_gain:+.4f}"
                  f"  ({n_positive}/50000 positive = {100*n_positive/50000:.1f}%)")
            if best_auto and best_gain > 0.01:
                print(f"         output={best_auto[0].tolist()}, "
                      f"trans={best_auto[1].tolist()}")

    # ─── Phase 4: Lyapunov Exponent (Chaos Check) ───
    print(f"\n{sep}")
    print("  Phase 4: Sensitivity to Game Length (Chaos Signature)")
    print(sep)

    print(f"\n  Simulating best-response dynamics for T-stage games...")
    print(f"  If optimal strategy depends on T → no uniform equilibrium")

    T_values = [10, 50, 100, 500, 1000, 5000]
    n_sim = 2000

    # For each T, find the best stationary strategy for each player
    # given the others play the "most stable" profile
    base_prof = all_profiles[0][0]

    print(f"\n  Base profile: {base_prof}")
    print(f"  {'T':>6} │ {'BR_P0':>8} {'BR_P1':>8} {'BR_P2':>8} │ "
          f"{'Pay_P0':>7} {'Pay_P1':>7} {'Pay_P2':>7}")
    print(f"  {'─'*65}")

    for T in T_values:
        best_responses = []
        best_payoffs = []

        for player in range(3):
            best_s = None
            best_p = -999

            for alt in strats:
                dp = list(base_prof)
                dp[player] = alt

                # Simulate T-stage game multiple times
                total = 0
                a_base = [FSA.from_stationary(s) for s in dp]

                for _ in range(n_sim):
                    gs = 0
                    cumr = 0
                    for t in range(T):
                        acts = [a_base[i].output[0, gs] for i in range(3)]
                        r = payoffs[gs][acts[0]][acts[1]][acts[2]]
                        cumr += r[player]
                        p1 = trans[gs][acts[0]][acts[1]][acts[2]]
                        gs = 1 if np.random.random() < p1 else 0
                    total += cumr / T

                avg = total / n_sim
                if avg > best_p:
                    best_p = avg
                    best_s = alt

            best_responses.append(best_s)
            best_payoffs.append(best_p)

        print(f"  {T:6d} │ {str(best_responses[0]):>8} "
              f"{str(best_responses[1]):>8} {str(best_responses[2]):>8} │ "
              f"{best_payoffs[0]:7.3f} {best_payoffs[1]:7.3f} {best_payoffs[2]:7.3f}")

    # ─── Phase 5: 2-Player Comparison ───
    print(f"\n{sep}")
    print("  Phase 5: 2-Player Control (Should Be Stable)")
    print(sep)
    print(f"\n  Same game structure but only 2 players (P0 vs P1).")
    print(f"  P2's action fixed at 0. Checking equilibrium stability...")

    for s0, s1 in product(strats, repeat=2):
        prof = [s0, s1, (0, 0)]  # fix P2
        pay = stationary_payoff(payoffs, trans, prof)

        is_eq = True
        for player in range(2):  # only check P0, P1
            for alt in strats:
                if alt == prof[player]:
                    continue
                dp = list(prof)
                dp[player] = alt
                dv = stationary_payoff(payoffs, trans, dp)
                if dv[player] > pay[player] + 1e-10:
                    is_eq = False
                    break
            if not is_eq:
                break

        if is_eq:
            print(f"    2P Eq: P0={s0}, P1={s1} → "
                  f"({pay[0]:.3f}, {pay[1]:.3f}, {pay[2]:.3f})")

    print(f"\n{'='*70}")
    print(f"  Analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
