"""Nash Equilibrium 보상행렬 기반 전략 솔버.

매 턴을 n×m 노말 폼 게임으로 모델링:
  - 행: 내 합법 액션 (n개)
  - 열: 상대 합법 액션 (m개)
  - 값: value_net(next_state) -내 관점 형세

2인 제로섬 게임의 Nash 균형 → 최적 혼합전략 + game_value.

3단계 전략 시스템:
  - 미드게임: 기존 1턴 Nash + Value Net (기본)
  - 엔드게임: 남은 포켓몬 합계 ≤ 3이면 멀티턴 완전탐색 Nash
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

import random as _rng

from battle_sim import BattleSimulator, BattleState, Pokemon
from data_loader import GameData, _to_id


# ═══════════════════════════════════════════════════════════════
#  페이즈 감지
# ═══════════════════════════════════════════════════════════════

def detect_phase(state: BattleState) -> str:
    """현재 배틀 상태의 페이즈를 판별.

    Returns:
        "endgame" - 남은 포켓몬 합계 <= 3
        "midgame" - 그 외
    """
    total_alive = state.sides[0].alive_count + state.sides[1].alive_count
    if total_alive <= 3:
        return "endgame"
    return "midgame"


# ═══════════════════════════════════════════════════════════════
#  LP 기반 Nash 균형 솔버
# ═══════════════════════════════════════════════════════════════

def solve_zero_sum_lp(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """2인 제로섬 게임의 Nash 균형을 LP로 풀기.

    행 플레이어(P1) 관점 maximin:
        maximize v
        s.t.  M^T · p >= v·1   (모든 상대 액션에 대해 최소 v 보장)
              sum(p) = 1, p >= 0

    Args:
        M: (n, m) 보상행렬 -P1 관점 (P1이 행, P2가 열)

    Returns:
        p1_strategy: (n,) P1 혼합전략
        p2_strategy: (m,) P2 혼합전략
        game_value: float -P1 관점 게임밸류
    """
    n, m = M.shape

    # 엣지 케이스: 단일 액션
    if n == 1 and m == 1:
        return np.array([1.0]), np.array([1.0]), float(M[0, 0])
    if n == 1:
        # P1 한 수 -P2는 최소화
        val = float(M[0, :].min())
        p2 = np.zeros(m)
        p2[int(M[0, :].argmin())] = 1.0
        return np.array([1.0]), p2, val
    if m == 1:
        # P2 한 수 -P1은 최대화
        val = float(M[:, 0].max())
        p1 = np.zeros(n)
        p1[int(M[:, 0].argmax())] = 1.0
        return p1, np.array([1.0]), val

    # ── P1 maximin LP ──
    # 변수: [p_0, ..., p_{n-1}, v]  (n+1 변수)
    # minimize -v  (= maximize v)
    c = np.zeros(n + 1)
    c[-1] = -1.0  # minimize -v

    # 부등식 제약: -M^T · p + v·1 <= 0
    # i.e. for each j: -sum_i M[i,j]*p_i + v <= 0
    A_ub = np.zeros((m, n + 1))
    A_ub[:, :n] = -M.T          # -M^T · p
    A_ub[:, -1] = 1.0           # +v
    b_ub = np.zeros(m)

    # 등식 제약: sum(p) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # 범위: p_i >= 0, v는 제한 없음
    bounds = [(0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        p1 = result.x[:n]
        game_value = result.x[-1]

        # P1 전략 정규화 (수치 오차 보정)
        p1 = np.maximum(p1, 0)
        p_sum = p1.sum()
        if p_sum > 0:
            p1 /= p_sum
        else:
            p1 = np.ones(n) / n

        # ── P2 maximin LP (대칭) ──
        # P2: maximize u  s.t. M · q >= u·1, sum(q) = 1, q >= 0
        # = P1 maximin on -M^T
        p2 = _solve_p2_lp(M)
        return p1, p2, float(game_value)

    # LP 실패 → fictitious play 폴백
    return _fictitious_play(M, iterations=100)


def _solve_p2_lp(M: np.ndarray) -> np.ndarray:
    """P2의 최적 혼합전략을 LP로 풀기."""
    n, m = M.shape

    # P2 minimax: minimize u  s.t. M · q <= u·1, sum(q) = 1, q >= 0
    # 변수: [q_0, ..., q_{m-1}, u]
    c = np.zeros(m + 1)
    c[-1] = 1.0  # minimize u

    # M · q - u·1 <= 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = M
    A_ub[:, -1] = -1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        q = result.x[:m]
        q = np.maximum(q, 0)
        q_sum = q.sum()
        if q_sum > 0:
            q /= q_sum
        else:
            q = np.ones(m) / m
        return q

    return np.ones(m) / m


def solve_robust_nash_lp(
    matrices: list[np.ndarray],
    weights: list[float],
) -> tuple[np.ndarray, float]:
    """Bayesian Nash LP — K개 상대 타입(세트)에 대해 robust한 단일 전략.

    상대가 K개 타입 중 하나이며, 상대는 자기 타입을 알고 최적 대응.
    나는 모든 타입에 robust한 단일 혼합전략을 구한다.

    정식화:
        maximize  Σ_k  w_k × u_k
        s.t.  M_k^T × p  ≥  u_k × 1   (각 타입 k마다)
              Σ p_i = 1,  p ≥ 0,  u_k free

    Args:
        matrices: K개 보상행렬. matrices[k] shape=(n, m_k), m_k 다를 수 있음.
                  모든 행렬은 동일한 n (내 액션 수)을 가져야 함.
        weights: K개 가중치 (확률). 합=1.0이어야 함.

    Returns:
        p: (n,) 내 혼합전략
        game_value: 가중 보장값
    """
    K = len(matrices)
    if K == 0:
        return np.array([1.0]), 0.0
    if K == 1:
        p1, _, gv = solve_zero_sum_lp(matrices[0])
        return p1, gv

    n = matrices[0].shape[0]

    # 변수: [p_0, ..., p_{n-1}, u_0, ..., u_{K-1}]  (n + K 변수)
    num_vars = n + K

    # 목적함수: maximize Σ_k w_k * u_k  →  minimize -Σ_k w_k * u_k
    c = np.zeros(num_vars)
    for k in range(K):
        c[n + k] = -weights[k]

    # 부등식 제약: M_k^T * p >= u_k * 1 (각 타입 k, 각 상대 액션 j)
    # → -M_k^T * p + u_k <= 0
    total_constraints = sum(m.shape[1] for m in matrices)
    A_ub = np.zeros((total_constraints, num_vars))
    b_ub = np.zeros(total_constraints)

    row = 0
    for k, M_k in enumerate(matrices):
        m_k = M_k.shape[1]
        for j in range(m_k):
            # -M_k[i, j] * p_i  +  u_k  <=  0
            A_ub[row, :n] = -M_k[:, j]
            A_ub[row, n + k] = 1.0
            row += 1

    # 등식 제약: Σ p_i = 1
    A_eq = np.zeros((1, num_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # 범위: p_i >= 0, u_k free
    bounds = [(0, None)] * n + [(None, None)] * K

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        p = result.x[:n]
        u = result.x[n:]

        # 정규화
        p = np.maximum(p, 0)
        p_sum = p.sum()
        if p_sum > 0:
            p /= p_sum
        else:
            p = np.ones(n) / n

        game_value = sum(weights[k] * u[k] for k in range(K))
        return p, float(game_value)

    # LP 실패 → 단순 가중 평균 행렬로 폴백
    m_max = max(m.shape[1] for m in matrices)
    M_avg = np.zeros((n, m_max))
    for k, M_k in enumerate(matrices):
        m_k = M_k.shape[1]
        M_avg[:, :m_k] += weights[k] * M_k
    p1, _, gv = solve_zero_sum_lp(M_avg)
    return p1, gv


def build_variant_payoff_matrix(
    state: BattleState,
    player: int,
    sim: BattleSimulator,
    evaluator,
    opp_variants: list[tuple[BattleState, float]],
    gd: GameData | None = None,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """K개 변형 상태 각각에 대해 payoff matrix 생성.

    내 legal_actions는 동일 (내 포켓몬은 안 바뀜).
    상대 legal_actions는 변형마다 다를 수 있음 (다른 기술셋).
    → 각각 독립 행렬 생성, solve_robust_nash_lp()에 전달.

    Args:
        state: 기본 배틀 상태 (내 쪽 정보용)
        player: 행렬의 행 플레이어
        sim: BattleSimulator
        evaluator: NetworkEvaluator
        opp_variants: [(variant_state, weight), ...] K개 변형
        gd: GameData

    Returns:
        matrices: K개 보상행렬
        weights_list: K개 가중치
        legal_me: 내 합법 액션 리스트 (공통)
    """
    legal_me = sim.get_legal_actions(state, player)
    matrices = []
    weights_list = []

    for variant_state, weight in opp_variants:
        M_k, _, _ = build_payoff_matrix(
            variant_state, player, sim, evaluator, gd=gd)
        matrices.append(M_k)
        weights_list.append(weight)

    return matrices, weights_list, legal_me


def _fictitious_play(M: np.ndarray, iterations: int = 100
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Fictitious play 폴백 -LP 실패 시 사용."""
    n, m = M.shape
    p1_counts = np.zeros(n)
    p2_counts = np.zeros(m)

    # 초기 균등 시작
    p1_counts[0] = 1
    p2_counts[0] = 1

    for t in range(iterations):
        # P1: P2의 경험적 분포에 대해 best response
        p2_freq = p2_counts / p2_counts.sum()
        payoffs_p1 = M @ p2_freq  # (n,)
        br1 = int(payoffs_p1.argmax())
        p1_counts[br1] += 1

        # P2: P1의 경험적 분포에 대해 best response
        p1_freq = p1_counts / p1_counts.sum()
        payoffs_p2 = M.T @ p1_freq  # (m,)
        br2 = int(payoffs_p2.argmin())
        p2_counts[br2] += 1

    p1 = p1_counts / p1_counts.sum()
    p2 = p2_counts / p2_counts.sum()
    game_value = float(p1 @ M @ p2)

    return p1, p2, game_value


# ═══════════════════════════════════════════════════════════════
#  Sweep Score — 목표 지향 평가
# ═══════════════════════════════════════════════════════════════

SWEEP_ALPHA = 0.0  # sweep_score 블렌딩 비중 (0=끔)

# Progress bonus: value_net tie일 때 데미지 진전 우선
# 상대 전체 HP 손실 비율 × α → tie-break only (전략 구조 불변)
PROGRESS_ALPHA = 0.01


def _atk_dict(poke: Pokemon) -> dict:
    """Pokemon → calc_damage 공격자 dict 변환."""
    return {
        "types": poke.types,
        "stats": poke.stats,
        "ability": poke.ability,
        "item": poke.item if not poke.item_consumed else "",
        "status": poke.status,
        "boosts": poke.boosts,
        "original_types": poke.types,
    }


def _def_dict(poke: Pokemon) -> dict:
    """Pokemon → calc_damage 방어자 dict 변환."""
    return {
        "types": poke.types,
        "stats": poke.stats,
        "ability": poke.ability,
        "item": poke.item if not poke.item_consumed else "",
        "status": poke.status,
        "boosts": poke.boosts,
        "cur_hp": poke.cur_hp,
    }


def _best_damage(attacker: Pokemon, defender: Pokemon, sim: BattleSimulator,
                 gd: GameData) -> int:
    """attacker가 defender에게 줄 수 있는 최대 평균 데미지 (HP 무관)."""
    best_dmg = 0
    for move_id in attacker.moves:
        move = gd.get_move(move_id)
        if not move or move.get("basePower", 0) == 0:
            continue
        _, _, avg = sim.dc.calc_damage(
            _atk_dict(attacker), _def_dict(defender), move)
        best_dmg = max(best_dmg, avg)
    return best_dmg


def _precompute_damage_table(state: BattleState, player: int,
                             sim: BattleSimulator, gd: GameData
                             ) -> dict[tuple[int, int], int]:
    """(my_team_idx, opp_team_idx) → best_damage 테이블 사전 계산.

    calc_damage는 스탯/타입에만 의존하므로 한 번만 계산하면 됨.
    next_state에서는 HP만 확인하면 됨.
    """
    opp = 1 - player
    table = {}
    for mi, my_poke in enumerate(state.sides[player].team):
        if my_poke.fainted:
            continue
        for oi, opp_poke in enumerate(state.sides[opp].team):
            if opp_poke.fainted:
                continue
            table[(mi, oi)] = _best_damage(my_poke, opp_poke, sim, gd)
    return table


def sweep_score_with_table(state: BattleState, player: int,
                           dmg_table: dict[tuple[int, int], int]) -> float:
    """사전 계산된 damage table로 sweep_score 계산. calc_damage 호출 없음."""
    opp = 1 - player
    my_alive = [(i, p) for i, p in enumerate(state.sides[player].team)
                if not p.fainted]
    opp_alive = [(i, p) for i, p in enumerate(state.sides[opp].team)
                 if not p.fainted]

    if not opp_alive:
        return 1.0
    if not my_alive:
        return 0.0

    best = 0.0
    for mi, my_poke in my_alive:
        kills = 0
        for oi, opp_poke in opp_alive:
            dmg = dmg_table.get((mi, oi), 0)
            if dmg > 0:
                turns_to_ko = (opp_poke.cur_hp + dmg - 1) // dmg
                if turns_to_ko <= 2:
                    kills += 1
        ratio = kills / len(opp_alive)
        best = max(best, ratio)

    return best


# ═══════════════════════════════════════════════════════════════
#  보상행렬 구성
# ═══════════════════════════════════════════════════════════════

def build_payoff_matrix(
    state: BattleState,
    player: int,
    sim: BattleSimulator,
    evaluator,
    gd: GameData | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """n×m 보상행렬 구성 (sim.step + value_net 배치 평가 + sweep_score 블렌딩).

    Args:
        state: 현재 배틀 상태
        player: 행렬의 행 플레이어 (0 or 1)
        sim: BattleSimulator
        evaluator: NetworkEvaluator (evaluate_values_batch 메서드 필요)
        gd: GameData (sweep_score용, None이면 블렌딩 스킵)

    Returns:
        M: (n, m) 보상행렬 -player 관점
        legal_me: n개 합법 액션 리스트
        legal_opp: m개 합법 액션 리스트
    """
    opp = 1 - player
    legal_me = sim.get_legal_actions(state, player)
    legal_opp = sim.get_legal_actions(state, opp)

    # 프루닝: 명백히 나쁜 행동 제거
    if hasattr(evaluator, 'prune_obvious'):
        legal_me = evaluator.prune_obvious(state, player, legal_me)
        legal_opp = evaluator.prune_obvious(state, opp, legal_opp)

    n = len(legal_me)
    m = len(legal_opp)

    # n×m 다음 상태 생성
    next_states = []
    terminal_mask = []   # True if terminal

    for a_me in legal_me:
        for a_opp in legal_opp:
            if player == 0:
                ns = sim.step(state, a_me, a_opp)
            else:
                ns = sim.step(state, a_opp, a_me)
            next_states.append(ns)
            terminal_mask.append(ns.is_terminal)

    # 배치 평가
    values = np.zeros(n * m, dtype=np.float32)

    # Terminal 상태: 직접 +1/-1 할당
    for i, ns in enumerate(next_states):
        if ns.is_terminal:
            if ns.winner == player:
                values[i] = 1.0
            elif ns.winner == opp:
                values[i] = -1.0
            else:
                values[i] = 0.0

    # Non-terminal 상태: value_net 배치 평가
    non_terminal_indices = [i for i, t in enumerate(terminal_mask) if not t]
    if non_terminal_indices:
        non_terminal_states = [next_states[i] for i in non_terminal_indices]
        batch_values = evaluator.evaluate_values_batch(
            non_terminal_states, player)
        for idx, val in zip(non_terminal_indices, batch_values):
            values[idx] = val

    # Sweep score 블렌딩: damage table 1회 계산, next_state마다 HP만 체크
    if gd is not None and non_terminal_indices and SWEEP_ALPHA > 0:
        dmg_table = _precompute_damage_table(state, player, sim, gd)
        for idx in non_terminal_indices:
            ns = next_states[idx]
            ss = sweep_score_with_table(ns, player, dmg_table)
            sweep_bonus = ss * 2.0 - 1.0  # [0,1] → [-1,+1]
            values[idx] = ((1 - SWEEP_ALPHA) * values[idx]
                           + SWEEP_ALPHA * sweep_bonus)

    # Progress bonus: 상대 HP 손실 비율로 tie-break
    # 전략 구조를 바꾸지 않되, 동일 가치 액션 중 데미지 진전 우선
    if PROGRESS_ALPHA > 0:
        opp_side = state.sides[1 - player]
        base_hp_sum = sum(max(p.max_hp, 1) for p in opp_side.team)
        base_hp_cur = sum(p.cur_hp for p in opp_side.team)

        for idx in non_terminal_indices:
            ns = next_states[idx]
            ns_opp = ns.sides[1 - player]
            ns_hp_cur = sum(p.cur_hp for p in ns_opp.team)
            hp_lost = (base_hp_cur - ns_hp_cur) / base_hp_sum
            values[idx] += PROGRESS_ALPHA * hp_lost

    M = values.reshape(n, m)
    return M, legal_me, legal_opp


# ═══════════════════════════════════════════════════════════════
#  Belief-weighted 보상행렬 (불완전정보)
# ═══════════════════════════════════════════════════════════════

def _sample_belief_pokemon(pokemon: Pokemon, gd: GameData,
                           revealed_moves: set[str] | None = None,
                           ) -> Pokemon:
    """Smogon 통계 기반으로 미공개 정보를 샘플링한 포켓몬 복제본 생성.

    - 공개된 기술은 유지, 미공개 슬롯은 Smogon 사용률 기반 샘플
    - 아이템도 미확인이면 Smogon 기반 샘플
    """
    stats = gd.get_stats(pokemon.name, "bss")
    if not stats:
        return pokemon.clone()

    clone = pokemon.clone()
    revealed = revealed_moves or set()

    # 기술 샘플링: 공개된 기술 유지, 나머지 Smogon 기반 샘플
    known_moves = [m for m in clone.moves if m in revealed]
    need = 4 - len(known_moves)

    if need > 0 and stats["moves"]:
        # 이미 공개된 기술 제외한 후보
        candidates = {m: w for m, w in stats["moves"].items()
                      if m not in revealed}
        if candidates:
            move_ids = list(candidates.keys())
            weights = [candidates[m] for m in move_ids]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                sampled = []
                for _ in range(need):
                    pick = _rng.choices(move_ids, weights=weights, k=1)[0]
                    if pick not in sampled and pick not in known_moves:
                        sampled.append(pick)
                    else:
                        # 중복 시 다시 뽑기 (최대 한번)
                        remaining = [m for m in move_ids
                                     if m not in sampled and m not in known_moves]
                        if remaining:
                            rw = [candidates[m] for m in remaining]
                            rt = sum(rw)
                            if rt > 0:
                                rw = [w / rt for w in rw]
                            pick2 = _rng.choices(remaining, weights=rw, k=1)[0]
                            sampled.append(pick2)
                clone.moves = known_moves + sampled
                # 4개로 맞추기
                clone.moves = clone.moves[:4]

    # 아이템 샘플링: 아이템이 아직 소모/사용되지 않았으면 Smogon 기반
    if stats["items"] and not pokemon.item_consumed:
        item_candidates = list(stats["items"].keys())
        item_weights = [stats["items"][i] for i in item_candidates]
        total = sum(item_weights)
        if total > 0:
            item_weights = [w / total for w in item_weights]
            clone.item = _rng.choices(item_candidates, weights=item_weights, k=1)[0]

    return clone


def build_belief_payoff_matrix(
    state: BattleState,
    player: int,
    sim: BattleSimulator,
    evaluator,
    gd: GameData,
    revealed_moves: dict[int, set[str]] | None = None,
    n_samples: int = 3,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Belief-weighted 보상행렬: 상대의 미공개 정보를 샘플링하여 평균.

    Args:
        state: 현재 배틀 상태
        player: 행렬의 행 플레이어
        sim: BattleSimulator
        evaluator: NetworkEvaluator
        gd: GameData (Smogon 통계 접근용)
        revealed_moves: {team_index: {revealed_move_ids}} 상대가 공개한 기술
        n_samples: belief 샘플 수 (많을수록 정확, 느림)

    Returns:
        M_avg: 평균 보상행렬
        legal_me, legal_opp: 합법 액션 리스트
    """
    opp = 1 - player
    opp_side = state.sides[opp]
    revealed = revealed_moves or {}

    # 첫 번째: 원본으로 legal actions 확정 (행렬 크기 결정)
    legal_me = sim.get_legal_actions(state, player)
    legal_opp = sim.get_legal_actions(state, opp)
    n = len(legal_me)
    m = len(legal_opp)

    M_sum = np.zeros((n, m), dtype=np.float64)

    for _ in range(n_samples):
        # 상대 팀을 belief 샘플로 교체한 state 복제
        belief_state = state.clone()
        belief_side = belief_state.sides[opp]
        for idx, poke in enumerate(belief_side.team):
            poke_revealed = revealed.get(idx, set())
            sampled = _sample_belief_pokemon(poke, gd, poke_revealed)
            belief_side.team[idx] = sampled

        # 이 belief state로 보상행렬 구성 (sweep_score 포함)
        M_sample, _, _ = build_payoff_matrix(
            belief_state, player, sim, evaluator, gd=gd)

        # 크기 맞는지 확인 (legal actions는 동일해야 함)
        if M_sample.shape == (n, m):
            M_sum += M_sample
        else:
            # legal actions가 달라진 경우 원본 사용
            M_orig, _, _ = build_payoff_matrix(state, player, sim, evaluator,
                                               gd=gd)
            M_sum += M_orig

    M_avg = M_sum / n_samples
    return M_avg.astype(np.float32), legal_me, legal_opp


# ═══════════════════════════════════════════════════════════════
#  NashSolver 클래스
# ═══════════════════════════════════════════════════════════════

class NashSolver:
    """매 턴 Nash 균형 기반 최적 혼합전략 계산."""

    def __init__(self, sim: BattleSimulator, gd, evaluator,
                 dirichlet_alpha: float = 0.15,
                 dirichlet_weight: float = 0.25,
                 epsilon_greedy: float = 0.05,
                 use_belief: bool = False,
                 belief_samples: int = 3,
                 endgame_solver=None,
                 midgame_depth: int = 2,
                 move_time_budget: float = 0):
        """
        Args:
            move_time_budget: 착수 시간 예산 (초). 0이면 고정 깊이 사용.
                              >0이면 iterative deepening (미드+엔드 공통).
        """
        self.sim = sim
        self.gd = gd
        self.evaluator = evaluator
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.epsilon_greedy = epsilon_greedy
        self.use_belief = use_belief
        self.belief_samples = belief_samples
        self.endgame_solver = endgame_solver
        self.midgame_depth = midgame_depth
        self.move_time_budget = move_time_budget
        # 각 사이드별 공개된 기술 추적: {side: {team_idx: set(move_ids)}}
        self.revealed_moves: dict[int, dict[int, set[str]]] = {0: {}, 1: {}}

    def observe_move(self, side: int, team_idx: int, move_id: str):
        """기술 사용을 관측하여 revealed_moves 갱신."""
        if team_idx not in self.revealed_moves[side]:
            self.revealed_moves[side][team_idx] = set()
        self.revealed_moves[side][team_idx].add(move_id)

    def reset_revealed(self):
        """새 게임 시작 시 공개 정보 초기화."""
        self.revealed_moves = {0: {}, 1: {}}

    def get_nash_strategies_both(
        self, state: BattleState, add_noise: bool = True,
        opp_variants: list[tuple[BattleState, float]] | None = None,
    ) -> tuple[dict[int, float], dict[int, float], float,
               dict[int, float], dict[int, float],
               dict[int, float], dict[int, float]]:
        """한 번의 행렬 구성으로 양쪽 전략 + 게임밸류 + CFR + reversal 반환.

        Args:
            state: 현재 배틀 상태
            add_noise: Dirichlet + ε-greedy 탐색 노이즈 적용 여부
            opp_variants: [(variant_state, weight), ...] 상대 세트 변형.
                          None 또는 len<=1이면 기존 단일 상태 사용.

        Returns:
            p1_dict: {action: probability} - P1 혼합전략
            p2_dict: {action: probability} - P2 혼합전략
            game_value: float - P1 관점 게임밸류 [-1, +1]
            p1_regret: {action: regret} - P1 CFR regret per action
            p2_regret: {action: regret} - P2 CFR regret per action
            p1_reversal: {action: max_j M[i,j]} - P1 역전 포텐셜
            p2_reversal: {action: max_j (-M)[j,i]} - P2 역전 포텐셜
        """
        # ── 시간 기반 iterative deepening (미드+엔드 공통) ──
        if self.move_time_budget > 0 and self.endgame_solver is not None:
            return self._timed_dispatch(state, add_noise,
                                        opp_variants=opp_variants)

        # ── 고정 깊이 모드 ──
        phase = detect_phase(state)
        if phase == "endgame" and self.endgame_solver is not None:
            return self._endgame_dispatch(state, add_noise,
                                          opp_variants=opp_variants)

        if self.endgame_solver is not None:
            return self._midgame_2turn(state, add_noise,
                                       opp_variants=opp_variants)

        # 폴백: 기존 1턴 보상행렬 (endgame_solver 없을 때)
        if self.use_belief:
            M, legal_p1, legal_p2 = build_belief_payoff_matrix(
                state, 0, self.sim, self.evaluator, self.gd,
                revealed_moves=self.revealed_moves[1],
                n_samples=self.belief_samples)
        else:
            M, legal_p1, legal_p2 = build_payoff_matrix(
                state, 0, self.sim, self.evaluator, gd=self.gd)

        n = len(legal_p1)
        m = len(legal_p2)

        # Nash 균형 풀기
        p1_strat, p2_strat, game_value = solve_zero_sum_lp(M)

        # ── CFR counterfactual values ──
        cf_p1 = M @ p2_strat
        regret_p1 = cf_p1 - game_value

        cf_p2 = -(M.T @ p1_strat)
        p2_value = -game_value
        regret_p2 = cf_p2 - p2_value

        # ── Reversal potential: 각 행동의 best-case 결과 ──
        # P1: max_j M[i,j] — 상대가 최악의 수를 뒀을 때의 결과
        rev_p1 = M.max(axis=1)            # shape (n,)
        # P2: max_i (-M)[j,i] = -min_i M[i,j]
        rev_p2 = (-M).max(axis=0)         # shape (m,)

        # 탐색 노이즈 적용
        if add_noise:
            p1_strat = self._add_exploration_noise(p1_strat)
            p2_strat = self._add_exploration_noise(p2_strat)

        # dict로 변환
        p1_dict = {a: float(p) for a, p in zip(legal_p1, p1_strat)}
        p2_dict = {a: float(p) for a, p in zip(legal_p2, p2_strat)}
        p1_regret = {a: float(r) for a, r in zip(legal_p1, regret_p1)}
        p2_regret = {a: float(r) for a, r in zip(legal_p2, regret_p2)}
        p1_reversal = {a: float(r) for a, r in zip(legal_p1, rev_p1)}
        p2_reversal = {a: float(r) for a, r in zip(legal_p2, rev_p2)}

        return (p1_dict, p2_dict, float(game_value),
                p1_regret, p2_regret, p1_reversal, p2_reversal)

    def _timed_dispatch(
        self, state: BattleState, add_noise: bool,
        opp_variants: list[tuple[BattleState, float]] | None = None,
    ) -> tuple[dict[int, float], dict[int, float], float,
               dict[int, float], dict[int, float],
               dict[int, float], dict[int, float]]:
        """시간 기반 iterative deepening. 미드+엔드 공통.

        시간 예산 내에서 depth=1부터 점점 깊게 탐색.
        확정(|gv|>0.99)이면 조기 종료.
        """
        p1_dict, p2_dict, game_value, depth = \
            self.endgame_solver.solve_both_timed(
                state, time_budget=self.move_time_budget,
                opp_variants=opp_variants)

        # 탐색 노이즈
        if add_noise and p1_dict:
            actions = list(p1_dict.keys())
            probs = np.array([p1_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p1_dict = {a: float(p) for a, p in zip(actions, probs)}

        if add_noise and p2_dict:
            actions = list(p2_dict.keys())
            probs = np.array([p2_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p2_dict = {a: float(p) for a, p in zip(actions, probs)}

        p1_regret = {a: 0.0 for a in p1_dict}
        p2_regret = {a: 0.0 for a in p2_dict}
        p1_reversal = {a: 0.0 for a in p1_dict}
        p2_reversal = {a: 0.0 for a in p2_dict}

        return (p1_dict, p2_dict, float(game_value),
                p1_regret, p2_regret, p1_reversal, p2_reversal)

    def _midgame_2turn(
        self, state: BattleState, add_noise: bool,
        opp_variants: list[tuple[BattleState, float]] | None = None,
    ) -> tuple[dict[int, float], dict[int, float], float,
               dict[int, float], dict[int, float],
               dict[int, float], dict[int, float]]:
        """미드게임 iterative deepening 탐색.

        Stockfish 방식: 얕은 depth부터 시작, 결과가 불명확하면 deeper.
        - |GV| > 0.9 → 명확, 즉시 종료
        - 최선수 안정 + GV 변동 < 0.1 → 수렴, 종료
        - 최대 midgame_depth까지 탐색
        """
        solver = self.endgame_solver
        saved_depth = solver.max_depth

        # depth 2부터 midgame_depth까지 2씩 증가
        depths = list(range(2, self.midgame_depth + 1, 2))
        if not depths:
            depths = [self.midgame_depth]
        # 마지막이 midgame_depth가 아니면 추가
        if depths[-1] < self.midgame_depth:
            depths.append(self.midgame_depth)

        p1_dict, p2_dict, game_value = {}, {}, 0.0
        prev_gv = None
        prev_best = None

        try:
            for d in depths:
                solver.max_depth = d
                p1_dict, p2_dict, game_value = solver.solve_both(
                    state, opp_variants=opp_variants)

                # 현재 최선수
                best_action = max(p1_dict, key=p1_dict.get) if p1_dict else None

                # 조기 종료 조건
                # 1) 명확한 포지션: 거의 이김/짐
                if abs(game_value) > 0.9:
                    break

                # 2) 수렴: 최선수 동일 + GV 변동 작음
                if (prev_gv is not None
                        and prev_best == best_action
                        and abs(game_value - prev_gv) < 0.1):
                    break

                prev_gv = game_value
                prev_best = best_action
        finally:
            solver.max_depth = saved_depth

        # 탐색 노이즈
        if add_noise and p1_dict:
            actions = list(p1_dict.keys())
            probs = np.array([p1_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p1_dict = {a: float(p) for a, p in zip(actions, probs)}

        if add_noise and p2_dict:
            actions = list(p2_dict.keys())
            probs = np.array([p2_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p2_dict = {a: float(p) for a, p in zip(actions, probs)}

        p1_regret = {a: 0.0 for a in p1_dict}
        p2_regret = {a: 0.0 for a in p2_dict}
        p1_reversal = {a: 0.0 for a in p1_dict}
        p2_reversal = {a: 0.0 for a in p2_dict}

        return (p1_dict, p2_dict, float(game_value),
                p1_regret, p2_regret, p1_reversal, p2_reversal)

    def _endgame_dispatch(
        self, state: BattleState, add_noise: bool,
        opp_variants: list[tuple[BattleState, float]] | None = None,
    ) -> tuple[dict[int, float], dict[int, float], float,
               dict[int, float], dict[int, float],
               dict[int, float], dict[int, float]]:
        """엔드게임 솔버로 양쪽 전략을 계산. 반환 형식은 7-tuple 동일.

        solve_both()로 1회만 풀어서 P1/P2 전략 동시 추출.
        """
        p1_dict, p2_dict, game_value = self.endgame_solver.solve_both(
            state, opp_variants=opp_variants)

        # 탐색 노이즈 적용
        if add_noise and p1_dict:
            actions = list(p1_dict.keys())
            probs = np.array([p1_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p1_dict = {a: float(p) for a, p in zip(actions, probs)}

        if add_noise and p2_dict:
            actions = list(p2_dict.keys())
            probs = np.array([p2_dict[a] for a in actions])
            probs = self._add_exploration_noise(probs)
            p2_dict = {a: float(p) for a, p in zip(actions, probs)}

        # 엔드게임에서는 regret/reversal 빈 dict 반환 (해당 정보 미계산)
        p1_regret = {a: 0.0 for a in p1_dict}
        p2_regret = {a: 0.0 for a in p2_dict}
        p1_reversal = {a: 0.0 for a in p1_dict}
        p2_reversal = {a: 0.0 for a in p2_dict}

        return (p1_dict, p2_dict, float(game_value),
                p1_regret, p2_regret, p1_reversal, p2_reversal)

    def _add_exploration_noise(self, strategy: np.ndarray) -> np.ndarray:
        """Dirichlet 노이즈 + ε-greedy 적용."""
        n = len(strategy)
        if n <= 1:
            return strategy

        # 1. Dirichlet 노이즈
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_weight
        mixed = (1 - eps) * strategy + eps * noise

        # 2. ε-greedy
        eg = self.epsilon_greedy
        uniform = np.ones(n) / n
        result = (1 - eg) * mixed + eg * uniform

        # 정규화
        result = np.maximum(result, 0)
        r_sum = result.sum()
        if r_sum > 0:
            result /= r_sum
        else:
            result = np.ones(n) / n

        return result


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """nash_solver.py 검증 -알려진 게임으로 LP 솔버 테스트."""
    print("=== Nash Solver 검증 ===\n")

    # 1. 가위바위보 (균등 1/3, game_value=0)
    print("1. 가위바위보:")
    M_rps = np.array([[0, -1, 1],
                      [1,  0, -1],
                      [-1, 1,  0]], dtype=np.float64)
    p1, p2, v = solve_zero_sum_lp(M_rps)
    print(f"   P1 전략: {p1}")
    print(f"   P2 전략: {p2}")
    print(f"   게임밸류: {v:.4f}")
    assert abs(v) < 0.01, f"Expected ~0, got {v}"
    assert all(abs(p - 1/3) < 0.05 for p in p1), f"Expected ~1/3 each, got {p1}"
    print("   PASS\n")

    # 2. 매칭 페니 (균등 1/2, game_value=0)
    print("2. 매칭 페니:")
    M_mp = np.array([[ 1, -1],
                     [-1,  1]], dtype=np.float64)
    p1, p2, v = solve_zero_sum_lp(M_mp)
    print(f"   P1 전략: {p1}")
    print(f"   P2 전략: {p2}")
    print(f"   게임밸류: {v:.4f}")
    assert abs(v) < 0.01, f"Expected ~0, got {v}"
    assert all(abs(p - 0.5) < 0.05 for p in p1), f"Expected ~0.5 each, got {p1}"
    print("   PASS\n")

    # 3. 지배전략이 있는 경우
    print("3. 지배전략:")
    M_dom = np.array([[3, 0],
                      [2, 1],
                      [0, 3]], dtype=np.float64)
    p1, p2, v = solve_zero_sum_lp(M_dom)
    print(f"   P1 전략: {p1}")
    print(f"   P2 전략: {p2}")
    print(f"   게임밸류: {v:.4f}")
    print("   (행 3은 지배당하므로 확률 ~0)")
    print("   PASS\n")

    # 4. 단일 액션 엣지 케이스
    print("4. 단일 액션 (1×1):")
    M_1x1 = np.array([[0.5]])
    p1, p2, v = solve_zero_sum_lp(M_1x1)
    assert abs(v - 0.5) < 0.01
    assert p1[0] == 1.0 and p2[0] == 1.0
    print(f"   v={v:.3f} -PASS\n")

    # 5. 1×m 엣지 케이스
    print("5. 단일 행 (1×3):")
    M_1xm = np.array([[0.2, -0.5, 0.8]])
    p1, p2, v = solve_zero_sum_lp(M_1xm)
    print(f"   P1 전략: {p1}, P2 전략: {p2}, v={v:.3f}")
    assert abs(v - (-0.5)) < 0.01  # P2는 최소값 선택
    print("   PASS\n")

    # 6. Fictitious play 폴백 테스트
    print("6. Fictitious play (가위바위보):")
    fp1, fp2, fv = _fictitious_play(M_rps, iterations=200)
    print(f"   P1 전략: {fp1}")
    print(f"   게임밸류: {fv:.4f}")
    assert abs(fv) < 0.1, f"Expected ~0, got {fv}"
    print("   PASS\n")

    print("=== 모든 Nash Solver 테스트 통과! ===")


if __name__ == "__main__":
    verify()
