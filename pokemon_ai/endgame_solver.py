"""엔드게임 Nash 솔버 — 남은 포켓몬이 적을 때 멀티턴 완전탐색.

핵심 아이디어:
  - 남은 포켓몬 합계 <= 3이면 엔드게임 진입
  - 재귀적으로 (내 액션 x 상대 액션) 모든 조합을 탐색
  - 각 리프에서 terminal → +1/-1, 깊이 한계 → value_net 폴백
  - 매 노드에서 Nash 균형 (LP)으로 최적 혼합전략 계산
  - Transposition table로 중복 상태 캐싱

깊이 자동 결정 (depth-aware 캐시 + 전폭 탐색):
  1v1 (2마리) → depth 8   (~6s GPU)
  2v1 (3마리) → depth 4   (~11s GPU)
  2v2 (4마리) → depth 3   (~16s CPU)
  3v+ (미드게임) → depth 2  (NashSolver에서 2턴 탐색)
"""

from __future__ import annotations

import numpy as np
from statistics import mean

from battle_sim import BattleSimulator, BattleState
from data_loader import GameData
from nash_solver import solve_zero_sum_lp, solve_robust_nash_lp


# ═══════════════════════════════════════════════════════════════
#  엔드게임 Nash 솔버
# ═══════════════════════════════════════════════════════════════

class EndgameNashSolver:
    """남은 포켓몬이 적을 때 멀티턴 재귀 Nash 탐색."""

    def __init__(self, sim: BattleSimulator, gd: GameData,
                 evaluator, max_depth: int = 3,
                 max_branch: int = 0):
        """
        Args:
            sim: BattleSimulator 인스턴스
            gd: GameData
            evaluator: NetworkEvaluator (evaluate_value 메서드 필요)
            max_depth: 최대 탐색 깊이 (auto_depth로 조정됨)
            max_branch: 깊이 1+ 에서 양쪽 액션 수 제한 (0=제한없음).
                        루트는 항상 전폭. 1-ply value net으로 상위 K개 선택.
        """
        self.sim = sim
        self.gd = gd
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.max_branch = max_branch
        # depth-aware cache: key → (strategy, value, searched_remaining_depth)
        # 캐시 히트 조건: cached_remaining >= 현재 remaining depth
        # → 얕은 결과가 깊은 탐색을 대체하지 않음
        self._cache: dict[int, tuple[dict[int, float], float, int]] = {}

    def clear_cache(self):
        """transposition table 초기화."""
        self._cache.clear()

    def solve(self, state: BattleState, player: int
              ) -> tuple[dict[int, float], float]:
        """엔드게임 Nash 전략 + game_value 반환.

        Args:
            state: 현재 배틀 상태
            player: 전략을 구할 플레이어 (0 or 1)

        Returns:
            strategy: {action_id: probability} 혼합전략
            game_value: float [-1, +1] player 관점 기댓값
        """
        self._cache.clear()
        depth = self._auto_depth(state)
        return self._recursive_nash(state, player, 0, depth)

    def solve_both(self, state: BattleState,
                   opp_variants: list[tuple[BattleState, float]] | None = None,
                   ) -> tuple[dict[int, float], dict[int, float], float]:
        """양쪽 전략을 한 번에 계산 (고정 깊이)."""
        self._cache.clear()
        max_depth = self._auto_depth(state)
        return self._solve_both_core(state, max_depth,
                                     opp_variants=opp_variants)

    def solve_both_timed(self, state: BattleState, time_budget: float = 40.0,
                         opp_variants: list[tuple[BattleState, float]] | None = None,
                         ) -> tuple[dict[int, float], dict[int, float],
                                    float, int]:
        """Iterative deepening within time budget.

        depth=1부터 시작, 시간이 허용하는 만큼 깊게.
        _auto_depth가 아닌 시간 예산이 깊이를 결정.
        depth-aware 캐시: 얕은 결과를 깊은 탐색이 재활용.
        |gv| > 0.99이면 확정으로 조기 종료.

        Returns:
            p1_dict, p2_dict, game_value, reached_depth
        """
        import time as _time
        t0 = _time.time()

        self._cache.clear()  # 한 번만 클리어
        best_p1: dict[int, float] = {}
        best_p2: dict[int, float] = {}
        best_gv = 0.0
        reached = 0
        prev_time = 0.0

        for depth in range(1, self.max_depth + 1):
            elapsed = _time.time() - t0
            if depth > 1 and elapsed > time_budget * 0.5:
                break

            iter_t0 = _time.time()
            p1, p2, gv = self._solve_both_core(state, depth,
                                                opp_variants=opp_variants)
            iter_time = _time.time() - iter_t0

            best_p1, best_p2, best_gv = p1, p2, gv
            reached = depth

            # 확정 결과 → 조기 종료
            if abs(gv) > 0.99:
                break

            # 다음 반복 시간 추정: 실측 분기비율 사용
            remaining = time_budget - (_time.time() - t0)
            if prev_time > 0.01:
                ratio = iter_time / prev_time
            else:
                ratio = 15.0  # 첫 반복은 보수적 추정
            estimated_next = iter_time * max(ratio, 3.0)
            if estimated_next > remaining:
                break

            prev_time = iter_time

        return best_p1, best_p2, best_gv, reached

    def _solve_both_core(self, state: BattleState, max_depth: int,
                         opp_variants: list[tuple[BattleState, float]] | None = None,
                         ) -> tuple[dict[int, float], dict[int, float], float]:
        """양쪽 전략 계산 핵심 로직. 캐시 클리어 하지 않음.

        루트 레벨에서 P1 관점 보상행렬을 만들고, 내부 재귀는 캐시 있는
        _recursive_nash를 재활용. solve_zero_sum_lp로 양쪽 전략 동시 추출.

        opp_variants가 있으면 (K>1) 루트에서 K개 행렬을 각각 구성하여
        solve_robust_nash_lp()로 모든 변형에 robust한 전략을 구함.
        재귀 탐색은 각 변형별 단일 상태로 진행 (지수 폭발 없음).
        """
        if state.is_terminal:
            v = 1.0 if state.winner == 0 else (-1.0 if state.winner == 1 else 0.0)
            return {}, {}, v

        # ── 변형이 없거나 1개면 기존 로직 ──
        if not opp_variants or len(opp_variants) <= 1:
            return self._solve_both_core_single(state, max_depth)

        # ── K개 변형: Robust Bayesian Nash LP ──
        legal_p1 = self._prune(state, 0, self.sim.get_legal_actions(state, 0))
        n = len(legal_p1)
        if n == 0:
            return {}, {}, 0.0

        matrices = []
        weights = []
        all_legal_p2 = []  # 첫 번째 변형의 legal_p2를 P2 전략용으로 사용
        n_samples = 3 if max_depth > 1 else 1

        for variant_state, weight in opp_variants:
            legal_p2_k = self._prune(variant_state, 1,
                                     self.sim.get_legal_actions(variant_state, 1))
            m_k = len(legal_p2_k)
            if m_k == 0:
                continue

            M_k = np.zeros((n, m_k), dtype=np.float64)
            for i, a1 in enumerate(legal_p1):
                for j, a2 in enumerate(legal_p2_k):
                    values = []
                    for _ in range(n_samples):
                        ns = self.sim.step(variant_state, a1, a2)
                        if ns.is_terminal:
                            values.append(
                                1.0 if ns.winner == 0
                                else (-1.0 if ns.winner == 1 else 0.0))
                        elif max_depth <= 1:
                            values.append(
                                self.evaluator.evaluate_value(ns, 0))
                        else:
                            _, v = self._recursive_nash(
                                ns, 0, 1, max_depth)
                            values.append(v)
                    M_k[i, j] = mean(values)

            matrices.append(M_k)
            weights.append(weight)
            all_legal_p2.append(legal_p2_k)

        if not matrices:
            return self._solve_both_core_single(state, max_depth)

        # Robust Nash LP로 P1 전략 계산
        p1_strat, game_value = solve_robust_nash_lp(matrices, weights)
        p1_dict = {a: float(p) for a, p in zip(legal_p1, p1_strat)}

        # P2 전략: 가중 평균 행렬의 첫 번째 변형 기준으로 근사
        # (P2의 정확한 전략은 각 타입별로 다르지만, 로그용으로 1차 근사)
        primary_p2 = all_legal_p2[0]
        _, p2_strat, _ = solve_zero_sum_lp(matrices[0])
        p2_dict = {a: float(p) for a, p in zip(primary_p2, p2_strat)}

        return p1_dict, p2_dict, float(game_value)

    def _prune(self, state: BattleState, player: int,
               actions: list[int]) -> list[int]:
        """evaluator에 prune_obvious가 있으면 호출."""
        if hasattr(self.evaluator, 'prune_obvious'):
            return self.evaluator.prune_obvious(state, player, actions)
        return actions

    def _solve_both_core_single(self, state: BattleState, max_depth: int
                                ) -> tuple[dict[int, float], dict[int, float], float]:
        """단일 상태 (변형 없음) 양쪽 전략 계산. 기존 로직."""
        if state.is_terminal:
            v = 1.0 if state.winner == 0 else (-1.0 if state.winner == 1 else 0.0)
            return {}, {}, v

        legal_p1 = self._prune(state, 0, self.sim.get_legal_actions(state, 0))
        legal_p2 = self._prune(state, 1, self.sim.get_legal_actions(state, 1))
        n, m = len(legal_p1), len(legal_p2)
        if n == 0 or m == 0:
            return {}, {}, 0.0

        # 루트: P1 관점 보상행렬 구성
        M = np.zeros((n, m), dtype=np.float64)
        n_samples = 3 if max_depth > 1 else 1

        for i, a1 in enumerate(legal_p1):
            for j, a2 in enumerate(legal_p2):
                values = []
                for _ in range(n_samples):
                    ns = self.sim.step(state, a1, a2)
                    if ns.is_terminal:
                        values.append(1.0 if ns.winner == 0
                                      else (-1.0 if ns.winner == 1 else 0.0))
                    elif max_depth <= 1:
                        values.append(self.evaluator.evaluate_value(ns, 0))
                    else:
                        _, v = self._recursive_nash(
                            ns, 0, 1, max_depth)
                        values.append(v)
                M[i, j] = mean(values)

        p1_strat, p2_strat, game_value = solve_zero_sum_lp(M)
        p1_dict = {a: float(p) for a, p in zip(legal_p1, p1_strat)}
        p2_dict = {a: float(p) for a, p in zip(legal_p2, p2_strat)}
        return p1_dict, p2_dict, float(game_value)

    def _recursive_nash(self, state: BattleState, player: int,
                        depth: int, max_depth: int
                        ) -> tuple[dict[int, float], float]:
        """재귀적 Nash 탐색.

        Algorithm:
        1. terminal → 정확한 승패 (+1/-1)
        2. transposition table 히트 → 캐시된 값
        3. legal_me × legal_opp 모든 조합에 대해 next state 평가
        4. 보상행렬 M[i][j] 구성
        5. solve_zero_sum_lp(M) → Nash 전략 + game_value
        6. transposition table에 캐시
        """
        # 1. Terminal 체크
        if state.is_terminal:
            if state.winner == player:
                return {}, 1.0
            elif state.winner == (1 - player):
                return {}, -1.0
            else:
                return {}, 0.0

        # 2. Transposition table (depth-aware)
        key = self._hash_state(state, player)
        remaining = max_depth - depth
        if key in self._cache:
            cached_strat, cached_val, cached_remaining = self._cache[key]
            if cached_remaining >= remaining:
                return cached_strat, cached_val

        # 3. Legal actions (pruning 적용)
        opp = 1 - player
        legal_me = self._prune(state, player,
                               self.sim.get_legal_actions(state, player))
        legal_opp = self._prune(state, opp,
                                self.sim.get_legal_actions(state, opp))

        # 가지치기: 깊이 1+ 에서 상위 K개만 탐색
        if self.max_branch > 0 and depth > 0:
            legal_me = self._prune_actions(
                state, player, legal_me, legal_opp)
            legal_opp = self._prune_actions(
                state, opp, legal_opp, legal_me)

        n = len(legal_me)
        m = len(legal_opp)

        if n == 0 or m == 0:
            return {}, 0.0

        # 4. 보상행렬 구성
        M = np.zeros((n, m), dtype=np.float64)
        is_leaf_depth = (depth + 1 >= max_depth)

        # 리프 깊이: sim.step 후 배치 평가 (개별 evaluate_value 대신)
        if is_leaf_depth:
            leaf_states: list[BattleState] = []
            leaf_positions: list[tuple[int, int, int]] = []  # (i, j, sample_idx)

            for i, a_me in enumerate(legal_me):
                for j, a_opp in enumerate(legal_opp):
                    if player == 0:
                        ns = self.sim.step(state, a_me, a_opp)
                    else:
                        ns = self.sim.step(state, a_opp, a_me)
                    if ns.is_terminal:
                        if ns.winner == player:
                            M[i, j] = 1.0
                        elif ns.winner == opp:
                            M[i, j] = -1.0
                    else:
                        leaf_states.append(ns)
                        leaf_positions.append((i, j, 0))

            # 배치 평가
            if leaf_states:
                batch_vals = self.evaluator.evaluate_values_batch(
                    leaf_states, player)
                for idx, (i, j, _) in enumerate(leaf_positions):
                    M[i, j] = float(batch_vals[idx])
        else:
            # 비-리프: 재귀 탐색
            n_samples = 1

            for i, a_me in enumerate(legal_me):
                for j, a_opp in enumerate(legal_opp):
                    values = []
                    for _ in range(n_samples):
                        if player == 0:
                            ns = self.sim.step(state, a_me, a_opp)
                        else:
                            ns = self.sim.step(state, a_opp, a_me)

                        if ns.is_terminal:
                            if ns.winner == player:
                                values.append(1.0)
                            elif ns.winner == opp:
                                values.append(-1.0)
                            else:
                                values.append(0.0)
                        else:
                            _, v = self._recursive_nash(ns, player, depth + 1,
                                                        max_depth)
                            values.append(v)
                    M[i, j] = mean(values)

        # 5. Nash 균형
        p1_strat, p2_strat, game_value = solve_zero_sum_lp(M)

        # player 관점 전략 (P1 = 행 = 내 전략)
        strategy = {a: float(p) for a, p in zip(legal_me, p1_strat)}

        # 6. 캐시 (탐색 깊이 포함)
        self._cache[key] = (strategy, float(game_value), remaining)
        return strategy, float(game_value)

    def _prune_actions(self, state: BattleState, player: int,
                       actions: list[int], opp_actions: list[int]
                       ) -> list[int]:
        """1-ply value net 평가로 상위 max_branch개 액션만 남기기.

        각 액션을 모든 상대 액션에 대해 시뮬레이션 후 배치 평가.
        평균값 기준 상위 K개 선택 (루트는 호출하지 않으므로 depth>0 전용).
        """
        if len(actions) <= self.max_branch:
            return actions

        opp = 1 - player
        n_me = len(actions)
        n_opp = len(opp_actions)

        # 모든 (action, opp_action) 조합 시뮬레이션
        all_ns: list[BattleState | None] = [None] * (n_me * n_opp)
        values = np.zeros(n_me * n_opp, dtype=np.float64)
        non_term: list[tuple[int, BattleState]] = []

        for i, a_me in enumerate(actions):
            for j, a_opp in enumerate(opp_actions):
                if player == 0:
                    ns = self.sim.step(state, a_me, a_opp)
                else:
                    ns = self.sim.step(state, a_opp, a_me)
                idx = i * n_opp + j
                if ns.is_terminal:
                    if ns.winner == player:
                        values[idx] = 1.0
                    elif ns.winner == opp:
                        values[idx] = -1.0
                else:
                    non_term.append((idx, ns))

        # 배치 평가
        if non_term:
            batch_states = [ns for _, ns in non_term]
            batch_vals = self.evaluator.evaluate_values_batch(
                batch_states, player)
            for (idx, _), v in zip(non_term, batch_vals):
                values[idx] = float(v)

        # 각 action의 상대 전액션 평균값으로 스코어링
        action_scores = []
        for i in range(n_me):
            avg = values[i * n_opp: (i + 1) * n_opp].mean()
            action_scores.append(avg)

        # 상위 K개 선택 (원래 순서 유지)
        indexed = sorted(enumerate(action_scores), key=lambda x: -x[1])
        top_indices = sorted(idx for idx, _ in indexed[:self.max_branch])
        return [actions[i] for i in top_indices]

    def _auto_depth(self, state: BattleState) -> int:
        """남은 포켓몬 수에 따라 탐색 깊이 자동 결정.

        depth-aware 캐시 + 전폭 탐색 (~30초 예산):
        | 남은 포켓몬 | 깊이 | 근거               |
        |-------------|------|--------------------|
        | 1v1 (2마리) | 8    | 6s (CPU)           |
        | 2v1 (3마리) | 4    | 32s CPU / ~20s GPU |
        | 2v2 (4마리) | 3    | 16s (CPU)          |
        | 3v+ (5+)    | 2    | 미드게임 기본      |
        """
        alive_0 = state.sides[0].alive_count
        alive_1 = state.sides[1].alive_count
        total = alive_0 + alive_1

        if total <= 2:
            d = 8
        elif total <= 3:
            d = 4
        elif total <= 4:
            d = 3
        else:
            d = 2
        return min(d, self.max_depth)

    def _estimate_best_damage(self, attacker, defender) -> int:
        """상대의 최강기 max 데미지 근사 (해싱용, 빠른 계산).

        STAB + 타입상성 + 부스트만 반영. 특성/아이템은 생략.
        """
        best = 0
        for move_id in attacker.moves:
            if not move_id:
                continue
            move = self.gd.get_move(move_id)
            if not move or move.get('is_status'):
                continue
            power = move.get('basePower', 0)
            if power <= 0:
                continue

            is_phys = move.get('is_physical', False)
            atk_key = 'atk' if is_phys else 'spa'
            def_key = 'def' if is_phys else 'spd'

            a = attacker.stats.get(atk_key, 100)
            d = defender.stats.get(def_key, 100)

            # 부스트 반영
            ab = attacker.boosts.get(atk_key, 0)
            db = defender.boosts.get(def_key, 0)
            if ab > 0:
                a = a * (2 + ab) // 2
            elif ab < 0:
                a = a * 2 // (2 - ab)
            if db > 0:
                d = d * (2 + db) // 2
            elif db < 0:
                d = d * 2 // (2 - db)

            # Gen 9 BSS (level 50): (2*50/5+2) = 22
            dmg = (22 * power * a // max(d, 1)) // 50 + 2

            # STAB
            move_type = move.get('type', '')
            if move_type in attacker.types:
                dmg = dmg * 3 // 2

            # 타입 상성
            eff = self.gd.effectiveness(move_type, defender.types)
            dmg = int(dmg * eff)

            if dmg > best:
                best = dmg
        return best

    def _hp_bucket(self, poke, opp_active,
                   hazard_dmg: int = 0) -> int:
        """HP → 확정/난수 N타 버킷 (상대 최강기 + 도트 + 장판 반영).

        0: 기절
        1: 확정1타 (HP ≤ min롤 데미지)
        2: 난수1타 (min롤 < HP ≤ max롤)
        3: 확정2타   4: 난수2타
        5: 확정3타   6: 난수3타
        7: 4타 이상 or 상대 유효타 없음

        hazard_dmg: 벤치 포켓몬이 교체 등장 시 받는 장판 데미지
        """
        if poke.fainted or poke.cur_hp <= 0:
            return 0

        max_dmg = self._estimate_best_damage(opp_active, poke)
        if max_dmg <= 0:
            return 7

        min_dmg = max(int(max_dmg * 0.85), 1)

        # 턴당 도트 데미지 (화상/독)
        chip = 0
        mhp = max(poke.max_hp, 1)
        if poke.status == 'brn':
            chip = mhp // 16
        elif poke.status == 'psn':
            chip = mhp // 8
        elif poke.status == 'tox':
            chip = mhp // 16

        # 장판 데미지는 교체 시 1회 → 실질 HP 깎기
        effective_hp = max(poke.cur_hp - hazard_dmg, 0)
        if effective_hp <= 0:
            return 0

        eff_min = min_dmg + chip
        eff_max = max_dmg + chip

        for n in range(1, 4):
            if effective_hp <= n * eff_min:
                return 2 * n - 1   # 확정N타
            if effective_hp <= n * eff_max:
                return 2 * n       # 난수N타
        return 7

    def _calc_hazard_damage(self, poke, side) -> int:
        """교체 등장 시 받는 장판 데미지 합계."""
        dmg = 0
        mhp = max(poke.max_hp, 1)

        # 스텔스록: max_hp * Rock상성 / 8
        if side.stealth_rock:
            eff = self.gd.effectiveness("Rock", poke.types)
            dmg += int(mhp * eff / 8)

        # 압정: 비행 면역, 층수별 1/8 → 1/6 → 1/4
        if side.spikes > 0 and "Flying" not in poke.types:
            spike_frac = [0, 1/8, 1/6, 1/4][min(side.spikes, 3)]
            dmg += int(mhp * spike_frac)

        return dmg

    def _hash_state(self, state: BattleState, player: int) -> int:
        """배틀 상태의 해시값 계산 (transposition table 키).

        HP를 확정/난수 N타 버킷으로 양자화 → 캐시 히트율 향상.
        데미지 롤 85~100%로 갈리는 확정/난수 경계를 반영.
        """
        parts: list = [player]
        for side_idx in range(2):
            side = state.sides[side_idx]
            opp_active = state.sides[1 - side_idx].active
            parts.append(side.active_idx)
            parts.append(side.tera_used)
            parts.append(side.stealth_rock)
            parts.append(side.spikes)
            parts.append(side.toxic_spikes)
            for idx, poke in enumerate(side.team):
                if opp_active.fainted:
                    # 상대 기절 → 간단한 HP% 폴백 (4버킷)
                    hp_b = (0 if poke.fainted
                            else min(int(poke.cur_hp * 4
                                         / max(poke.max_hp, 1)) + 1, 4))
                else:
                    # 벤치 포켓몬은 교체 시 장판 데미지 선반영
                    h_dmg = (self._calc_hazard_damage(poke, side)
                             if idx != side.active_idx else 0)
                    hp_b = self._hp_bucket(poke, opp_active, h_dmg)
                parts.append(hp_b)
                parts.append(poke.fainted)
                parts.append(poke.status)
                parts.append(poke.is_tera)
                # 부스트
                for stat in ("atk", "def", "spa", "spd", "spe"):
                    parts.append(poke.boosts.get(stat, 0))
        parts.append(int(state.weather))
        parts.append(state.weather_turns)
        parts.append(int(state.terrain))
        parts.append(state.terrain_turns)
        parts.append(state.trick_room_turns)
        return hash(tuple(parts))


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """엔드게임 솔버 단독 테스트."""
    print("=== Endgame Solver 검증 ===\n")

    import os, sys, torch
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import GameData
    from battle_sim import BattleSimulator, make_pokemon
    from neural_net import PokemonNet, NetworkEvaluator

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)

    # 모델 로드 (선택적)
    ckpt_path = os.path.join(os.path.dirname(__file__),
                             "checkpoints", "nash_best_iter4.pt")
    model = PokemonNet()
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Model loaded: {ckpt_path}")
    model.eval()
    evaluator = NetworkEvaluator(model, gd, device="cpu")

    solver = EndgameNashSolver(sim, gd, evaluator, max_depth=3)

    # 1v1 테스트: 코라이돈 vs 딩루
    print("\n--- 1v1 테스트: 코라이돈 vs 딩루 ---")
    koraidon = make_pokemon(gd, "Koraidon",
        moves=["Close Combat", "Flare Blitz", "Dragon Claw", "Swords Dance"],
        ability="Orichalcum Pulse", item="Life Orb",
        nature="Adamant",
        evs={"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
        tera_type="Fighting")

    tinglu = make_pokemon(gd, "Ting-Lu",
        moves=["Earthquake", "Ruination", "Whirlwind", "Stealth Rock"],
        ability="Vessel of Ruin", item="Leftovers",
        nature="Impish",
        evs={"hp": 244, "atk": 4, "def": 116, "spa": 0, "spd": 132, "spe": 12},
        tera_type="Poison")

    state = sim.create_battle_state([koraidon], [tinglu])
    print(f"코라이돈 HP: {koraidon.cur_hp}, 딩루 HP: {tinglu.cur_hp}")
    print(f"auto_depth: {solver._auto_depth(state)}")

    strategy, gv = solver.solve(state, 0)
    print(f"P1 전략: {strategy}")
    print(f"game_value: {gv:+.4f}")
    print(f"캐시 크기: {len(solver._cache)}")

    # 1v1 테스트: 딩루 관점
    print("\n--- 1v1 테스트: 딩루 관점 ---")
    strategy2, gv2 = solver.solve(state, 1)
    print(f"P2 전략: {strategy2}")
    print(f"game_value (P2): {gv2:+.4f}")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
