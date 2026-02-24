"""Stockfish 방식 룰 기반 평가함수 — 신경망 없이 포켓몬 룰을 직접 하드코딩.

핵심 아이디어: 포켓몬은 바둑이 아니라 체스다.
- 룰이 명시적, 데이터베이스 완비 → 룰 기반 평가 + 탐색이 정답
- DamageCalculator를 직접 호출하여 정확한 데미지 계산
- 10개 컴포넌트의 가중합 → tanh → [-1, +1]

NetworkEvaluator와 동일 인터페이스 → drop-in 교체 가능.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

from data_loader import GameData, ABILITY_EFFECTS, ITEM_EFFECTS, _TYPE_BOOST_ITEMS
from damage_calc import DamageCalculator
from battle_sim import (
    BattleState, Side, Pokemon, Weather, Terrain,
    WEATHER_STR, TERRAIN_STR, NUM_ACTIONS,
)


class RuleBasedEvaluator:
    """룰 기반 포지션 평가 — Stockfish 스타일."""

    # ── 기본 컴포넌트 가중치 ──
    # 순서: material, hp_econ, matchup, sweep, counter, hazard,
    #       status, boosts, field, tera, breakthrough, def_core
    _DEFAULT_W = (1.2, 0.6, 0.5, 1.5, 0.8, 0.3, 0.4, 0.6, 0.3, 0.6, 1.0, 0.7)

    # ── 매크로 × 서브골별 가중치 ──
    # setup_sweep 단계: support → land → setup → sweep
    _SS_SUPPORT = (1.2, 0.4, 0.7, 1.0, 1.2, 0.4, 0.4, 0.3, 0.3, 0.8, 1.0, 0.5)
    _SS_LAND    = (1.2, 0.5, 0.8, 1.5, 1.0, 0.3, 0.4, 0.3, 0.3, 0.8, 0.8, 0.5)
    _SS_SETUP   = (1.2, 0.3, 0.5, 2.0, 0.8, 0.2, 0.4, 1.5, 0.3, 0.6, 0.5, 0.3)
    _SS_SWEEP   = (1.2, 0.3, 0.8, 2.5, 0.5, 0.2, 0.4, 1.2, 0.3, 0.4, 0.5, 0.3)
    # break_clean 단계: break → transition → clean
    _BC_BREAK   = (1.2, 0.5, 0.8, 1.0, 0.8, 0.3, 0.4, 0.4, 0.3, 0.6, 1.8, 0.7)
    _BC_TRANS   = (1.2, 0.6, 0.6, 1.2, 0.8, 0.3, 0.4, 0.4, 0.3, 0.6, 1.2, 0.7)
    _BC_CLEAN   = (1.2, 0.4, 0.9, 1.8, 0.5, 0.2, 0.4, 0.4, 0.3, 0.5, 0.8, 0.5)
    # cycle 단계: hazards → cycle → close
    _CY_HAZARDS = (1.2, 0.8, 0.5, 0.6, 1.0, 0.8, 0.4, 0.3, 0.3, 0.6, 0.5, 1.0)
    _CY_CYCLE   = (1.2, 1.2, 0.4, 0.6, 1.0, 0.5, 0.5, 0.2, 0.3, 0.6, 0.5, 1.2)
    _CY_CLOSE   = (1.2, 0.6, 0.7, 1.2, 0.8, 0.3, 0.4, 0.4, 0.3, 0.6, 1.0, 0.7)

    def __init__(self, game_data: GameData):
        self.gd = game_data
        self.dc = DamageCalculator(game_data)
        self.macro_type: str | None = None
        self.macro_info: dict | None = None

    def set_macro(self, macro_type: str | None, macro_info: dict | None = None):
        """매크로 전략 설정. None이면 기본 가중치."""
        self.macro_type = macro_type
        self.macro_info = macro_info

    # ═══════════════════════════════════════════════════════════════
    #  공개 인터페이스 (NetworkEvaluator 호환)
    # ═══════════════════════════════════════════════════════════════

    def evaluate_value(self, state: BattleState, player: int) -> float:
        """[-1, +1] 룰 기반 포지션 평가."""
        if state.is_terminal:
            if state.winner == player:
                return 1.0
            elif state.winner == (1 - player):
                return -1.0
            else:
                return 0.0
        return self._compute_value(state, player)

    def evaluate_values_batch(self, states: list[BattleState],
                              player: int) -> np.ndarray:
        """배치 평가 (루프, GPU 불필요)."""
        if not states:
            return np.array([], dtype=np.float32)
        return np.array([self.evaluate_value(s, player) for s in states],
                        dtype=np.float32)

    def evaluate(self, state: BattleState, player: int,
                 legal_actions: list[int]) -> tuple[dict[int, float], float]:
        """MCTS 호환 — 균등 prior + value."""
        value = self.evaluate_value(state, player)
        n = len(legal_actions)
        if n == 0:
            return {}, value
        uniform = 1.0 / n
        priors = {a: uniform for a in legal_actions}
        return priors, value

    # ═══════════════════════════════════════════════════════════════
    #  액션 프루닝 — 명백히 나쁜 행동 제거
    # ═══════════════════════════════════════════════════════════════

    def prune_obvious(self, state: BattleState, player: int,
                      legal_actions: list[int]) -> list[int]:
        """룰 상 무효인 행동만 제거. 최소 1개는 보장.

        상대 교체를 고려하면 "현재 대면에서 나빠 보이는 수"도
        교체 읽기로는 최적일 수 있음. Nash 균형을 해치지 않도록
        게임이 효과를 막는 경우만 제거.

        제거 대상:
          1. 풀HP에서 회복기 (회복량 0)
          2. 이미 설치된 헤저드 재설치 (게임이 실패 처리)
          3. 이미 +6인 스탯에 부스트 (효과 없음)
          4. 연속 방어 (protect_count >= 1이면 성공률 1/3 이하)
        """
        if len(legal_actions) <= 1:
            return legal_actions

        side = state.sides[player]
        opp = state.sides[1 - player]
        active = side.active

        if active.fainted:
            return legal_actions

        bad = set()

        # 4. 연속 방어 프루닝 — protect_count >= 1이면 성공률 33% 이하
        _PROTECT_MOVES = {"protect", "detect", "banefulbunker",
                          "spikyshield", "kingsshield", "obstruct",
                          "silktrap", "burningbulwark"}
        if active.protect_count >= 1:
            for action in legal_actions:
                if 4 <= action <= 8:
                    continue
                is_tera = action >= 9
                move_idx = action - 9 if is_tera else action
                if move_idx < len(active.moves):
                    mid = active.moves[move_idx]
                    if mid in _PROTECT_MOVES:
                        bad.add(action)

        for action in legal_actions:
            if action in bad:
                continue
            # 교체(4~8)는 절대 제거 안 함
            if 4 <= action <= 8:
                continue

            is_tera = action >= 9
            move_idx = action - 9 if is_tera else action
            if move_idx >= len(active.moves):
                continue

            move_id = active.moves[move_idx]
            move = self.gd.get_move(move_id)
            if not move:
                continue

            is_status = move.get("is_status", False)

            # 1. 풀HP에서 회복기 (회복량 0)
            if move.get("flags", {}).get("heal"):
                if active.cur_hp >= active.max_hp:
                    bad.add(action)
                    continue

            # 2. 이미 설치된 헤저드 재설치 (게임이 실패 처리)
            sc = move.get("sideCondition", "")
            if sc == "stealthrock" and opp.stealth_rock:
                bad.add(action)
                continue
            if sc == "spikes" and opp.spikes >= 3:
                bad.add(action)
                continue
            if sc == "toxicspikes" and opp.toxic_spikes >= 2:
                bad.add(action)
                continue
            if sc == "stickyweb" and opp.sticky_web:
                bad.add(action)
                continue

            # 3. 이미 +6인 스탯에 부스트 (효과 없음)
            boosts = move.get("boosts")
            if is_status and boosts:
                positive_boosts = {k: v for k, v in boosts.items()
                                   if v > 0}
                if positive_boosts:
                    all_maxed = all(
                        active.boosts.get(k, 0) >= 6
                        for k in positive_boosts
                    )
                    if all_maxed:
                        bad.add(action)
                        continue

        remaining = [a for a in legal_actions if a not in bad]
        return remaining if remaining else legal_actions

    # ═══════════════════════════════════════════════════════════════
    #  핵심 평가 엔진
    # ═══════════════════════════════════════════════════════════════

    def _compute_value(self, state: BattleState, player: int) -> float:
        """12개 컴포넌트 가중합 → 매크로 서브골 보너스 → tanh."""
        my = state.sides[player]
        opp = state.sides[1 - player]
        w_str = WEATHER_STR.get(state.weather, "")
        t_str = TERRAIN_STR.get(state.terrain, "")
        trick_room = state.trick_room_turns > 0

        # 매크로 서브골 감지 → 가중치 선택
        phase = self._detect_macro_phase(my, opp)
        w = self._get_phase_weights(phase)

        components = [
            self._score_material(my, opp),
            self._score_hp_economy(my, opp),
            self._score_active_matchup(my, opp, w_str, t_str, trick_room),
            self._score_sweep_threat(my, opp, w_str, t_str, trick_room),
            self._score_counter_availability(my, opp, w_str, t_str),
            self._score_hazards(my, opp),
            self._score_status(my, opp),
            self._score_boosts(my, opp),
            self._score_field(my, opp, w_str, t_str),
            self._score_tera(my, opp),
            self._score_breakthrough(my, opp, w_str, t_str),
            self._score_defensive_core(my, opp, w_str, t_str),
        ]

        score = sum(wi * ci for wi, ci in zip(w, components))

        # ── 매크로 서브골 보너스 ──
        score += self._macro_subgoal_bonus(my, opp, phase)

        # 0.95 스케일: 극단에서도 해상도 보존 (±1.0은 진짜 종료에만)
        return math.tanh(score) * 0.95

    # ─── 매크로 서브골 시스템 ─────────────────────────────────────

    def _detect_macro_phase(self, my: Side, opp: Side) -> str:
        """현재 게임 상태에서 매크로 서브골 단계를 판단.

        Returns:
            "ss_support" / "ss_land" / "ss_setup" / "ss_sweep"
            "bc_break" / "bc_trans" / "bc_clean"
            "cy_hazards" / "cy_cycle" / "cy_close"
            "default" (매크로 없음)
        """
        if not self.macro_type or not self.macro_info:
            return "default"

        my_alive = {p.name: p for p in my.team if not p.fainted}
        opp_alive_cnt = opp.alive_count

        if self.macro_type == "setup_sweep":
            ace_name = self.macro_info.get("ace", "")
            ace = my_alive.get(ace_name)

            if not ace:
                return "default"  # 에이스 죽음 → 플랜 붕괴, 기본 모드

            ace_is_active = (my.active.name == ace_name and
                             not my.active.fainted)
            ace_has_boost = False
            if ace_is_active:
                atk_b = my.active.boosts.get("atk", 0)
                spa_b = my.active.boosts.get("spa", 0)
                ace_has_boost = max(atk_b, spa_b) >= 1

            if ace_is_active and ace_has_boost:
                return "ss_sweep"   # 부스트 완료 → 스윕
            if ace_is_active:
                return "ss_setup"   # 에이스 나와있음 → 세팅
            # 에이스가 벤치에 있음
            # 상대 위협이 줄었으면 착지, 아니면 서포트
            if opp_alive_cnt <= 2:
                return "ss_land"
            return "ss_support"

        elif self.macro_type == "break_clean":
            breaker = self.macro_info.get("breaker", "")
            cleaner = self.macro_info.get("cleaner", "")
            brk_alive = breaker in my_alive
            cln_alive = cleaner in my_alive

            if not cln_alive:
                return "default"  # 클리너 죽음 → 플랜 붕괴

            if opp_alive_cnt <= 1:
                return "bc_clean"   # 상대 1마리 → 마무리
            if not brk_alive:
                return "bc_clean"   # 브레이커 소모됨 → 클린 전환
            # 상대 HP 총합으로 판단
            opp_avg_hp = sum(p.hp_pct for p in opp.team
                             if not p.fainted) / max(1, opp_alive_cnt)
            if opp_avg_hp < 0.5:
                return "bc_trans"   # 상대 깎임 → 전환 단계
            return "bc_break"       # 아직 뚫어야 함

        elif self.macro_type == "cycle":
            a1 = self.macro_info.get("anchor1", "")
            a2 = self.macro_info.get("anchor2", "")
            alive_anchors = sum(1 for a in (a1, a2) if a in my_alive)

            if alive_anchors == 0:
                return "default"

            # 헤저드 설치 여부
            has_hazards = opp.stealth_rock or opp.spikes > 0
            if not has_hazards and opp.alive_count >= 3:
                return "cy_hazards"

            opp_avg_hp = sum(p.hp_pct for p in opp.team
                             if not p.fainted) / max(1, opp.alive_count)
            if opp_avg_hp < 0.4 or opp.alive_count <= 1:
                return "cy_close"
            return "cy_cycle"

        return "default"

    def _get_phase_weights(self, phase: str) -> tuple:
        """서브골 단계에 맞는 가중치 반환."""
        return {
            "ss_support": self._SS_SUPPORT,
            "ss_land":    self._SS_LAND,
            "ss_setup":   self._SS_SETUP,
            "ss_sweep":   self._SS_SWEEP,
            "bc_break":   self._BC_BREAK,
            "bc_trans":   self._BC_TRANS,
            "bc_clean":   self._BC_CLEAN,
            "cy_hazards": self._CY_HAZARDS,
            "cy_cycle":   self._CY_CYCLE,
            "cy_close":   self._CY_CLOSE,
        }.get(phase, self._DEFAULT_W)

    def _macro_subgoal_bonus(self, my: Side, opp: Side,
                              phase: str) -> float:
        """서브골 단계별 구체적 보너스/페널티."""
        if phase == "default" or not self.macro_info:
            return 0.0

        bonus = 0.0
        my_alive = {p.name: p for p in my.team if not p.fainted}

        if phase.startswith("ss_"):
            ace_name = self.macro_info.get("ace", "")
            ace = my_alive.get(ace_name)

            if not ace:
                return -0.8  # 에이스 죽으면 큰 감점

            if phase == "ss_support":
                # 에이스 온존 + 테라 보존 중시
                bonus += 0.3  # 에이스 생존
                if not my.tera_used:
                    bonus += 0.3  # 테라 아직 안 씀 → 좋음
                # 에이스 HP 높을수록 좋음
                bonus += 0.2 * ace.hp_pct

            elif phase == "ss_land":
                # 에이스 착지 준비
                bonus += 0.3
                if not my.tera_used:
                    bonus += 0.2

            elif phase == "ss_setup":
                # 에이스 세팅 중 — HP, 테라, 부스트 중시
                atk_b = my.active.boosts.get("atk", 0)
                spa_b = my.active.boosts.get("spa", 0)
                bonus += 0.4 * max(atk_b, spa_b)  # 부스트 단계당 +0.4
                bonus += 0.3 * my.active.hp_pct    # HP 보존
                if not my.tera_used:
                    bonus += 0.2

            elif phase == "ss_sweep":
                # 스윕 중 — 공격 위력 + 선공 가치
                atk_b = my.active.boosts.get("atk", 0)
                spa_b = my.active.boosts.get("spa", 0)
                bonus += 0.3 * max(atk_b, spa_b)
                bonus += 0.2 * my.active.hp_pct

        elif phase.startswith("bc_"):
            breaker = self.macro_info.get("breaker", "")
            cleaner = self.macro_info.get("cleaner", "")
            cln = my_alive.get(cleaner)

            if phase == "bc_break":
                # 브레이커 공격력 발휘 + 클리너 온존
                if cln:
                    bonus += 0.2 * cln.hp_pct  # 클리너 HP 보존
                if not my.tera_used:
                    bonus += 0.15

            elif phase == "bc_trans":
                # 전환 — 클리너 준비
                if cln:
                    bonus += 0.3 * cln.hp_pct

            elif phase == "bc_clean":
                # 클린 — 클리너가 나와서 마무리
                if cln and my.active.name == cleaner:
                    bonus += 0.3  # 클리너가 액티브

        elif phase.startswith("cy_"):
            a1 = self.macro_info.get("anchor1", "")
            a2 = self.macro_info.get("anchor2", "")
            alive_anchors = sum(1 for a in (a1, a2) if a in my_alive)

            if phase == "cy_hazards":
                bonus += 0.2 * alive_anchors
            elif phase == "cy_cycle":
                # 앵커 HP 합산
                for name in (a1, a2):
                    p = my_alive.get(name)
                    if p:
                        bonus += 0.15 * p.hp_pct
            elif phase == "cy_close":
                bonus += 0.1 * alive_anchors

        return bonus

    # ─── 1. 머테리얼 ────────────────────────────────────────────

    def _score_material(self, my: Side, opp: Side) -> float:
        """생존 포켓몬 수 차이. 범위 [-3, +3]."""
        return float(my.alive_count - opp.alive_count)

    # ─── 2. HP 경제 ─────────────────────────────────────────────

    def _score_hp_economy(self, my: Side, opp: Side) -> float:
        """양측 평균 HP% 차이 × 2. 범위 [-2, +2]."""
        my_hp = self._avg_hp_pct(my)
        opp_hp = self._avg_hp_pct(opp)
        return (my_hp - opp_hp) * 2.0

    @staticmethod
    def _avg_hp_pct(side: Side) -> float:
        alive = [p for p in side.team if not p.fainted]
        if not alive:
            return 0.0
        return sum(p.hp_pct for p in alive) / len(alive)

    # ─── 3. 액티브 매치업 ───────────────────────────────────────

    def _score_active_matchup(self, my: Side, opp: Side,
                              weather: str, terrain: str,
                              trick_room: bool) -> float:
        """현재 1v1 누가 이기나. 범위 [-1.5, +1.5]."""
        my_poke = my.active
        opp_poke = opp.active

        if my_poke.fainted or opp_poke.fainted:
            return 0.0

        my_best_dmg = self._best_move_damage(my_poke, opp_poke, weather, terrain)
        opp_best_dmg = self._best_move_damage(opp_poke, my_poke, weather, terrain)

        my_ohko = my_best_dmg >= opp_poke.cur_hp
        opp_ohko = opp_best_dmg >= my_poke.cur_hp

        # 스피드 비교 (트릭룸 고려)
        my_spe = my_poke.effective_speed(weather, terrain)
        opp_spe = opp_poke.effective_speed(weather, terrain)
        if trick_room:
            i_am_faster = my_spe < opp_spe
            speed_tie = my_spe == opp_spe
        else:
            i_am_faster = my_spe > opp_spe
            speed_tie = my_spe == opp_spe

        # 우선도 체크 — 우선도가 높은 기술이 있으면 선공 판정 오버라이드
        my_prio = self._best_priority(my_poke, opp_poke, weather, terrain)
        opp_prio = self._best_priority(opp_poke, my_poke, weather, terrain)
        if my_prio > opp_prio:
            i_am_faster = True
            speed_tie = False
        elif opp_prio > my_prio:
            i_am_faster = False
            speed_tie = False

        # 동속 + 양측 확1 → 코인 플립, 0점
        if speed_tie and my_ohko and opp_ohko:
            return 0.0

        # 선공 + 확1 = 결정적 우위
        if i_am_faster and my_ohko:
            return 1.5
        if not i_am_faster and opp_ohko:
            return -1.5

        # 동속 + 한쪽만 확1 → 50% 우위
        if speed_tie:
            if my_ohko and not opp_ohko:
                return 0.75
            if opp_ohko and not my_ohko:
                return -0.75

        # 양측 확1 — 스피드가 결정적
        if my_ohko and opp_ohko:
            return 0.8 if i_am_faster else -0.8

        # 교환 시뮬레이션: 잔여 HP 기반
        if opp_poke.cur_hp > 0 and my_poke.cur_hp > 0:
            my_frac = my_best_dmg / opp_poke.cur_hp  # 내가 주는 비율
            opp_frac = opp_best_dmg / my_poke.cur_hp  # 상대가 주는 비율
            advantage = (my_frac - opp_frac)
            return max(-1.5, min(1.5, advantage))

        return 0.0

    # ─── 4. 스윕 위협 ──────────────────────────────────────────

    def _score_sweep_threat(self, my: Side, opp: Side,
                            weather: str, terrain: str,
                            trick_room: bool) -> float:
        """한 마리가 상대 전체를 쓸 수 있는가. 범위 [-1, +1]."""
        my_best = self._calc_sweep_ratio(my, opp, weather, terrain, trick_room)
        opp_best = self._calc_sweep_ratio(opp, my, weather, terrain, trick_room)
        return max(-1.0, min(1.0, my_best - opp_best))

    def _calc_sweep_ratio(self, attacker_side: Side, defender_side: Side,
                          weather: str, terrain: str,
                          trick_room: bool) -> float:
        """공격측 팀에서 최고 스윕 비율."""
        best = 0.0
        alive_defs = [p for p in defender_side.team if not p.fainted]
        if not alive_defs:
            return 1.0

        for atk in attacker_side.team:
            if atk.fainted:
                continue
            sweep_pts = 0.0
            for dfn in alive_defs:
                atk_dmg = self._best_move_damage(atk, dfn, weather, terrain)
                dfn_dmg = self._best_move_damage(dfn, atk, weather, terrain)

                atk_spe = atk.effective_speed(weather, terrain)
                dfn_spe = dfn.effective_speed(weather, terrain)
                if trick_room:
                    faster = atk_spe < dfn_spe
                else:
                    faster = atk_spe > dfn_spe

                # 우선도 오버라이드
                atk_prio = self._best_priority(atk, dfn, weather, terrain)
                dfn_prio = self._best_priority(dfn, atk, weather, terrain)
                if atk_prio > dfn_prio:
                    faster = True
                elif dfn_prio > atk_prio:
                    faster = False

                ohko = atk_dmg >= dfn.cur_hp
                survives_hit = atk.cur_hp > dfn_dmg

                if faster and ohko:
                    sweep_pts += 1.0
                elif ohko and survives_hit:
                    sweep_pts += 0.7
                elif faster and atk_dmg * 2 >= dfn.cur_hp:
                    sweep_pts += 0.3

            ratio = sweep_pts / len(alive_defs)
            best = max(best, ratio)

        return best

    # ─── 5. 카운터 유무 ─────────────────────────────────────────

    def _score_counter_availability(self, my: Side, opp: Side,
                                    weather: str, terrain: str) -> float:
        """상대 위협에 대한 체크/카운터 존재 여부. 범위 [-1, +1].
        양측 다 계산하여 차이를 냄 (zero-sum 보장)."""
        my_penalty = self._unchecked_ratio(my, opp, weather, terrain)
        opp_penalty = self._unchecked_ratio(opp, my, weather, terrain)
        return opp_penalty - my_penalty

    def _unchecked_ratio(self, defending: Side, attacking: Side,
                         weather: str, terrain: str) -> float:
        """attacking측의 위협 중 defending측이 체크하지 못하는 비율."""
        threats = [p for p in attacking.team if not p.fainted]
        if not threats:
            return 0.0

        unchecked = 0
        for threat in threats:
            has_counter = False
            for mine in defending.team:
                if mine.fainted:
                    continue
                # 카운터 조건: 1타 생존 + 33%↑ 데미지
                threat_dmg = self._best_move_damage(threat, mine, weather, terrain)
                if threat_dmg >= mine.cur_hp:
                    continue  # 1타 못 버팀
                my_dmg = self._best_move_damage(mine, threat, weather, terrain)
                if threat.cur_hp > 0 and my_dmg / threat.cur_hp >= 0.33:
                    has_counter = True
                    break
            if not has_counter:
                unchecked += 1

        return unchecked / len(threats)

    # ─── 6. 헤저드 ──────────────────────────────────────────────

    def _score_hazards(self, my: Side, opp: Side) -> float:
        """설치기 유불리. 범위 [-1, +1]."""
        my_penalty = self._hazard_pressure(my, opp)
        opp_penalty = self._hazard_pressure(opp, my)
        return opp_penalty - my_penalty

    def _hazard_pressure(self, target_side: Side, hazard_side: Side) -> float:
        """target_side에 깔린 상대 헤저드의 압박. 높을수록 불리."""
        pressure = 0.0

        # 스텔스록: 1/8 × 타입상성
        if target_side.stealth_rock:
            bench = [p for p in target_side.team
                     if not p.fainted and p != target_side.active]
            if bench:
                avg_eff = 0.0
                for p in bench:
                    eff = self.gd.effectiveness("Rock", p.types)
                    # 헤비듀티부츠 면역
                    if ITEM_EFFECTS.get(p.item, {}).get("hazard_immune"):
                        eff = 0.0
                    avg_eff += eff
                avg_eff /= len(bench)
                pressure += 0.3 * avg_eff  # 기본 0.3, 4배약이면 1.2

        # 압정 (1~3층: 1/8, 1/6, 1/4)
        if target_side.spikes > 0:
            spike_dmg = [0, 0.125, 0.167, 0.25][target_side.spikes]
            bench = [p for p in target_side.team
                     if not p.fainted and p != target_side.active]
            grounded_count = 0
            for p in bench:
                if not self._is_grounded(p):
                    continue
                if ITEM_EFFECTS.get(p.item, {}).get("hazard_immune"):
                    continue
                grounded_count += 1
            if bench:
                pressure += spike_dmg * (grounded_count / max(len(bench), 1))

        # 독압정
        if target_side.toxic_spikes > 0:
            bench = [p for p in target_side.team
                     if not p.fainted and p != target_side.active]
            vulnerable = 0
            for p in bench:
                if not self._is_grounded(p):
                    continue
                if ITEM_EFFECTS.get(p.item, {}).get("hazard_immune"):
                    continue
                if "Poison" in p.types or "Steel" in p.types:
                    continue  # 독/강철은 면역
                vulnerable += 1
            if bench:
                tox_val = 0.15 * target_side.toxic_spikes
                pressure += tox_val * (vulnerable / max(len(bench), 1))

        # 끈적망
        if target_side.sticky_web:
            bench = [p for p in target_side.team
                     if not p.fainted and p != target_side.active]
            grounded_count = 0
            for p in bench:
                if not self._is_grounded(p):
                    continue
                if ITEM_EFFECTS.get(p.item, {}).get("hazard_immune"):
                    continue
                grounded_count += 1
            if bench:
                pressure += 0.2 * (grounded_count / max(len(bench), 1))

        return pressure

    def _is_grounded(self, poke: Pokemon) -> bool:
        """지면에 닿아있는지 (비행, 부유, 풍선 체크)."""
        if "Flying" in poke.types:
            return False
        if ABILITY_EFFECTS.get(poke.ability, {}).get("ground_immune"):
            return False
        if ITEM_EFFECTS.get(poke.item, {}).get("levitate"):
            return False
        return True

    # ─── 7. 상태이상 ────────────────────────────────────────────

    def _score_status(self, my: Side, opp: Side) -> float:
        """상태이상 유불리. 범위 [-1.5, +1.5]."""
        my_pen = sum(self._status_penalty(p) for p in my.team if not p.fainted)
        opp_pen = sum(self._status_penalty(p) for p in opp.team if not p.fainted)
        return opp_pen - my_pen

    def _status_penalty(self, poke: Pokemon) -> float:
        """한 포켓몬의 상태이상 페널티."""
        st = poke.status
        if not st:
            return 0.0

        if st == "brn":
            # 물공 위주면 큰 손해
            if poke.stats.get("atk", 0) > poke.stats.get("spa", 0):
                return 0.4
            return 0.15

        if st == "par":
            base_spe = poke.stats.get("spe", 100)
            return 0.25 + (base_spe / 500.0) * 0.15

        if st == "tox":
            return 0.3

        if st == "psn":
            return 0.15

        if st == "slp":
            return 0.35

        if st == "frz":
            return 0.4

        return 0.0

    # ─── 8. 랭크업 ──────────────────────────────────────────────

    def _score_boosts(self, my: Side, opp: Side) -> float:
        """랭크업 가치. 범위 [-2, +2]."""
        my_val = sum(self._boost_value(p) for p in my.team if not p.fainted)
        opp_val = sum(self._boost_value(p) for p in opp.team if not p.fainted)
        return max(-2.0, min(2.0, my_val - opp_val))

    def _boost_value(self, poke: Pokemon) -> float:
        """한 포켓몬의 부스트 가치."""
        val = 0.0
        boosts = poke.boosts

        # 공격 부스트
        atk_boost = boosts.get("atk", 0)
        spa_boost = boosts.get("spa", 0)

        # 주력 공격스탯 결정
        main_atk_stat = max(poke.stats.get("atk", 100),
                            poke.stats.get("spa", 100))
        scale = main_atk_stat / 200.0

        if atk_boost > 0:
            val += atk_boost * 0.3 * (poke.stats.get("atk", 100) / 200.0)
        elif atk_boost < 0:
            val += atk_boost * 0.15 * (poke.stats.get("atk", 100) / 200.0)

        if spa_boost > 0:
            val += spa_boost * 0.3 * (poke.stats.get("spa", 100) / 200.0)
        elif spa_boost < 0:
            val += spa_boost * 0.15 * (poke.stats.get("spa", 100) / 200.0)

        # 스피드 부스트
        spe_boost = boosts.get("spe", 0)
        if spe_boost != 0:
            val += spe_boost * 0.15

        # 스피드+공격 동시 부스트 시너지 (스윕 잠재력)
        offensive_boost = max(atk_boost, spa_boost)
        if spe_boost > 0 and offensive_boost > 0:
            val += 0.2 * spe_boost

        # 방어 부스트
        def_boost = boosts.get("def", 0)
        spd_boost = boosts.get("spd", 0)
        val += def_boost * 0.1
        val += spd_boost * 0.1

        return val

    # ─── 9. 필드 조건 ───────────────────────────────────────────

    def _score_field(self, my: Side, opp: Side,
                     weather: str, terrain: str) -> float:
        """날씨/필드/벽/순풍 유불리. 범위 [-0.5, +0.5]."""
        score = 0.0

        # 벽 (내 리플렉트/빛의장막)
        if my.reflect_turns > 0:
            score += 0.15
        if my.light_screen_turns > 0:
            score += 0.15
        if opp.reflect_turns > 0:
            score -= 0.15
        if opp.light_screen_turns > 0:
            score -= 0.15

        # 순풍
        if my.tailwind_turns > 0:
            score += 0.2
        if opp.tailwind_turns > 0:
            score -= 0.2

        # 날씨 — 누가 더 이득인지
        if weather:
            my_benefit = self._weather_benefit(my, weather)
            opp_benefit = self._weather_benefit(opp, weather)
            score += (my_benefit - opp_benefit) * 0.15

        # 필드 — 누가 더 이득인지
        if terrain:
            my_benefit = self._terrain_benefit(my, terrain)
            opp_benefit = self._terrain_benefit(opp, terrain)
            score += (my_benefit - opp_benefit) * 0.15

        return max(-0.5, min(0.5, score))

    def _weather_benefit(self, side: Side, weather: str) -> float:
        """날씨로부터 얻는 이득."""
        benefit = 0.0
        for p in side.team:
            if p.fainted:
                continue
            ab_fx = ABILITY_EFFECTS.get(p.ability, {})
            # 스피드 날씨 특성 (쓱쓱 등)
            if ab_fx.get("speed_weather") == weather:
                benefit += 0.5
            # 공격 보정 (날씨 기술)
            for mid in p.moves:
                mv = self.gd.get_move(mid)
                if not mv:
                    continue
                if weather == "sun" and mv["type"] == "Fire":
                    benefit += 0.2
                elif weather == "rain" and mv["type"] == "Water":
                    benefit += 0.2
        return benefit

    def _terrain_benefit(self, side: Side, terrain: str) -> float:
        """필드로부터 얻는 이득."""
        benefit = 0.0
        type_map = {
            "electric": "Electric",
            "grassy": "Grass",
            "psychic": "Psychic",
        }
        boosted_type = type_map.get(terrain)
        for p in side.team:
            if p.fainted:
                continue
            if boosted_type:
                for mid in p.moves:
                    mv = self.gd.get_move(mid)
                    if mv and mv["type"] == boosted_type:
                        benefit += 0.2
                        break
        return benefit

    # ─── 10. 테라 자원 ──────────────────────────────────────────

    def _score_tera(self, my: Side, opp: Side) -> float:
        """테라 사용 가능 여부 차이. 범위 [-1, +1]."""
        my_tera = 0.0 if my.tera_used else 1.0
        opp_tera = 0.0 if opp.tera_used else 1.0
        return my_tera - opp_tera

    # ─── 11. 돌파 가능성 ──────────────────────────────────────

    def _has_recovery(self, poke: Pokemon) -> bool:
        """회복기(50%+) 또는 재생력 보유 여부."""
        # 재생력
        if ABILITY_EFFECTS.get(poke.ability, {}).get("switch_heal"):
            return True
        # 회복 기술 (flags.heal)
        for mid in poke.moves:
            mv = self.gd.get_move(mid)
            if mv and mv.get("flags", {}).get("heal"):
                return True
        return False

    def _has_anti_stall(self, poke: Pokemon) -> bool:
        """사이클 방해 도구 보유 (도발/트릭/맹독/앵콜)."""
        anti_stall_moves = {
            "taunt", "trick", "switcheroo", "toxic", "encore",
            "knockoff", "haze", "clearsmog", "perishsong",
        }
        for mid in poke.moves:
            if mid in anti_stall_moves:
                return True
        return False

    def _score_breakthrough(self, my: Side, opp: Side,
                            weather: str, terrain: str) -> float:
        """회복기 가진 상대를 뚫을 수 있는가. 범위 [-1, +1].
        뚫을 수 없으면 감점, 상대가 뚫을 수 없으면 가점."""
        my_stuck = self._breakthrough_penalty(my, opp, weather, terrain)
        opp_stuck = self._breakthrough_penalty(opp, my, weather, terrain)
        return opp_stuck - my_stuck

    def _breakthrough_penalty(self, attacker: Side, defender: Side,
                              weather: str, terrain: str) -> float:
        """attacker가 defender의 회복기 포켓몬을 못 뚫는 정도.

        핵심: 상대 남은 수가 적을수록 벽 1마리의 중요도가 폭증.
        - 3마리 중 1벽: 귀찮지만 나머지 먼저 잡으면 됨
        - 1마리=벽: 이놈 못 잡으면 절대 못 이김
        """
        penalty = 0.0
        defender_alive = [p for p in defender.team if not p.fainted]
        n_alive = len(defender_alive)
        if n_alive == 0:
            return 0.0

        for wall in defender_alive:
            if not self._has_recovery(wall):
                continue

            # 천진(Unaware) 체크 — 부스트 무시
            wall_ignores_boosts = ABILITY_EFFECTS.get(
                wall.ability, {}).get("ignore_boosts", False)

            # attacker 팀에서 최강 데미지 찾기
            can_break = False
            has_anti_tool = False
            team_best_dmg = 0

            for atk in attacker.team:
                if atk.fainted:
                    continue

                # 사이클 방해 도구 체크
                if self._has_anti_stall(atk):
                    has_anti_tool = True

                # 데미지 계산 (천진이면 부스트 0으로)
                best_dmg = 0
                for mid in atk.moves:
                    mv = self.gd.get_move(mid)
                    if not mv or mv["is_status"] or mv["basePower"] <= 0:
                        continue
                    dmg = self._calc_damage_fast(
                        atk, wall, mv, weather, terrain)
                    # 천진이면 부스트 없이 재계산
                    if wall_ignores_boosts and any(
                            atk.boosts.get(k, 0) != 0
                            for k in ("atk", "spa")):
                        dmg = self._calc_damage_no_boost(
                            atk, wall, mv, weather, terrain)
                    if dmg > best_dmg:
                        best_dmg = dmg

                if best_dmg > team_best_dmg:
                    team_best_dmg = best_dmg

                # 2타 이내 = 돌파 가능 (회복 턴에 잡기)
                if wall.max_hp > 0 and best_dmg / wall.max_hp >= 0.50:
                    can_break = True
                    break

            if can_break:
                continue

            # ── 못 뚫는 벽 발견 ──

            # 벽 중요도: 상대 남은 수에 반비례
            if n_alive <= 1:
                base_penalty = 2.0   # 이놈만 잡으면 되는데 못 잡음 = 거의 패
            elif n_alive == 2:
                base_penalty = 1.0   # 벽이 절반 = 심각
            else:
                base_penalty = 0.5   # 나머지 먼저 처리 가능

            # 벽이 공격도 되면 추가 감점 (Sacred Fire, Scald 등)
            # 시간 끌수록 우리만 손해
            wall_threatens = False
            atk_alive = [p for p in attacker.team if not p.fainted]
            if atk_alive:
                for target in atk_alive:
                    wall_dmg = self._best_move_damage(
                        wall, target, weather, terrain)
                    # 벽이 4타 이내 잡을 수 있으면 = 공격적 벽
                    if target.max_hp > 0 and wall_dmg / target.max_hp >= 0.25:
                        wall_threatens = True
                        break

            if has_anti_tool:
                penalty += 0.2
            elif wall_threatens:
                # 못 뚫는 벽 + 공격까지 함 = 시간 끌면 패
                penalty += base_penalty * 1.3
            else:
                penalty += base_penalty

        return min(2.5, penalty)

    def _calc_damage_no_boost(self, attacker: Pokemon, defender: Pokemon,
                              move: dict, weather: str = "",
                              terrain: str = "") -> int:
        """부스트 0으로 데미지 계산 (천진 대응)."""
        # 임시로 부스트 제거 후 계산
        saved = dict(attacker.boosts)
        attacker.boosts = {k: 0 for k in attacker.boosts}
        dmg = self._calc_damage_fast(attacker, defender, move, weather, terrain)
        attacker.boosts = saved
        return dmg

    # ─── 12. 방어 코어 ─────────────────────────────────────

    def _score_defensive_core(self, my: Side, opp: Side,
                              weather: str, terrain: str) -> float:
        """상대 물벽+특벽 사이클 성립 여부. 범위 [-1, +1].
        상대 방어 코어 생존 = 감점, 내 코어 생존 = 가점."""
        my_core = self._has_defensive_core(my)
        opp_core = self._has_defensive_core(opp)

        score = 0.0
        if opp_core:
            score -= 0.5  # 상대 사이클 성립 가능
        if my_core:
            score += 0.5  # 내 사이클 성립 가능

        # 상대 코어 + 회복기 = 더 강력한 사이클
        if opp_core:
            opp_alive = [p for p in opp.team if not p.fainted]
            recovery_count = sum(1 for p in opp_alive if self._has_recovery(p))
            if recovery_count >= 2:
                score -= 0.3  # 회복기 2마리 이상 = 철벽 사이클
        if my_core:
            my_alive = [p for p in my.team if not p.fainted]
            recovery_count = sum(1 for p in my_alive if self._has_recovery(p))
            if recovery_count >= 2:
                score += 0.3

        return max(-1.0, min(1.0, score))

    def _has_defensive_core(self, side: Side) -> bool:
        """물리벽 + 특수벽이 모두 생존해 있는가."""
        alive = [p for p in side.team if not p.fainted]
        if len(alive) < 2:
            return False

        has_phys_wall = False
        has_spec_wall = False

        for p in alive:
            def_stat = p.stats.get("def", 0)
            spd_stat = p.stats.get("spd", 0)
            hp_stat = p.stats.get("hp", 0)

            # 물리벽: 방어 높고 HP 괜찮은 놈
            if def_stat >= 110 and hp_stat >= 80:
                has_phys_wall = True
            # 특수벽: 특방 높고 HP 괜찮은 놈
            if spd_stat >= 110 and hp_stat >= 80:
                has_spec_wall = True

        return has_phys_wall and has_spec_wall

    # ═══════════════════════════════════════════════════════════════
    #  팀 선출 (Team Selection) — 6마리에서 최적 3마리 선택
    # ═══════════════════════════════════════════════════════════════

    def select_team(self, my_team_6: list, opp_team_6: list,
                    weather: str = "", terrain: str = "") -> list:
        """6마리에서 최적 3마리 선출 + 리드 결정.

        상대 6마리 전체를 보고, 어떤 3마리를 데려가든 대응할 수 있는 조합 선택.
        C(6,3)=20개 조합 평가 → 최고 점수 반환.
        리드 적합도가 가장 높은 포켓몬을 인덱스 0에 배치.
        """
        from itertools import combinations

        if len(my_team_6) <= 3:
            team = list(my_team_6)
        else:
            best_score = -float('inf')
            team = list(my_team_6[:3])

            for combo in combinations(my_team_6, 3):
                candidates = list(combo)
                score = self._eval_team_selection(
                    candidates, opp_team_6, weather, terrain)
                if score > best_score:
                    best_score = score
                    team = candidates

        # 리드 결정: 리드 적합도가 가장 높은 놈을 [0]에
        team.sort(key=lambda p: self._lead_score(p, opp_team_6, weather, terrain),
                  reverse=True)
        return team

    # ─── 리드 적합도 ──────────────────────────────────────────

    # 선발용 기술 세트
    _HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
    _PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot",
                    "teleport", "batonpass"}
    _LEAD_MOVES = {"fakeout", "taunt", "encore"}
    _SETUP_MOVES = {
        "swordsdance", "dragondance", "nastyplot", "calmmind",
        "quiverdance", "bellydrum", "shellsmash", "bulkup",
        "irondefense", "amnesia", "coil", "shiftgear",
        "tidyup", "victorydance", "tailwind",
    }

    def _lead_score(self, poke, opp_team: list,
                    weather: str, terrain: str) -> float:
        """리드 적합도 점수. 높을수록 선발로 적합.

        가산:
          헤저드 기술 보유 (+2.0) — 스텔록/압정은 빨리 깔아야 함
          페이크아웃/도발/앵콜 (+1.5) — 초동 장악
          유턴/볼트체인지 (+1.0) — 정보 수집 + 안전 교체
          기합의띠 (+0.8) — 1타 버티고 일 함
          고속 (스피드 130+: +0.5, 100+: +0.2)

        감산:
          세팅기(검무/용무/나비춤 등) (-1.5) — 후반 에이스, 선발 낭비
          회복기 보유 (-0.5) — 받이는 중반에 투입
        """
        score = 0.0
        move_set = set(poke.moves)

        # 헤저드 세터 → 선발 최우선
        if move_set & self._HAZARD_MOVES:
            score += 2.0

        # 초동 장악기
        if move_set & self._LEAD_MOVES:
            score += 1.5

        # 피벗 (유턴/볼트체인지)
        if move_set & self._PIVOT_MOVES:
            score += 1.0

        # 기합의띠
        if poke.item == "focussash":
            score += 0.8

        # 고속 → 선발 유리
        spe = poke.effective_speed("", "")
        if spe >= 130:
            score += 0.5
        elif spe >= 100:
            score += 0.2

        # 세팅 스위퍼 → 선발 부적합 (후반 에이스)
        if move_set & self._SETUP_MOVES:
            score -= 1.5

        # 회복기 보유 → 받이 역할, 중반 투입
        if self._has_recovery(poke):
            score -= 0.5

        # 대면 유불리 — 확킬 기준
        i_ohko = 0    # 내가 상대를 확1
        they_ohko = 0  # 상대가 나를 확1
        for opp in opp_team:
            my_dmg = self._best_move_damage(poke, opp, weather, terrain)
            opp_dmg = self._best_move_damage(opp, poke, weather, terrain)
            if opp.max_hp > 0 and my_dmg >= opp.max_hp:
                i_ohko += 1
            if poke.max_hp > 0 and opp_dmg >= poke.max_hp:
                they_ohko += 1
        if opp_team:
            n = len(opp_team)
            score += 1.0 * (i_ohko / n)      # 내가 작살내는 대면 많으면 가산
            score -= 1.5 * (they_ohko / n)    # 내가 작살나는 대면 많으면 큰 감점

        return score

    def _eval_team_selection(self, my_3: list, opp_6: list,
                             weather: str, terrain: str) -> float:
        """선출 조합 점수. 높을수록 좋음.

        5개 컴포넌트:
        1. 위협 커버리지 (2.0) — 상대 각 포켓몬을 체크할 수 있는가
        2. 킬 프레셔 (1.5) — 상대를 얼마나 빨리 잡는가
        3. 스윕 포텐셜 (1.0) — 승리 루트 존재
        4. 방어 시너지 (1.0) — 약점 겹침 방지
        5. 스피드 밸런스 (0.5) — 속도 다양성
        """
        score = 0.0

        # 1. 위협 커버리지 (가중치 2.0) — 가장 중요
        score += 2.0 * self._sel_threat_coverage(my_3, opp_6, weather, terrain)

        # 2. 킬 프레셔 (가중치 1.5)
        score += 1.5 * self._sel_kill_pressure(my_3, opp_6, weather, terrain)

        # 3. 스윕 포텐셜 (가중치 1.0)
        score += 1.0 * self._sel_sweep_potential(my_3, opp_6, weather, terrain)

        # 4. 방어 시너지 (가중치 1.0)
        score += 1.0 * self._sel_defensive_synergy(my_3)

        # 5. 스피드 밸런스 (가중치 0.5)
        score += 0.5 * self._sel_speed_balance(my_3)

        return score

    # ─── 선출 1: 위협 커버리지 ─────────────────────────────────

    def _sel_threat_coverage(self, my_3: list, opp_6: list,
                             weather: str, terrain: str) -> float:
        """상대 위협 커버 비율. [0, 1].

        상대 각 포켓몬에 대해: 내 팀에 '체크' 가능한 놈이 있는가?
        체크 = 상대 최강기 1타 생존 + 내가 33%↑ 데미지.
        """
        if not opp_6:
            return 1.0

        covered = 0
        for opp_mon in opp_6:
            for my_mon in my_3:
                # 상대의 나에 대한 최강 데미지
                opp_dmg = self._best_move_damage(opp_mon, my_mon, weather, terrain)
                # 1타 생존?
                if opp_dmg >= my_mon.max_hp:
                    continue
                # 내가 상대에게 33%+ 데미지?
                my_dmg = self._best_move_damage(my_mon, opp_mon, weather, terrain)
                if opp_mon.max_hp > 0 and my_dmg / opp_mon.max_hp >= 0.33:
                    covered += 1
                    break
            # break가 안 걸리면 = uncovered threat

        return covered / len(opp_6)

    # ─── 선출 2: 킬 프레셔 ────────────────────────────────────

    def _sel_kill_pressure(self, my_3: list, opp_6: list,
                           weather: str, terrain: str) -> float:
        """상대를 얼마나 빨리 잡는가. [0, 1].

        확1 = 1.0점, 확2 = 0.7점, 확3 = 0.3점.
        """
        if not opp_6:
            return 1.0

        total = 0.0
        for opp_mon in opp_6:
            best_ratio = 0.0
            for my_mon in my_3:
                dmg = self._best_move_damage(my_mon, opp_mon, weather, terrain)
                if opp_mon.max_hp > 0:
                    ratio = dmg / opp_mon.max_hp
                    best_ratio = max(best_ratio, ratio)

            if best_ratio >= 1.0:
                total += 1.0
            elif best_ratio >= 0.5:
                total += 0.7
            elif best_ratio >= 0.33:
                total += 0.3

        return total / len(opp_6)

    # ─── 선출 3: 스윕 포텐셜 ──────────────────────────────────

    def _sel_sweep_potential(self, my_3: list, opp_6: list,
                             weather: str, terrain: str) -> float:
        """승리 루트 — 한 마리가 상대 다수를 잡을 수 있는가. [0, 1]."""
        if not opp_6:
            return 1.0

        best_ratio = 0.0
        for my_mon in my_3:
            killable = 0.0
            for opp_mon in opp_6:
                dmg = self._best_move_damage(my_mon, opp_mon, weather, terrain)
                opp_dmg = self._best_move_damage(opp_mon, my_mon, weather, terrain)

                # 스피드: 아이템(스카프) 반영, 날씨/부스트는 없음
                my_spe = my_mon.effective_speed("", "")
                opp_spe = opp_mon.effective_speed("", "")
                faster = my_spe > opp_spe

                # 우선도 오버라이드
                prio = self._best_priority(my_mon, opp_mon, weather, terrain)
                opp_prio = self._best_priority(opp_mon, my_mon, weather, terrain)
                if prio > opp_prio:
                    faster = True
                elif opp_prio > prio:
                    faster = False

                ohko = dmg >= opp_mon.max_hp
                survives = my_mon.max_hp > opp_dmg

                if faster and ohko:
                    killable += 1.0
                elif ohko and survives:
                    killable += 0.7

            ratio = killable / len(opp_6)
            best_ratio = max(best_ratio, ratio)

        return best_ratio

    # ─── 선출 4: 방어 시너지 ──────────────────────────────────

    _ALL_TYPES = [
        "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
        "Fighting", "Poison", "Ground", "Flying", "Psychic",
        "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
    ]

    def _sel_defensive_synergy(self, my_3: list) -> float:
        """타입 약점 겹침 방지 + 내성 다양성. [-1, 1]."""
        # 각 공격 타입에 대해 약점인 내 포켓몬 수 카운트
        triple_weak = 0
        double_weak = 0
        resist_types = 0

        for atk_type in self._ALL_TYPES:
            weak_count = 0
            has_resist = False
            for p in my_3:
                eff = self.gd.effectiveness(atk_type, p.types)
                if eff > 1.0:
                    weak_count += 1
                if eff < 1.0:
                    has_resist = True
            if weak_count >= 3:
                triple_weak += 1
            elif weak_count >= 2:
                double_weak += 1
            if has_resist:
                resist_types += 1

        score = 0.0
        score -= triple_weak * 0.5   # 3마리 모두 약한 타입 = 치명적
        score -= double_weak * 0.15  # 2마리가 약한 타입
        score += (resist_types / len(self._ALL_TYPES)) * 0.5  # 내성 커버 비율

        return max(-1.0, min(1.0, score))

    # ─── 선출 5: 스피드 밸런스 ────────────────────────────────

    def _sel_speed_balance(self, my_3: list) -> float:
        """스피드 다양성. [-0.5, 1.0]."""
        speeds = sorted([p.effective_speed("", "") for p in my_3])
        fastest = speeds[-1]

        # 최소 1마리 빠른 놈
        if fastest >= 130:
            score = 0.5
        elif fastest >= 100:
            score = 0.2
        else:
            score = -0.5  # 전부 느림 = 선공 못 잡음

        # 다양성 보너스 — 빠른 놈 + 느린 놈 다 있으면
        speed_range = speeds[-1] - speeds[0]
        if speed_range >= 50:
            score += 0.3

        return max(-0.5, min(1.0, score))

    # ═══════════════════════════════════════════════════════════════
    #  데미지 계산 헬퍼 (순수 Python — 텐서 없음, 빠름)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _boost_mul(stage: int) -> float:
        stage = max(-6, min(6, stage))
        if stage >= 0:
            return (2 + stage) / 2.0
        return 2.0 / (2 - stage)

    def _calc_damage_fast(self, attacker: Pokemon, defender: Pokemon,
                          move: dict, weather: str = "",
                          terrain: str = "") -> int:
        """순수 Python 기대 데미지 (텐서 없음). 평균 롤 0.925 사용."""
        if move["is_status"] or move["basePower"] <= 0:
            return 0

        level = 50
        is_phys = move["is_physical"]
        atk_key = "atk" if is_phys else "spa"
        def_key = "def" if is_phys else "spd"

        atk_val = float(attacker.stats[atk_key])
        def_val = float(defender.stats[def_key])

        # 부스트
        atk_val *= self._boost_mul(attacker.boosts.get(atk_key, 0))
        def_val *= self._boost_mul(defender.boosts.get(def_key, 0))

        # STAB
        move_type = move["type"]
        tera_type = attacker.tera_type if attacker.is_tera else None
        # 테라 후 attacker.types는 [tera_type]으로 변경되어 있음
        # 원래 타입은 dex에서 가져와야 정확함
        if attacker.is_tera:
            dex = self.gd.get_pokemon(attacker.species_id)
            original_types = dex.get("types", attacker.types) if dex else attacker.types
        else:
            original_types = attacker.types

        if tera_type:
            if move_type == tera_type and move_type in original_types:
                stab = 2.0
            elif move_type == tera_type or move_type in original_types:
                stab = 1.5
            else:
                stab = 1.0
        else:
            stab = 1.5 if move_type in attacker.types else 1.0

        # 특성 STAB
        atk_ability = attacker.ability
        ab_fx = ABILITY_EFFECTS.get(atk_ability, {})
        if ab_fx.get("stab_mod"):
            all_stab = set(original_types)
            if tera_type:
                all_stab.add(tera_type)
            if move_type in all_stab:
                stab = ab_fx["stab_mod"]

        # 타입상성 (특성 연동)
        def_ability = defender.ability
        def_ab_fx_imm = ABILITY_EFFECTS.get(def_ability, {})
        is_mold_breaker = ab_fx.get("mold_breaker", False)

        type_eff = self.gd.effectiveness(move_type, defender.types)

        # 타입 면역을 특성이 뚫는 경우 (심안/배짱)
        if type_eff == 0:
            if ab_fx.get("ignore_ghost_immune"):
                if move_type in ("Normal", "Fighting") and "Ghost" in defender.types:
                    other = [t for t in defender.types if t != "Ghost"]
                    type_eff = self.gd.effectiveness(move_type, other) if other else 1.0
                    if type_eff == 0:
                        type_eff = 1.0
            if type_eff == 0:
                return 0

        # 특성 기반 면역 (틀깨기가 아니면)
        if not is_mold_breaker:
            ability_immune_map = {
                "ground_immune": "Ground",
                "fire_immune": "Fire",
                "electric_immune": "Electric",
                "water_immune": "Water",
                "grass_immune": "Grass",
            }
            for key, imm_type in ability_immune_map.items():
                if def_ab_fx_imm.get(key) and move_type == imm_type:
                    return 0
            if def_ab_fx_imm.get("wonder_guard") and type_eff <= 1.0:
                return 0

        # 날씨
        weather_mod = 1.0
        if weather == "sun":
            if move_type == "Fire":
                weather_mod = 1.5
            elif move_type == "Water":
                weather_mod = 0.5
        elif weather == "rain":
            if move_type == "Water":
                weather_mod = 1.5
            elif move_type == "Fire":
                weather_mod = 0.5

        # 필드
        terrain_mod = 1.0
        if terrain == "electric" and move_type == "Electric":
            terrain_mod = 1.3
        elif terrain == "grassy" and move_type == "Grass":
            terrain_mod = 1.3
        elif terrain == "psychic" and move_type == "Psychic":
            terrain_mod = 1.3
        elif terrain == "misty" and move_type == "Dragon":
            terrain_mod = 0.5

        # 아이템
        item_mod = 1.0
        atk_item = attacker.item if not attacker.item_consumed else ""
        item_fx = ITEM_EFFECTS.get(atk_item, {})
        if "damage_mod" in item_fx:
            item_mod *= item_fx["damage_mod"]
        if "super_eff_mod" in item_fx and type_eff > 1.0:
            item_mod *= item_fx["super_eff_mod"]

        if atk_item in _TYPE_BOOST_ITEMS:
            bt, bv = _TYPE_BOOST_ITEMS[atk_item]
            if move_type == bt:
                item_mod *= bv

        # 방어측 아이템
        def_item = defender.item if not defender.item_consumed else ""
        def_item_fx = ITEM_EFFECTS.get(def_item, {})
        if "spd_mod" in def_item_fx and move["is_special"]:
            def_val *= def_item_fx["spd_mod"]
        if "def_mod" in def_item_fx and "spd_mod" in def_item_fx:
            if is_phys:
                def_val *= def_item_fx["def_mod"]
            else:
                def_val *= def_item_fx["spd_mod"]

        # 공격측 특성
        ability_mod = 1.0
        if ab_fx.get("atk_mod") and is_phys:
            ability_mod *= ab_fx["atk_mod"]
        if ab_fx.get("low_bp_mod") and move["basePower"] <= ab_fx.get("bp_threshold", 60):
            ability_mod *= ab_fx["low_bp_mod"]
        if ab_fx.get("contact_mod") and move.get("flags", {}).get("contact"):
            ability_mod *= ab_fx["contact_mod"]
        if ab_fx.get("punch_mod") and move.get("flags", {}).get("punch"):
            ability_mod *= ab_fx["punch_mod"]
        if ab_fx.get("bite_mod") and move.get("flags", {}).get("bite"):
            ability_mod *= ab_fx["bite_mod"]
        if ab_fx.get("secondary_mod") and move.get("secondary"):
            ability_mod *= ab_fx["secondary_mod"]

        # 방어측 특성
        def_ab_fx = ABILITY_EFFECTS.get(defender.ability, {})
        if def_ab_fx.get("super_eff_mod") and type_eff > 1.0:
            ability_mod *= def_ab_fx["super_eff_mod"]
        if def_ab_fx.get("full_hp_damage_mod"):
            if defender.cur_hp >= defender.max_hp:
                ability_mod *= def_ab_fx["full_hp_damage_mod"]
        if def_ab_fx.get("special_damage_mod") and move["is_special"]:
            ability_mod *= def_ab_fx["special_damage_mod"]
        if def_ab_fx.get("fire_resist") and move_type == "Fire":
            ability_mod *= def_ab_fx["fire_resist"]
        if def_ab_fx.get("ice_resist") and move_type == "Ice":
            ability_mod *= def_ab_fx["ice_resist"]

        # Ruin abilities
        if def_ab_fx.get("foe_atk_mod") and is_phys:
            atk_val *= def_ab_fx["foe_atk_mod"]
        if def_ab_fx.get("foe_spa_mod") and move["is_special"]:
            atk_val *= def_ab_fx["foe_spa_mod"]
        if ab_fx.get("foe_def_mod") and is_phys:
            def_val *= ab_fx["foe_def_mod"]
        if ab_fx.get("foe_spd_mod") and move["is_special"]:
            def_val *= ab_fx["foe_spd_mod"]

        # 화상
        is_burned = (attacker.status == "brn" and is_phys
                     and not ab_fx.get("ignore_burn"))
        burn_mod = 0.5 if is_burned else 1.0

        power = float(move["basePower"])
        base = ((2.0 * level / 5.0 + 2.0) * power * atk_val / def_val) / 50.0 + 2.0
        damage = base * 0.925 * stab * type_eff * burn_mod
        damage *= weather_mod * item_mod * ability_mod * terrain_mod

        return max(1, int(damage))

    def _best_move_damage(self, attacker: Pokemon, defender: Pokemon,
                          weather: str = "", terrain: str = "") -> int:
        """공격자의 최강 기술 기대 데미지 반환 (순수 Python)."""
        best_dmg = 0
        for mid in attacker.moves:
            mv = self.gd.get_move(mid)
            if not mv or mv["is_status"] or mv["basePower"] <= 0:
                continue
            dmg = self._calc_damage_fast(attacker, defender, mv, weather, terrain)
            if dmg > best_dmg:
                best_dmg = dmg
        return best_dmg

    def _best_priority(self, attacker: Pokemon, defender: Pokemon,
                       weather: str = "", terrain: str = "") -> int:
        """공격자가 가진 기술 중 의미 있는 최고 우선도 반환."""
        best_prio = 0
        for mid in attacker.moves:
            mv = self.gd.get_move(mid)
            if not mv or mv["is_status"]:
                continue
            prio = mv.get("priority", 0)
            if prio <= 0:
                continue
            dmg = self._calc_damage_fast(attacker, defender, mv, weather, terrain)
            if dmg > 0 and prio > best_prio:
                best_prio = prio
        return best_prio


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """RuleBasedEvaluator 기본 검증."""
    print("=== RuleBasedEvaluator 검증 ===\n")

    from battle_sim import BattleSimulator, make_pokemon_from_stats

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)
    evaluator = RuleBasedEvaluator(gd)

    # 테스트 1: 기본 평가
    team1 = [make_pokemon_from_stats(gd, n, "bss")
             for n in ["Koraidon", "Ting-Lu", "Flutter Mane"]]
    team2 = [make_pokemon_from_stats(gd, n, "bss")
             for n in ["Miraidon", "Gholdengo", "Great Tusk"]]
    state = sim.create_battle_state(team1, team2)

    v0 = evaluator.evaluate_value(state, 0)
    v1 = evaluator.evaluate_value(state, 1)
    print(f"초기 상태: P0={v0:+.4f}, P1={v1:+.4f}")
    print(f"  (대칭이면 부호 반대)")

    # 테스트 2: 3v1 상황
    state_3v1 = state.clone()
    for p in state_3v1.sides[1].team[1:]:
        p.fainted = True
        p.cur_hp = 0
    v_3v1 = evaluator.evaluate_value(state_3v1, 0)
    print(f"\n3v1 (P0 유리): {v_3v1:+.4f} (≥ +0.8 예상)")

    # 테스트 3: 1v3 상황
    v_1v3 = evaluator.evaluate_value(state_3v1, 1)
    print(f"1v3 (P1 관점): {v_1v3:+.4f} (≤ -0.8 예상)")

    # 테스트 4: MCTS 인터페이스
    legal = sim.get_legal_actions(state, 0)
    priors, val = evaluator.evaluate(state, 0, legal)
    print(f"\nevaluate() 테스트:")
    print(f"  Legal: {legal}")
    print(f"  Prior sum: {sum(priors.values()):.4f}")
    print(f"  Value: {val:+.4f}")

    # 테스트 5: 배치 평가
    states = [state, state_3v1]
    values = evaluator.evaluate_values_batch(states, 0)
    print(f"\n배치 평가: {values}")

    # 테스트 6: 랭업 상태
    state_boost = state.clone()
    state_boost.sides[1].active.boosts["atk"] = 2
    state_boost.sides[1].active.boosts["spe"] = 2
    v_boost = evaluator.evaluate_value(state_boost, 0)
    print(f"\n상대 +2공+2스 후 P0 관점: {v_boost:+.4f} (기본보다 낮아야 함, 기본={v0:+.4f})")

    # 성능 테스트
    import time
    N = 1000
    start = time.time()
    for _ in range(N):
        evaluator.evaluate_value(state, 0)
    elapsed = time.time() - start
    print(f"\n성능: {N}회 평가 = {elapsed*1000:.1f}ms "
          f"({elapsed/N*1000:.3f}ms/eval)")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
