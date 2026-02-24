"""경량 배틀 시뮬레이터 — MCTS 롤아웃용 텐서 기반 상태.

100% 정확할 필요 없음 → Showdown 서버 대신 근사 엔진으로 빠르게 시뮬.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch

from data_loader import (
    GameData, TYPE_TO_IDX, STATUS_TO_IDX, STAT_KEYS, _to_id,
    ABILITY_EFFECTS, ITEM_EFFECTS,
)
from damage_calc import DamageCalculator


# ═══════════════════════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════════════════════

class ActionType(IntEnum):
    MOVE1 = 0
    MOVE2 = 1
    MOVE3 = 2
    MOVE4 = 3
    SWITCH1 = 4
    SWITCH2 = 5
    SWITCH3 = 6
    SWITCH4 = 7
    SWITCH5 = 8
    TERA_MOVE1 = 9
    TERA_MOVE2 = 10
    TERA_MOVE3 = 11
    TERA_MOVE4 = 12

NUM_ACTIONS = 13  # 기술4 + 교체5 + 테라기술4

class Weather(IntEnum):
    NONE = 0; SUN = 1; RAIN = 2; SAND = 3; SNOW = 4

class Terrain(IntEnum):
    NONE = 0; ELECTRIC = 1; GRASSY = 2; MISTY = 3; PSYCHIC = 4

WEATHER_STR = {Weather.NONE: "", Weather.SUN: "sun", Weather.RAIN: "rain",
               Weather.SAND: "sand", Weather.SNOW: "snow"}
TERRAIN_STR = {Terrain.NONE: "", Terrain.ELECTRIC: "electric",
               Terrain.GRASSY: "grassy", Terrain.MISTY: "misty",
               Terrain.PSYCHIC: "psychic"}


# ═══════════════════════════════════════════════════════════════
#  포켓몬 / 배틀 상태
# ═══════════════════════════════════════════════════════════════

@dataclass
class Pokemon:
    """단일 포켓몬 상태."""
    species_id: str
    name: str
    types: list[str]
    base_stats: dict[str, int]
    stats: dict[str, int]       # 실전 스탯 (레벨 50)
    ability: str
    item: str
    moves: list[str]            # move_id 4개
    cur_hp: int = 0
    max_hp: int = 0
    status: str = ""            # brn, par, psn, tox, slp, frz
    tox_counter: int = 0
    boosts: dict = field(default_factory=lambda: {k: 0 for k in STAT_KEYS[1:]})
    tera_type: str = ""         # 테라스탈 타입
    is_tera: bool = False
    fainted: bool = False
    has_moved: bool = False     # 이번 턴 행동 여부
    protect_count: int = 0      # 연속 방어 카운트
    protect_active: bool = False  # 이번 턴 방어 활성
    sleep_turns: int = 0        # 잠듦 경과 턴
    choice_locked_move: str = ""  # 초이스 잠금 기술
    substitute_hp: int = 0      # 대타출동 HP
    item_consumed: bool = False
    pp: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    max_pp: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def __post_init__(self):
        if self.cur_hp <= 0:
            self.cur_hp = self.stats["hp"]
        if self.max_hp <= 0:
            self.max_hp = self.stats["hp"]

    @property
    def hp_pct(self) -> float:
        return self.cur_hp / max(self.max_hp, 1)

    def effective_speed(self, weather: str = "", terrain: str = "") -> float:
        """실효 스피드 (부스트, 마비, 아이템, 스카프, 날씨/필드 특성 포함)."""
        spe = self.stats["spe"]
        boost = self.boosts.get("spe", 0)
        if boost > 0:
            spe = spe * (2 + boost) / 2
        elif boost < 0:
            spe = spe * 2 / (2 - boost)
        if self.status == "par":
            spe *= 0.5
        item_fx = ITEM_EFFECTS.get(self.item, {})
        if item_fx.get("speed_mod") and not self.item_consumed:
            spe *= item_fx["speed_mod"]
        # 속도 특성 (Swift Swim, Chlorophyll 등)
        ability_fx = ABILITY_EFFECTS.get(self.ability, {})
        if ability_fx.get("speed_weather") and ability_fx["speed_weather"] == weather:
            spe *= ability_fx.get("speed_mul", 1.0)
        if ability_fx.get("speed_terrain") and ability_fx["speed_terrain"] == terrain:
            spe *= ability_fx.get("speed_mul", 1.0)
        return spe

    def clone(self) -> Pokemon:
        return Pokemon(
            species_id=self.species_id, name=self.name,
            types=list(self.types), base_stats=dict(self.base_stats),
            stats=dict(self.stats), ability=self.ability, item=self.item,
            moves=list(self.moves), cur_hp=self.cur_hp, max_hp=self.max_hp,
            status=self.status, tox_counter=self.tox_counter,
            boosts=dict(self.boosts), tera_type=self.tera_type,
            is_tera=self.is_tera, fainted=self.fainted,
            has_moved=self.has_moved, protect_count=self.protect_count,
            protect_active=self.protect_active,
            sleep_turns=self.sleep_turns,
            choice_locked_move=self.choice_locked_move,
            substitute_hp=self.substitute_hp,
            item_consumed=self.item_consumed,
            pp=list(self.pp),
            max_pp=list(self.max_pp),
        )


@dataclass
class Side:
    """한 플레이어의 상태."""
    team: list[Pokemon]
    active_idx: int = 0
    tera_used: bool = False
    # 설치기
    stealth_rock: bool = False
    spikes: int = 0        # 0~3
    toxic_spikes: int = 0  # 0~2
    sticky_web: bool = False
    reflect_turns: int = 0
    light_screen_turns: int = 0
    tailwind_turns: int = 0

    @property
    def active(self) -> Pokemon:
        return self.team[self.active_idx]

    @property
    def alive_count(self) -> int:
        return sum(1 for p in self.team if not p.fainted)

    @property
    def bench(self) -> list[tuple[int, Pokemon]]:
        """교체 가능한 벤치 포켓몬."""
        return [(i, p) for i, p in enumerate(self.team)
                if i != self.active_idx and not p.fainted]

    def clone(self) -> Side:
        return Side(
            team=[p.clone() for p in self.team],
            active_idx=self.active_idx,
            tera_used=self.tera_used,
            stealth_rock=self.stealth_rock,
            spikes=self.spikes,
            toxic_spikes=self.toxic_spikes,
            sticky_web=self.sticky_web,
            reflect_turns=self.reflect_turns,
            light_screen_turns=self.light_screen_turns,
            tailwind_turns=self.tailwind_turns,
        )


@dataclass
class BattleState:
    """배틀 전체 상태."""
    sides: list[Side]  # [0] = P1, [1] = P2
    weather: Weather = Weather.NONE
    weather_turns: int = 0
    terrain: Terrain = Terrain.NONE
    terrain_turns: int = 0
    trick_room_turns: int = 0
    turn: int = 0
    winner: int = -1  # -1=진행중, 0=P1승, 1=P2승
    turn_log: list = None  # 턴 내 이벤트 로그

    @property
    def is_terminal(self) -> bool:
        return self.winner >= 0

    def log_event(self, msg: str):
        """턴 내 이벤트 기록 (turn_log가 활성화된 경우에만)."""
        if self.turn_log is not None:
            self.turn_log.append(msg)

    def clone(self) -> BattleState:
        return BattleState(
            sides=[s.clone() for s in self.sides],
            weather=self.weather, weather_turns=self.weather_turns,
            terrain=self.terrain, terrain_turns=self.terrain_turns,
            trick_room_turns=self.trick_room_turns,
            turn=self.turn, winner=self.winner,
            turn_log=[] if self.turn_log is not None else None,
        )


# ═══════════════════════════════════════════════════════════════
#  배틀 시뮬레이터
# ═══════════════════════════════════════════════════════════════

class BattleSimulator:
    """경량 싱글 배틀 시뮬레이터."""

    def __init__(self, game_data: GameData):
        self.gd = game_data
        self.dc = DamageCalculator(game_data)

    # ─── 적법한 액션 목록 ────────────────────────────────────
    def get_legal_actions(self, state: BattleState, player: int,
                          filter_immune: bool = False) -> list[int]:
        """현재 상태에서 가능한 액션 리스트.

        filter_immune=True: 상대에게 면역(0×)인 공격기를 제거.
        MCTS 탐색용. 모든 기술이 면역이면 필터 무시.
        """
        side = state.sides[player]
        active = side.active
        actions = []

        if active.fainted:
            # 교체만 가능
            for idx, poke in side.bench:
                actions.append(ActionType.SWITCH1 + idx)
            return actions

        opp = state.sides[1 - player].active
        immune_indices: set[int] = set()

        # 면역 체크 (filter_immune 모드)
        if filter_immune and opp and not opp.fainted:
            for i, move_id in enumerate(active.moves):
                if i >= 4:
                    break
                move = self.gd.get_move(move_id)
                if move and not move["is_status"]:
                    eff = self.gd.effectiveness(move["type"], opp.types)
                    if eff == 0:
                        immune_indices.add(i)

        # 초이스 잠금 체크
        choice_item = ITEM_EFFECTS.get(active.item, {}).get("choice_lock")
        is_choice_locked = (choice_item and active.choice_locked_move
                            and not active.item_consumed)

        # 기술 사용
        for i, move_id in enumerate(active.moves):
            if i < 4:
                if active.pp[i] <= 0:
                    continue
                if is_choice_locked and move_id != active.choice_locked_move:
                    continue
                if i not in immune_indices:
                    actions.append(ActionType.MOVE1 + i)

        # 테라스탈 기술
        if not side.tera_used and active.tera_type:
            for i, move_id in enumerate(active.moves):
                if i < 4:
                    if active.pp[i] <= 0:
                        continue
                    move = self.gd.get_move(move_id)
                    if move and not move["is_status"]:
                        # 면역 필터: 기술 타입 기준 (테라는 STAB만 바꿈, 기술 타입 불변)
                        if filter_immune and opp and not opp.fainted:
                            eff = self.gd.effectiveness(move["type"], opp.types)
                            if eff == 0:
                                continue
                        actions.append(ActionType.TERA_MOVE1 + i)

        # 교체
        for idx, poke in side.bench:
            actions.append(ActionType.SWITCH1 + idx)

        # 면역 필터로 기술이 전부 제거되면 필터 없이 재시도
        has_moves = any(a <= ActionType.MOVE4 or a >= ActionType.TERA_MOVE1
                        for a in actions)
        if filter_immune and not has_moves:
            return self.get_legal_actions(state, player, filter_immune=False)

        return actions if actions else [ActionType.MOVE1]

    # ─── 턴 진행 ─────────────────────────────────────────────
    def step(self, state: BattleState,
             action_p1: int, action_p2: int) -> BattleState:
        """state × (action_p1, action_p2) → next_state."""
        s = state.clone()
        s.turn += 1
        # turn_log가 이전에 활성화되어 있었으면 새 턴도 활성화
        if state.turn_log is not None:
            s.turn_log = []

        # 양쪽 protect_active 리셋
        for p in [0, 1]:
            s.sides[p].active.protect_active = False

        # 액션 파싱
        a1 = self._parse_action(s, 0, action_p1)
        a2 = self._parse_action(s, 1, action_p2)

        pname = ["P1", "P2"]

        # 1. 교체 먼저 처리
        for player, action in [(0, a1), (1, a2)]:
            if action["type"] == "switch":
                old_name = s.sides[player].active.name
                self._do_switch(s, player, action["target"])
                new = s.sides[player].active
                s.log_event(f"{pname[player]}: {old_name} → {new.name} 교체 "
                            f"(HP {new.cur_hp}/{new.max_hp})")

        # 2. 기술 사용 순서 결정
        movers = []
        for player, action in [(0, a1), (1, a2)]:
            if action["type"] == "move":
                movers.append((player, action))

        if len(movers) == 2:
            movers = self._sort_by_speed(s, movers)

        # 3. 기술 실행
        for player, action in movers:
            if s.is_terminal:
                break
            if s.sides[player].active.fainted:
                continue
            opp = 1 - player
            atk = s.sides[player].active
            dfn = s.sides[opp].active
            atk_hp_before = atk.cur_hp
            dfn_hp_before = dfn.cur_hp
            dfn_name_before = dfn.name

            move = self.gd.get_move(action["move_id"])
            move_name = move["name"] if move else action["move_id"]
            tera_str = "Tera+" if action.get("tera") else ""

            self._execute_move(s, player, action)

            # 기술 사용 로그
            dfn_after = s.sides[opp].active
            atk_after = s.sides[player].active
            dmg_dealt = dfn_hp_before - dfn.cur_hp if dfn.name == dfn_name_before else 0
            dmg_taken = atk_hp_before - atk_after.cur_hp

            log_line = f"{pname[player]}: {atk.name} {tera_str}{move_name}"
            if dmg_dealt > 0:
                log_line += f" → {dfn_name_before}에 {dmg_dealt}dmg"
            if dfn.fainted:
                log_line += f" (기절!)"
            # 피벗 교체 감지
            if dfn_after.name != dfn_name_before and not dfn.fainted:
                pass  # 상대가 바뀐 건 상대 턴에서 기록
            if atk_after.name != atk.name:
                log_line += f" → {atk_after.name}으로 교체"
            if dmg_taken > 0 and atk_after.name == atk.name:
                log_line += f" (반동/접촉 {dmg_taken}dmg)"
            s.log_event(log_line)

            # 피벗으로 교체된 경우
            if s.sides[player].active.name != atk.name:
                new = s.sides[player].active
                s.log_event(f"  └ 피벗: {atk.name} → {new.name} "
                            f"(HP {new.cur_hp}/{new.max_hp})")

        # 4. 턴 종료 처리
        hp_before = [s.sides[p].active.cur_hp for p in [0, 1]]
        self._end_of_turn(s)
        for p in [0, 1]:
            diff = hp_before[p] - s.sides[p].active.cur_hp
            if diff > 0:
                act = s.sides[p].active
                s.log_event(f"  턴종료: {pname[p]} {act.name} "
                            f"-{diff}hp (잔여효과) → {act.cur_hp}/{act.max_hp}")
            elif diff < 0:
                act = s.sides[p].active
                s.log_event(f"  턴종료: {pname[p]} {act.name} "
                            f"+{-diff}hp (회복) → {act.cur_hp}/{act.max_hp}")

        # 5. 강제 교체 처리 (기절 시)
        for player in [0, 1]:
            if s.sides[player].active.fainted and s.sides[player].alive_count > 0:
                bench = s.sides[player].bench
                if bench:
                    idx, _ = random.choice(bench)
                    old_name = s.sides[player].active.name
                    self._do_switch(s, player, idx)
                    new = s.sides[player].active
                    s.log_event(f"{pname[player]}: {old_name} 기절 → "
                                f"{new.name} 강제교체 "
                                f"(HP {new.cur_hp}/{new.max_hp})")

        # 6. 승패 판정
        for player in [0, 1]:
            if s.sides[player].alive_count == 0:
                s.winner = 1 - player

        return s

    def _parse_action(self, state: BattleState, player: int,
                      action: int) -> dict:
        """액션 번호 → {type, move_id, move_idx, target, tera}."""
        side = state.sides[player]
        if action >= ActionType.TERA_MOVE1:
            # 테라스탈 기술
            move_idx = action - ActionType.TERA_MOVE1
            move_id = side.active.moves[move_idx] if move_idx < len(side.active.moves) else ""
            return {"type": "move", "move_id": move_id, "move_idx": move_idx, "tera": True}
        elif action >= ActionType.SWITCH1:
            target = action - ActionType.SWITCH1
            return {"type": "switch", "target": target}
        else:
            move_idx = action - ActionType.MOVE1
            move_id = side.active.moves[move_idx] if move_idx < len(side.active.moves) else ""
            return {"type": "move", "move_id": move_id, "move_idx": move_idx, "tera": False}

    def _sort_by_speed(self, state: BattleState,
                       movers: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
        """스피드/우선도로 행동 순서 결정."""
        weather = WEATHER_STR[state.weather]
        terrain = TERRAIN_STR[state.terrain]
        def priority_key(item):
            player, action = item
            move = self.gd.get_move(action["move_id"])
            priority = move["priority"] if move else 0
            speed = state.sides[player].active.effective_speed(weather, terrain)
            if state.trick_room_turns > 0:
                speed = -speed  # 트릭룸: 느린 쪽 먼저
            return (-priority, -speed)

        return sorted(movers, key=priority_key)

    # ─── 교체 처리 ───────────────────────────────────────────
    def _do_switch(self, state: BattleState, player: int, target_idx: int):
        side = state.sides[player]
        old = side.active

        # 부스트 초기화
        old.boosts = {k: 0 for k in STAT_KEYS[1:]}
        old.protect_count = 0

        side.active_idx = target_idx
        new = side.active

        # 교체 등장 효과
        # Stealth Rock
        if side.stealth_rock:
            eff = self.gd.effectiveness("Rock", new.types)
            dmg = int(new.max_hp * eff / 8)
            item_fx = ITEM_EFFECTS.get(new.item, {})
            if not item_fx.get("hazard_immune"):
                new.cur_hp = max(0, new.cur_hp - dmg)

        # Spikes
        if side.spikes > 0:
            item_fx = ITEM_EFFECTS.get(new.item, {})
            ability_fx = ABILITY_EFFECTS.get(new.ability, {})
            if not ability_fx.get("ground_immune") and not item_fx.get("hazard_immune"):
                if "Flying" not in new.types:
                    spike_dmg = [0, 1/8, 1/6, 1/4][min(side.spikes, 3)]
                    new.cur_hp = max(0, new.cur_hp - int(new.max_hp * spike_dmg))

        # Toxic Spikes
        if side.toxic_spikes > 0:
            item_fx = ITEM_EFFECTS.get(new.item, {})
            ability_fx = ABILITY_EFFECTS.get(new.ability, {})
            if not ability_fx.get("ground_immune") \
               and not item_fx.get("hazard_immune") and "Flying" not in new.types:
                if "Poison" in new.types:
                    side.toxic_spikes = 0  # 독타입이 흡수
                elif not new.status:
                    new.status = "psn" if side.toxic_spikes == 1 else "tox"

        # Sticky Web
        if side.sticky_web:
            item_fx = ITEM_EFFECTS.get(new.item, {})
            ability_fx = ABILITY_EFFECTS.get(new.ability, {})
            if not ability_fx.get("ground_immune") \
               and not item_fx.get("hazard_immune") and "Flying" not in new.types:
                new.boosts["spe"] = max(-6, new.boosts.get("spe", 0) - 1)

        # 교체 시 초이스 잠금 해제
        old.choice_locked_move = ""

        # Intimidate
        ability_fx = ABILITY_EFFECTS.get(new.ability, {})
        if ability_fx.get("on_switch_atk_drop"):
            opp = state.sides[1 - player].active
            if not opp.fainted:
                opp.boosts["atk"] = max(-6, opp.boosts.get("atk", 0) - 1)

        # Weather/Terrain setters
        if ability_fx.get("set_weather"):
            w_map = {"sun": Weather.SUN, "rain": Weather.RAIN,
                     "sand": Weather.SAND, "snow": Weather.SNOW}
            state.weather = w_map.get(ability_fx["set_weather"], Weather.NONE)
            state.weather_turns = 5
        if ability_fx.get("set_terrain"):
            t_map = {"electric": Terrain.ELECTRIC, "grassy": Terrain.GRASSY,
                     "misty": Terrain.MISTY, "psychic": Terrain.PSYCHIC}
            state.terrain = t_map.get(ability_fx["set_terrain"], Terrain.NONE)
            state.terrain_turns = 5

        # 기절 체크
        if new.cur_hp <= 0:
            new.fainted = True
            new.cur_hp = 0

        # Regenerator
        if ABILITY_EFFECTS.get(old.ability, {}).get("switch_heal"):
            if not old.fainted:
                heal = int(old.max_hp * ABILITY_EFFECTS[old.ability]["switch_heal"])
                old.cur_hp = min(old.max_hp, old.cur_hp + heal)

    # ─── 기술 실행 ───────────────────────────────────────────
    def _execute_move(self, state: BattleState, player: int, action: dict):
        side = state.sides[player]
        opp_side = state.sides[1 - player]
        attacker = side.active
        defender = opp_side.active

        move_id = action["move_id"]
        move_idx = action.get("move_idx", -1)

        # PP 차감 — PP가 0이면 Struggle로 전환
        if 0 <= move_idx < len(attacker.pp) and attacker.pp[move_idx] > 0:
            attacker.pp[move_idx] -= 1
        else:
            # Struggle: 타입 없음, 위력 50, 반동 25% max HP
            self._execute_struggle(state, player)
            return

        move = self.gd.get_move(move_id)
        if not move:
            return

        # 마비 행동불능 (25%)
        if attacker.status == "par" and random.random() < 0.25:
            return

        # 잠듦 턴 메카닉
        if attacker.status == "slp":
            attacker.sleep_turns += 1
            if attacker.sleep_turns >= random.randint(1, 3):
                attacker.status = ""
                attacker.sleep_turns = 0
            else:
                return

        # 얼음 해동 메카닉
        if attacker.status == "frz":
            if random.random() < 0.20:
                attacker.status = ""
            else:
                return

        # 방어/판별 체크
        if defender.protect_active:
            return

        # 비-protect 기술 → protect_count 리셋
        attacker.protect_count = 0

        # 테라스탈
        if action.get("tera") and not side.tera_used and attacker.tera_type:
            side.tera_used = True
            attacker.is_tera = True
            attacker.types = [attacker.tera_type]

        # 명중 판정
        accuracy = move["accuracy"]
        if accuracy is True or accuracy is None:
            pass  # 필중기
        else:
            acc = accuracy / 100.0
            # Hustle 등 정확도 보정
            ability_fx = ABILITY_EFFECTS.get(attacker.ability, {})
            if ability_fx.get("accuracy_mod"):
                acc *= ability_fx["accuracy_mod"]
            if ability_fx.get("always_hit") or ABILITY_EFFECTS.get(defender.ability, {}).get("always_hit"):
                acc = 1.0
            if random.random() > acc:
                return  # 빗나감

        # 상태이상기 간단 처리
        if move["is_status"]:
            self._handle_status_move(state, player, move)
            return

        if defender.fainted:
            return

        # 급소 확률
        crit_stage = move.get("critRatio", 1) - 1
        crit_rates = [1/24, 1/8, 1/2, 1.0]
        is_crit = random.random() < crit_rates[min(crit_stage, 3)]

        # 데미지 계산
        tera_type = attacker.tera_type if attacker.is_tera else None

        # original_types: 테라 전 원래 타입 (dex 기준)
        dex_data = self.gd.get_pokemon(attacker.species_id)
        original_types = dex_data.get("types", attacker.types) if dex_data else attacker.types

        # 테라블래스트: 테라 상태이면 기술 타입이 테라 타입으로 변경
        actual_move = move
        if move.get("id") == "terablast" and attacker.is_tera and attacker.tera_type:
            actual_move = dict(move)
            actual_move["type"] = attacker.tera_type

        atk_dict = {
            "types": attacker.types,
            "stats": attacker.stats,
            "ability": attacker.ability,
            "item": attacker.item if not attacker.item_consumed else "",
            "status": attacker.status,
            "boosts": attacker.boosts,
            "original_types": original_types,
        }
        def_dict = {
            "types": defender.types,
            "stats": defender.stats,
            "ability": defender.ability,
            "item": defender.item if not defender.item_consumed else "",
            "status": defender.status,
            "boosts": defender.boosts,
            "cur_hp": defender.cur_hp,
        }

        _, _, dmg = self.dc.calc_damage(
            atk_dict, def_dict, actual_move,
            weather=WEATHER_STR[state.weather],
            terrain=TERRAIN_STR[state.terrain],
            tera_type=tera_type,
            is_crit=is_crit,
        )

        # 리플렉터/빛의장벽 데미지 감소
        if opp_side.reflect_turns > 0 and move["is_physical"] and not is_crit:
            dmg = int(dmg * 0.5)
        if opp_side.light_screen_turns > 0 and move["is_special"] and not is_crit:
            dmg = int(dmg * 0.5)

        # 멀티히트
        multihit = move.get("multihit")
        if multihit:
            if isinstance(multihit, list):
                min_h, max_h = multihit
                item_fx = ITEM_EFFECTS.get(attacker.item, {})
                if item_fx.get("multihit_min"):
                    hits = random.randint(item_fx["multihit_min"], max_h)
                else:
                    hits = random.choices(
                        range(min_h, max_h + 1),
                        weights=[35, 35, 15, 15] if max_h - min_h == 3 else None
                    )[0]
            else:
                hits = multihit
            dmg = dmg * hits

        # Focus Sash
        def_item_fx = ITEM_EFFECTS.get(defender.item, {})
        if def_item_fx.get("sash") and defender.cur_hp >= defender.max_hp and not defender.item_consumed:
            if dmg >= defender.cur_hp:
                dmg = defender.cur_hp - 1
                defender.item_consumed = True

        # Sturdy
        def_ability_fx = ABILITY_EFFECTS.get(defender.ability, {})
        if def_ability_fx.get("sash_like") and defender.cur_hp >= defender.max_hp:
            if dmg >= defender.cur_hp:
                dmg = defender.cur_hp - 1

        # 대타출동: 대타가 데미지 흡수
        if defender.substitute_hp > 0:
            defender.substitute_hp = max(0, defender.substitute_hp - dmg)
        else:
            defender.cur_hp = max(0, defender.cur_hp - dmg)

        # Life Orb 반동
        item_fx = ITEM_EFFECTS.get(attacker.item, {})
        if item_fx.get("recoil_pct") and not attacker.item_consumed:
            ability_fx = ABILITY_EFFECTS.get(attacker.ability, {})
            if not ability_fx.get("indirect_immune"):
                recoil = int(attacker.max_hp * item_fx["recoil_pct"])
                attacker.cur_hp = max(0, attacker.cur_hp - recoil)

        # 반동기 (recoil)
        if move.get("recoil"):
            num, den = move["recoil"]
            ability_fx = ABILITY_EFFECTS.get(attacker.ability, {})
            if not ability_fx.get("indirect_immune"):
                recoil = int(dmg * num / den)
                attacker.cur_hp = max(0, attacker.cur_hp - recoil)

        # 흡수기 (drain)
        if move.get("drain"):
            num, den = move["drain"]
            heal = int(dmg * num / den)
            attacker.cur_hp = min(attacker.max_hp, attacker.cur_hp + heal)

        # 부가 효과 (간략화)
        secondary = move.get("secondary")
        if secondary and isinstance(secondary, dict):
            chance = secondary.get("chance", 100) / 100.0
            ability_fx = ABILITY_EFFECTS.get(attacker.ability, {})
            if ability_fx.get("secondary_chance_mod"):
                chance = min(1.0, chance * ability_fx["secondary_chance_mod"])
            if random.random() < chance:
                if secondary.get("boosts"):
                    for stat, val in secondary["boosts"].items():
                        if stat in defender.boosts:
                            defender.boosts[stat] = max(-6, min(6, defender.boosts[stat] + val))
                if secondary.get("status"):
                    if not defender.status:
                        defender.status = secondary["status"]

        # 울퉁불퉁멧 접촉 데미지
        def_item_fx = ITEM_EFFECTS.get(defender.item, {})
        if def_item_fx.get("contact_damage_pct") and move.get("flags", {}).get("contact"):
            if not attacker.fainted and attacker.cur_hp > 0:
                if not ABILITY_EFFECTS.get(attacker.ability, {}).get("indirect_immune"):
                    rocky_dmg = int(attacker.max_hp * def_item_fx["contact_damage_pct"])
                    attacker.cur_hp = max(0, attacker.cur_hp - rocky_dmg)

        # 약점보험
        if ITEM_EFFECTS.get(defender.item, {}).get("wp") and not defender.item_consumed:
            type_eff = self.gd.effectiveness(move["type"], defender.types)
            if type_eff > 1.0 and not defender.fainted and defender.cur_hp > 0:
                defender.boosts["atk"] = min(6, defender.boosts.get("atk", 0) + 2)
                defender.boosts["spa"] = min(6, defender.boosts.get("spa", 0) + 2)
                defender.item_consumed = True

        # 유턴/볼트체인지 (피벗)
        pivot_moves = {"uturn", "voltswitch", "flipturn", "partingshot"}
        if move_id in pivot_moves and not defender.fainted:
            bench = side.bench
            if bench:
                idx, _ = random.choice(bench)
                self._do_switch(state, player, idx)

        # 초이스 잠금
        atk_item_fx = ITEM_EFFECTS.get(attacker.item, {})
        if atk_item_fx.get("choice_lock") and not attacker.item_consumed:
            attacker.choice_locked_move = move_id

        # 기절 처리
        if defender.cur_hp <= 0:
            defender.fainted = True
            defender.cur_hp = 0
        if attacker.cur_hp <= 0:
            attacker.fainted = True
            attacker.cur_hp = 0

    def _handle_status_move(self, state: BattleState, player: int, move: dict):
        """상태이상/보조기 간단 처리."""
        side = state.sides[player]
        opp_side = state.sides[1 - player]
        attacker = side.active
        defender = opp_side.active
        move_id = _to_id(move["name"])

        # 방어/판별
        protect_moves = {"protect", "detect", "banefulbunker",
                         "kingsshield", "spikyshield", "silktrap"}
        if move_id in protect_moves:
            chance = 1.0 / (3 ** attacker.protect_count)
            if random.random() < chance:
                attacker.protect_active = True
                attacker.protect_count += 1
            else:
                attacker.protect_count = 0
            return

        # 대타출동
        if move_id == "substitute":
            cost = attacker.max_hp // 4
            if attacker.cur_hp > cost and attacker.substitute_hp == 0:
                attacker.cur_hp -= cost
                attacker.substitute_hp = cost
            return

        # 비-protect 기술 → protect_count 리셋
        attacker.protect_count = 0

        # 스탯 변화
        boosts = move.get("boosts")
        if boosts:
            target = defender if move.get("target") in ("normal", "allAdjacentFoes", "allAdjacent") else attacker
            for stat, val in boosts.items():
                if stat in target.boosts:
                    target.boosts[stat] = max(-6, min(6, target.boosts[stat] + val))

        self_boost = move.get("selfBoost", {}).get("boosts")
        if self_boost:
            for stat, val in self_boost.items():
                if stat in attacker.boosts:
                    attacker.boosts[stat] = max(-6, min(6, attacker.boosts[stat] + val))

        # 주요 상태이상기
        status = move.get("status")
        if status and not defender.status and not defender.fainted:
            # 타입 면역 체크
            if status in ("psn", "tox") and "Poison" in defender.types:
                pass
            elif status in ("psn", "tox") and "Steel" in defender.types:
                pass
            elif status == "par" and "Electric" in defender.types:
                pass
            elif status == "brn" and "Fire" in defender.types:
                pass
            else:
                defender.status = status

        # 설치기
        if move_id in ("stealthrock",):
            opp_side.stealth_rock = True
        elif move_id in ("spikes",):
            opp_side.spikes = min(3, opp_side.spikes + 1)
        elif move_id in ("toxicspikes",):
            opp_side.toxic_spikes = min(2, opp_side.toxic_spikes + 1)
        elif move_id in ("stickyweb",):
            opp_side.sticky_web = True

        # 스크린
        if move_id in ("reflect",):
            side.reflect_turns = 5
        elif move_id in ("lightscreen",):
            side.light_screen_turns = 5
        elif move_id in ("auroraveil",) and state.weather in (Weather.SNOW,):
            side.reflect_turns = 5
            side.light_screen_turns = 5

        # 날씨
        weather_moves = {
            "sunnyday": Weather.SUN, "raindance": Weather.RAIN,
            "sandstorm": Weather.SAND, "snowscape": Weather.SNOW,
        }
        if move_id in weather_moves:
            state.weather = weather_moves[move_id]
            state.weather_turns = 5

        # 필드
        terrain_moves = {
            "electricterrain": Terrain.ELECTRIC, "grassyterrain": Terrain.GRASSY,
            "mistyterrain": Terrain.MISTY, "psychicterrain": Terrain.PSYCHIC,
        }
        if move_id in terrain_moves:
            state.terrain = terrain_moves[move_id]
            state.terrain_turns = 5

        # 트릭룸
        if move_id == "trickroom":
            if state.trick_room_turns > 0:
                state.trick_room_turns = 0
            else:
                state.trick_room_turns = 5

        # Tailwind
        if move_id == "tailwind":
            side.tailwind_turns = 4

        # 회복기
        if move_id in ("recover", "roost", "softboiled", "milkdrink",
                        "slackoff", "moonlight", "morningsun", "synthesis",
                        "shoreup", "strengthsap"):
            heal = attacker.max_hp // 2
            attacker.cur_hp = min(attacker.max_hp, attacker.cur_hp + heal)

        # Defog
        if move_id == "defog":
            opp_side.stealth_rock = False
            opp_side.spikes = 0
            opp_side.toxic_spikes = 0
            opp_side.sticky_web = False
            side.stealth_rock = False
            side.spikes = 0
            side.toxic_spikes = 0
            side.sticky_web = False

        # Rapid Spin
        if move_id == "rapidspin":
            side.stealth_rock = False
            side.spikes = 0
            side.toxic_spikes = 0
            side.sticky_web = False

        # Whirlwind / Roar (상대 강제 교체)
        if move_id in ("whirlwind", "roar", "dragontail", "circlethrow"):
            bench = opp_side.bench
            if bench:
                idx, _ = random.choice(bench)
                self._do_switch(state, 1 - player, idx)

    def _execute_struggle(self, state: BattleState, player: int):
        """Struggle: 타입 없음, 위력 50, 반동 = max HP의 25%."""
        side = state.sides[player]
        opp_side = state.sides[1 - player]
        attacker = side.active
        defender = opp_side.active

        if defender.fainted or attacker.fainted:
            return

        # Struggle은 protect 관통하지 않음
        if defender.protect_active:
            return

        # 간단한 고정 데미지 (위력 50 기준 근사)
        # 실제로는 타입 없음 → 등배, STAB 없음
        atk_stat = attacker.stats["atk"]
        def_stat = defender.stats["def"]
        dmg = int(((22 * 50 * atk_stat / def_stat) / 50 + 2))
        dmg = max(1, dmg)

        if defender.substitute_hp > 0:
            defender.substitute_hp = max(0, defender.substitute_hp - dmg)
        else:
            defender.cur_hp = max(0, defender.cur_hp - dmg)

        # 반동: max HP의 25%
        recoil = max(1, attacker.max_hp // 4)
        attacker.cur_hp = max(0, attacker.cur_hp - recoil)

        # 기절 처리
        if defender.cur_hp <= 0:
            defender.fainted = True
            defender.cur_hp = 0
        if attacker.cur_hp <= 0:
            attacker.fainted = True
            attacker.cur_hp = 0

    # ─── 턴 종료 처리 ────────────────────────────────────────
    def _end_of_turn(self, state: BattleState):
        """턴 종료: 날씨 데미지, 상태이상 데미지, 잔턴 감소 등."""
        # 날씨 턴 감소
        if state.weather_turns > 0:
            state.weather_turns -= 1
            if state.weather_turns <= 0:
                state.weather = Weather.NONE

        # 필드 턴 감소
        if state.terrain_turns > 0:
            state.terrain_turns -= 1
            if state.terrain_turns <= 0:
                state.terrain = Terrain.NONE

        # 트릭룸
        if state.trick_room_turns > 0:
            state.trick_room_turns -= 1

        for player in [0, 1]:
            side = state.sides[player]
            poke = side.active
            if poke.fainted:
                continue

            ability_fx = ABILITY_EFFECTS.get(poke.ability, {})
            is_magic_guard = ability_fx.get("indirect_immune", False)

            # 날씨 데미지
            if not is_magic_guard:
                if state.weather == Weather.SAND:
                    if not any(t in ("Rock", "Ground", "Steel") for t in poke.types):
                        poke.cur_hp = max(0, poke.cur_hp - poke.max_hp // 16)

            # 상태이상 데미지
            if not is_magic_guard:
                if poke.status == "brn":
                    poke.cur_hp = max(0, poke.cur_hp - poke.max_hp // 16)
                elif poke.status == "psn":
                    poke.cur_hp = max(0, poke.cur_hp - poke.max_hp // 8)
                elif poke.status == "tox":
                    poke.tox_counter += 1
                    dmg = poke.max_hp * poke.tox_counter // 16
                    poke.cur_hp = max(0, poke.cur_hp - dmg)

            # Leftovers / Black Sludge
            item_fx = ITEM_EFFECTS.get(poke.item, {})
            if item_fx.get("end_turn_heal_pct") and not poke.item_consumed:
                heal = int(poke.max_hp * item_fx["end_turn_heal_pct"])
                poke.cur_hp = min(poke.max_hp, poke.cur_hp + heal)

            # Grassy Terrain 회복
            if state.terrain == Terrain.GRASSY:
                poke.cur_hp = min(poke.max_hp, poke.cur_hp + poke.max_hp // 16)

            # 스크린 턴 감소
            if side.reflect_turns > 0:
                side.reflect_turns -= 1
            if side.light_screen_turns > 0:
                side.light_screen_turns -= 1
            if side.tailwind_turns > 0:
                side.tailwind_turns -= 1

            # 기절 체크
            if poke.cur_hp <= 0:
                poke.fainted = True
                poke.cur_hp = 0

    # ─── 팀 프리뷰에서 포켓몬 선택 후 배틀 상태 생성 ────────
    def create_battle_state(
        self,
        team1: list[Pokemon],
        team2: list[Pokemon],
    ) -> BattleState:
        """두 팀으로 초기 배틀 상태 생성."""
        return BattleState(
            sides=[
                Side(team=team1, active_idx=0),
                Side(team=team2, active_idx=0),
            ]
        )

    # ─── 스마트 롤아웃 (MCTS용) ────────────────────────────────
    def _smart_pick(self, state: BattleState, player: int) -> int:
        """면역/비효과 기술을 피하는 스마트 랜덤 선택."""
        actions = self.get_legal_actions(state, player)
        if not actions:
            return 0

        side = state.sides[player]
        opp = state.sides[1 - player].active
        active = side.active

        # 가중치 기반 선택: 면역기 0, 반감 0.3, 등배 1, 효과 2
        weights = []
        for a in actions:
            w = 1.0
            move_id = None
            if a <= ActionType.MOVE4:
                idx = a - ActionType.MOVE1
                move_id = active.moves[idx] if idx < len(active.moves) else None
            elif a >= ActionType.TERA_MOVE1:
                idx = a - ActionType.TERA_MOVE1
                move_id = active.moves[idx] if idx < len(active.moves) else None

            if move_id and not opp.fainted:
                move_data = self.gd.get_move(move_id)
                if move_data and not move_data["is_status"]:
                    eff = self.gd.effectiveness(move_data["type"], opp.types)
                    if eff == 0:
                        w = 0.0
                    elif eff <= 0.25:
                        w = 0.1
                    elif eff == 0.5:
                        w = 0.3
                    elif eff >= 2.0:
                        w = 2.0
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            return random.choice(actions)
        return random.choices(actions, weights=weights, k=1)[0]

    def rollout(self, state: BattleState, max_turns: int = 50) -> int:
        """스마트 랜덤 롤아웃. 면역기를 피하면서 게임을 끝까지 진행."""
        s = state.clone()
        for _ in range(max_turns):
            if s.is_terminal:
                return s.winner

            a1 = self._smart_pick(s, 0)
            a2 = self._smart_pick(s, 1)
            s = self.step(s, a1, a2)

        # 턴 초과 → HP 비율로 판정
        hp1 = sum(p.cur_hp for p in s.sides[0].team)
        hp2 = sum(p.cur_hp for p in s.sides[1].team)
        return 0 if hp1 >= hp2 else 1

    # ─── 배치 롤아웃 (GPU 병렬은 아니지만, 여러 게임 동시) ──
    def batch_rollout(self, state: BattleState,
                      n_rollouts: int = 256, max_turns: int = 50) -> float:
        """n_rollouts 회 랜덤 롤아웃 → P1 승률 반환."""
        wins = sum(self.rollout(state, max_turns) == 0
                   for _ in range(n_rollouts))
        return wins / n_rollouts


# ═══════════════════════════════════════════════════════════════
#  팀 생성 헬퍼 (stats 기반)
# ═══════════════════════════════════════════════════════════════

def make_pokemon(
    gd: GameData,
    species: str,
    moves: list[str],
    ability: str = "",
    item: str = "",
    nature: str = "Adamant",
    evs: dict | None = None,
    tera_type: str = "",
) -> Pokemon:
    """간편하게 포켓몬 객체 생성."""
    dex = gd.get_pokemon(species)
    if not dex:
        raise ValueError(f"Unknown pokemon: {species}")

    dc = DamageCalculator(gd)
    if evs is None:
        evs = {k: 0 for k in STAT_KEYS}
    stats = dc.calc_stats_from_spread(dex["baseStats"], nature, evs)

    if not ability:
        ability = _to_id(dex["abilities"][0]) if dex["abilities"] else ""
    else:
        ability = _to_id(ability)

    move_ids = [_to_id(m) for m in moves]
    pp_list = []
    for mid in move_ids:
        move_data = gd.get_move(mid)
        pp_list.append(move_data["pp"] if move_data else 5)

    return Pokemon(
        species_id=_to_id(species),
        name=dex["name"],
        types=dex["types"],
        base_stats=dex["baseStats"],
        stats=stats,
        ability=ability,
        item=_to_id(item),
        moves=move_ids,
        tera_type=tera_type,
        pp=pp_list,
        max_pp=list(pp_list),
    )


def make_pokemon_from_set_dict(
    gd: GameData,
    species: str,
    set_dict: dict,
    observed_pokemon: Pokemon | None = None,
) -> Pokemon:
    """세트 사전(SET_DB 엔트리) → Pokemon. 관측된 상태 보존.

    Args:
        gd: GameData
        species: 포켓몬 종 이름/ID
        set_dict: {"name","moves","item","ability","nature","evs","tera_type",...}
        observed_pokemon: 관측된 런타임 상태(HP, 상태이상, 부스트 등) 보존용.
                          None이면 풀 HP로 생성.

    Returns:
        Pokemon 객체 (관측된 HP 비율, 상태이상, 부스트 등 보존)
    """
    dex = gd.get_pokemon(species)
    if not dex:
        raise ValueError(f"Unknown pokemon: {species}")

    dc = DamageCalculator(gd)
    nature = set_dict.get("nature", "Hardy")
    evs = set_dict.get("evs", {k: 0 for k in STAT_KEYS})
    stats = dc.calc_stats_from_spread(dex["baseStats"], nature, evs)

    # 특성
    ability = set_dict.get("ability", "")
    if not ability:
        ability = _to_id(dex["abilities"][0]) if dex["abilities"] else ""
    else:
        ability = _to_id(ability)

    # 기술: 관측된 것 유지 + set_dict에서 보충
    if observed_pokemon is not None:
        observed_moves = [m for m in observed_pokemon.moves if m]
        set_moves = [_to_id(m) for m in set_dict.get("moves", [])]
        moves = list(observed_moves)
        for mv in set_moves:
            if mv not in moves and len(moves) < 4:
                moves.append(mv)
    else:
        moves = [_to_id(m) for m in set_dict.get("moves", [])]

    moves = moves[:4]

    # PP
    pp_list = []
    for mid in moves:
        move_data = gd.get_move(mid)
        pp_list.append(move_data["pp"] if move_data else 5)

    # 아이템: 관측된 것 우선
    if observed_pokemon is not None and observed_pokemon.item:
        item = observed_pokemon.item
    else:
        item = _to_id(set_dict.get("item", ""))

    # 테라 타입: 관측된 것 우선
    if observed_pokemon is not None and observed_pokemon.tera_type:
        tera_type = observed_pokemon.tera_type
    else:
        tera_type = set_dict.get("tera_type", "")

    # 타입
    types = list(dex["types"])

    # 기본 Pokemon 생성
    poke = Pokemon(
        species_id=_to_id(species),
        name=dex["name"],
        types=types,
        base_stats=dex["baseStats"],
        stats=stats,
        ability=ability,
        item=item,
        moves=moves,
        tera_type=tera_type,
        pp=pp_list,
        max_pp=list(pp_list),
    )

    # 관측된 런타임 상태 보존
    if observed_pokemon is not None:
        # HP 비율 보존: 관측된 HP% × 새 max_hp
        hp_pct = observed_pokemon.hp_pct
        poke.max_hp = poke.stats["hp"]
        poke.cur_hp = max(1, int(hp_pct * poke.max_hp))

        poke.status = observed_pokemon.status
        poke.tox_counter = observed_pokemon.tox_counter
        poke.boosts = dict(observed_pokemon.boosts)
        poke.is_tera = observed_pokemon.is_tera
        poke.fainted = observed_pokemon.fainted
        poke.has_moved = observed_pokemon.has_moved
        poke.protect_count = observed_pokemon.protect_count
        poke.protect_active = observed_pokemon.protect_active
        poke.sleep_turns = observed_pokemon.sleep_turns
        poke.choice_locked_move = observed_pokemon.choice_locked_move
        poke.substitute_hp = observed_pokemon.substitute_hp
        poke.item_consumed = observed_pokemon.item_consumed

        # 이미 테라한 경우 타입 갱신
        if poke.is_tera and poke.tera_type:
            poke.types = [poke.tera_type]

        # 기절 상태 처리
        if poke.fainted:
            poke.cur_hp = 0

    return poke


def make_pokemon_from_stats(
    gd: GameData,
    species: str,
    format_name: str = "bss",
) -> Pokemon:
    """Smogon 통계에서 가장 인기 있는 세트로 포켓몬 생성."""
    usage = gd.get_stats(species, format_name)
    if not usage:
        raise ValueError(f"No stats for {species} in {format_name}")

    dex = gd.get_pokemon(species)
    if not dex:
        raise ValueError(f"Unknown pokemon: {species}")

    # 최인기 아이템
    items = sorted(usage["items"].items(), key=lambda x: -x[1])
    item = items[0][0] if items else ""

    # 최인기 기술 4개
    moves_sorted = sorted(usage["moves"].items(), key=lambda x: -x[1])
    top_moves = [m[0] for m in moves_sorted[:4]]

    # 최인기 스프레드
    spreads = usage.get("spreads", [])
    if spreads:
        best = max(spreads, key=lambda x: x["weight"])
        nature = best["nature"]
        evs = best["evs"]
    else:
        nature = "Adamant"
        evs = {k: 0 for k in STAT_KEYS}

    # 최인기 특성
    abilities = sorted(usage["abilities"].items(), key=lambda x: -x[1])
    ability = abilities[0][0] if abilities else ""

    dc = DamageCalculator(gd)
    stats = dc.calc_stats_from_spread(dex["baseStats"], nature, evs)

    pp_list = []
    for mid in top_moves:
        move_data = gd.get_move(mid)
        pp_list.append(move_data["pp"] if move_data else 5)

    return Pokemon(
        species_id=_to_id(species),
        name=dex["name"],
        types=dex["types"],
        base_stats=dex["baseStats"],
        stats=stats,
        ability=ability,
        item=item,
        moves=top_moves,
        tera_type=dex["types"][0],  # 기본: 첫 번째 타입
        pp=pp_list,
        max_pp=list(pp_list),
    )


# ═══════════════════════════════════════════════════════════════
#  Showdown Paste 파서
# ═══════════════════════════════════════════════════════════════

import re as _re


def parse_showdown_paste(gd: GameData, paste: str) -> list[Pokemon]:
    """Showdown paste 텍스트 → Pokemon 객체 리스트."""
    pokemon_list = []
    # 빈 줄로 포켓몬 구분
    blocks = _re.split(r"\n\n+", paste.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")

        species = ""
        item = ""
        ability = ""
        tera_type = ""
        nature = "Adamant"
        evs: dict[str, int] = {k: 0 for k in STAT_KEYS}
        moves: list[str] = []

        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:
                # 첫 줄: "Name @ Item" 또는 "Name (F) @ Item"
                # 성별 표기 제거: (M), (F)
                cleaned = _re.sub(r"\s*\([MF]\)\s*", " ", line).strip()
                if " @ " in cleaned:
                    species, item = cleaned.split(" @ ", 1)
                else:
                    species = cleaned
                species = species.strip()
                item = item.strip()
            elif line.startswith("Ability:"):
                ability = line.split(":", 1)[1].strip()
            elif line.startswith("Tera Type:"):
                tera_type = line.split(":", 1)[1].strip()
            elif line.endswith("Nature"):
                nature = line.replace("Nature", "").strip()
            elif line.startswith("EVs:"):
                ev_str = line.split(":", 1)[1].strip()
                for part in ev_str.split("/"):
                    part = part.strip()
                    m = _re.match(r"(\d+)\s+(\w+)", part)
                    if m:
                        val = int(m.group(1))
                        stat_name = m.group(2)
                        stat_map = {
                            "HP": "hp", "Atk": "atk", "Def": "def",
                            "SpA": "spa", "SpD": "spd", "Spe": "spe",
                        }
                        key = stat_map.get(stat_name, "")
                        if key:
                            evs[key] = val
            elif line.startswith("- "):
                moves.append(line[2:].strip())
            # IVs: 무시 (인코딩에 영향 없음)

        if species and moves:
            pokemon_list.append(make_pokemon(
                gd, species, moves,
                ability=ability, item=item,
                nature=nature, evs=evs,
                tera_type=tera_type,
            ))

    return pokemon_list


def load_sample_teams(gd: GameData, path: str) -> list[list[Pokemon]]:
    """sample_teams.txt → [[6 Pokemon], [6 Pokemon], ...] 리스트."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # "=== Team N ===" 구분자로 팀 분리
    team_blocks = _re.split(r"^===.*===\s*$", content, flags=_re.MULTILINE)
    teams = []
    for block in team_blocks:
        block = block.strip()
        if not block:
            continue
        team = parse_showdown_paste(gd, block)
        if team:
            teams.append(team)
    return teams


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """1v1 시뮬레이션 검증."""
    import time
    print("=== Battle Simulator 검증 ===\n")

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)

    # 코라이돈 vs 딩루
    koraidon = make_pokemon(gd, "Koraidon",
        moves=["Close Combat", "Flare Blitz", "Dragon Claw", "Swords Dance"],
        ability="Orichalcum Pulse", item="Life Orb",
        nature="Adamant", evs={"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
        tera_type="Fighting")

    tinglu = make_pokemon(gd, "Ting-Lu",
        moves=["Earthquake", "Ruination", "Whirlwind", "Stealth Rock"],
        ability="Vessel of Ruin", item="Leftovers",
        nature="Impish", evs={"hp": 244, "atk": 4, "def": 116, "spa": 0, "spd": 132, "spe": 12},
        tera_type="Poison")

    state = sim.create_battle_state([koraidon], [tinglu])
    print(f"코라이돈 HP: {koraidon.cur_hp}/{koraidon.max_hp}")
    print(f"딩루 HP: {tinglu.cur_hp}/{tinglu.max_hp}")

    # 코라이돈 인파이트 vs 딩루 지진
    actions = sim.get_legal_actions(state, 0)
    print(f"코라이돈 가능 액션: {actions}")

    s1 = sim.step(state, ActionType.MOVE1, ActionType.MOVE1)  # Close Combat vs Earthquake
    p1 = s1.sides[0].active
    p2 = s1.sides[1].active
    print(f"\n턴 1 후:")
    print(f"  코라이돈 HP: {p1.cur_hp}/{p1.max_hp}")
    print(f"  딩루 HP: {p2.cur_hp}/{p2.max_hp}")
    print(f"  승자: {'없음' if not s1.is_terminal else f'P{s1.winner+1}'}")

    # 3마리 팀 롤아웃 테스트
    print("\n--- 3v3 롤아웃 테스트 ---")
    team1 = [
        make_pokemon_from_stats(gd, "Koraidon", "bss"),
        make_pokemon_from_stats(gd, "Ting-Lu", "bss"),
        make_pokemon_from_stats(gd, "Flutter Mane", "bss"),
    ]
    team2 = [
        make_pokemon_from_stats(gd, "Miraidon", "bss"),
        make_pokemon_from_stats(gd, "Gholdengo", "bss"),
        make_pokemon_from_stats(gd, "Great Tusk", "bss"),
    ]

    state = sim.create_battle_state(team1, team2)

    start = time.time()
    n_rollouts = 200
    winrate = sim.batch_rollout(state, n_rollouts, max_turns=50)
    elapsed = time.time() - start

    print(f"P1 팀: {[p.name for p in team1]}")
    print(f"P2 팀: {[p.name for p in team2]}")
    print(f"{n_rollouts}회 롤아웃 → P1 승률: {winrate:.1%}")
    print(f"소요 시간: {elapsed:.2f}s ({n_rollouts/elapsed:.0f} 롤아웃/초)")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
