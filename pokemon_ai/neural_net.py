"""AlphaZero Neural Network — State 인코더 + ResNet(Policy+Value head).

State 인코딩 (1614차원):
  아군 포켓몬 ×3 (792) + 상대 포켓몬 ×3 (792) + 사이드 ×2 (16) + 글로벌 (14)
  per-Pokemon = 264 dim:
    기본(63) + 기술(4×42=168) + 아이템효과(12) + 특성효과(16) + 매치업(5)

  per-Move = 42 dim:
    type(18) + power(1) + category(3) + priority(1) + eff(1) = 24 (기존)
    + accuracy(1) + burn/para/sleep/freeze/flinch(5) + recoil(1) + drain(1)
      + self_boost(1) + target_debuff(1) + self_switch(1) + force_switch(1)
      + sets_hazard(1) + heals(1) + is_contact(1)                    = 15 (기술스펙)
    + damage_pct(1) + stab(1) + can_ohko(1)                          = 3 (지식계산)

네트워크 (~2.8M params):
  Input(1614) → Linear(1614→512) → 6× ResBlock(512) → LayerNorm
    ├─ Policy: Linear(512→256) → ReLU → Linear(256→13) → [마스킹 후 softmax]
    │    + Move Shortcut: per-move(42d) → MLP(42→32→1) → action 0-3, 4-7 bias
    └─ Value:  Linear(512→256) → ReLU → Linear(256→1)  → tanh [-1,+1]

TeamPreviewNet (~300K params):
  프리뷰 인코딩 (804차원) → 3× ResBlock(256) → Policy(20 combos) + Value
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_loader import (
    GameData, TYPE_TO_IDX, STATUS_TO_IDX, STAT_KEYS, TYPES, _to_id,
    ITEM_EFFECTS, _TYPE_BOOST_ITEMS, ABILITY_EFFECTS,
)
from battle_sim import BattleState, ActionType, NUM_ACTIONS, Weather, Terrain, Pokemon, make_pokemon_from_stats, WEATHER_STR, TERRAIN_STR


# ═══════════════════════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════════════════════

NUM_TYPES = len(TYPES)            # 18
NUM_STATUS = len(STATUS_TO_IDX)   # 7 (none, brn, par, psn, tox, slp, frz)
MOVES_PER_POKEMON = 4

# ── per-Move 피처 차원 ────────────────────────────────────────
# 기존 24: type(18) + power(1) + category(3) + priority(1) + effectiveness(1)
# 추가 15: accuracy(1) + burn/para/sleep/freeze/flinch(5) + recoil(1) + drain(1)
#          + self_boost(1) + target_debuff(1) + self_switch(1) + force_switch(1)
#          + sets_hazard(1) + heals(1) + is_contact(1)
# 추가  3: damage_pct(1) + stab(1) + can_ohko(1)
MOVE_BASE_DIM = 24     # 기존 (type/power/category/priority/eff)
MOVE_EFFECTS_DIM = 15  # 기술 스펙 (accuracy, 상태이상, recoil 등)
MOVE_KNOWLEDGE_DIM = 3 # 게임 지식 계산 (damage_pct, stab, can_ohko)
MOVE_DIM = MOVE_BASE_DIM + MOVE_EFFECTS_DIM + MOVE_KNOWLEDGE_DIM  # 42

# ── 아이템 효과 인코딩 ────────────────────────────────────────
N_ITEM_FX = 12   # offensive_mod, speed_mod, choice_lock, def_mod, spd_mod, sash,
                  # heal_pct, hazard_immune, recoil_pct, contact_punish,
                  # prevent_secondary, cure_status

# ── 특성 효과 인코딩 ──────────────────────────────────────────
N_ABILITY_FX = 16  # atk_mod, full_hp_shield, super_eff_reduce, foe_def_mod,
                    # foe_spd_mod, foe_atk_mod, foe_spa_mod, intimidate,
                    # ground_immune, fire_immune, speed_boost, switch_heal,
                    # mold_breaker, ignore_boosts, special_shield, sets_weather_terrain

# ── 상태이상 효과 ─────────────────────────────────────────────
N_STATUS_FX = 5   # phys_atk_penalty, speed_penalty, cant_move_chance, dot_damage_pct, turns_remaining

# ── 매치업 피처 ───────────────────────────────────────────────
MATCHUP_DIM = 5   # speed_advantage, speed_ratio, max_incoming_pct, can_be_ohko, defensive_eff

# ── 포켓몬/상태 차원 ──────────────────────────────────────────
# 기본 63: HP(1) + faint(1) + stats(6) + types(18) + status_onehot(7) + status_effects(5)
#          + boosts(5) + tera(18) + is_tera(1) + tera_avail(1)
POKEMON_BASE_DIM = 63
POKEMON_DIM = POKEMON_BASE_DIM + MOVES_PER_POKEMON * MOVE_DIM + N_ITEM_FX + N_ABILITY_FX + MATCHUP_DIM
# = 63 + 4*42 + 12 + 16 + 5 = 264

TEAM_SIZE = 3
SIDE_DIM = 8        # SR, spikes, toxic_spikes, sticky_web, reflect, light_screen, tailwind, tera_used
GLOBAL_DIM = 14     # weather(5) + weather_turns(1) + terrain(5) + terrain_turns(1) + trick_room(1) + turn(1)
STATE_DIM = POKEMON_DIM * TEAM_SIZE * 2 + SIDE_DIM * 2 + GLOBAL_DIM  # 1614

# ── Showdown 직접 인코딩용 상수 ───────────────────────────────
# Showdown move: 42 + pp_frac(1) = 43
SHOWDOWN_MOVE_DIM = MOVE_DIM + 1  # 43
# Showdown pokemon: 기본63 + boosts_extra(2: accuracy/evasion) + moves(4×43) + item(12) + ability(16) + matchup(5) = 270
SHOWDOWN_POKEMON_DIM = (POKEMON_BASE_DIM + 2 + MOVES_PER_POKEMON * SHOWDOWN_MOVE_DIM
                        + N_ITEM_FX + N_ABILITY_FX + MATCHUP_DIM)  # 270
SHOWDOWN_STATE_DIM = SHOWDOWN_POKEMON_DIM * TEAM_SIZE * 2 + SIDE_DIM * 2 + GLOBAL_DIM  # 1650

# ── (레거시 호환용: showdown_mcts.py 가 import하는 이름들) ──────
N_ITEMS = N_ITEM_FX
N_ABILITIES = N_ABILITY_FX
ITEM_TO_IDX: dict[str, int] = {}   # 더 이상 one-hot 사용 안 함, 호환용 빈 딕트
ABILITY_TO_IDX: dict[str, int] = {}


# ═══════════════════════════════════════════════════════════════
#  헬퍼 함수: 기술 효과 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_move_effects(move_data: dict) -> np.ndarray:
    """moves.json 필드 → (15,) 기술 스펙 벡터.

    [accuracy, burn_chance, para_chance, sleep_chance, freeze_chance,
     flinch_chance, recoil_pct, drain_pct, self_boost_total,
     target_debuff_expected, self_switch, force_switch, sets_hazard,
     heals, is_contact]
    """
    vec = np.zeros(MOVE_EFFECTS_DIM, dtype=np.float32)

    # 0: accuracy (True → 1.0, int → /100)
    acc = move_data.get("accuracy", 100)
    vec[0] = 1.0 if acc is True else (acc / 100.0 if isinstance(acc, (int, float)) else 1.0)

    # 1-5: 상태이상/풀죽음 확률
    # 직접 상태이상 (status 필드, 100% 확률)
    direct_status = move_data.get("status", "")
    if direct_status == "brn":
        vec[1] = 1.0
    elif direct_status == "par":
        vec[2] = 1.0
    elif direct_status == "slp":
        vec[3] = 1.0

    # secondary 상태이상 (확률부)
    secondary = move_data.get("secondary")
    if secondary and isinstance(secondary, dict):
        chance = secondary.get("chance", 100) / 100.0
        sec_status = secondary.get("status", "")
        if sec_status == "brn":
            vec[1] = max(vec[1], chance)
        elif sec_status == "par":
            vec[2] = max(vec[2], chance)
        elif sec_status == "slp":
            vec[3] = max(vec[3], chance)
        elif sec_status == "frz":
            vec[4] = chance
        # volatileStatus
        vol_status = secondary.get("volatileStatus", "")
        if vol_status == "flinch":
            vec[5] = chance
        # secondary boosts (target 디버프)
        sec_boosts = secondary.get("boosts")
        if sec_boosts and isinstance(sec_boosts, dict):
            debuff_sum = sum(sec_boosts.values())
            vec[9] = debuff_sum * chance / 6.0  # target_debuff_expected

    # 6: recoil_pct
    recoil = move_data.get("recoil")
    if recoil and isinstance(recoil, (list, tuple)) and len(recoil) == 2:
        vec[6] = recoil[0] / max(recoil[1], 1)

    # 7: drain_pct
    drain = move_data.get("drain")
    if drain and isinstance(drain, (list, tuple)) and len(drain) == 2:
        vec[7] = drain[0] / max(drain[1], 1)

    # 8: self_boost_total
    # 변화기 boosts (e.g. 칼춤 {'atk': 2})
    boosts = move_data.get("boosts")
    if boosts and isinstance(boosts, dict) and move_data.get("target") == "self":
        vec[8] = sum(boosts.values()) / 6.0
    # self.boosts (e.g. 인파 self: {'boosts': {'def': -1, 'spd': -1}})
    self_data = move_data.get("self")
    if self_data and isinstance(self_data, dict):
        self_boosts = self_data.get("boosts")
        if self_boosts and isinstance(self_boosts, dict):
            vec[8] += sum(self_boosts.values()) / 6.0

    # 10: self_switch (U-turn etc.)
    vec[10] = float(bool(move_data.get("selfSwitch")))

    # 11: force_switch (Roar etc.)
    vec[11] = float(bool(move_data.get("forceSwitch")))

    # 12: sets_hazard (Stealth Rock etc.)
    vec[12] = float(bool(move_data.get("sideCondition")))

    # 13: heals (Recover: [1,2] → 0.5)
    heal = move_data.get("heal")
    if heal and isinstance(heal, (list, tuple)) and len(heal) == 2:
        vec[13] = heal[0] / max(heal[1], 1)

    # 14: is_contact
    flags = move_data.get("flags", {})
    vec[14] = float(bool(flags.get("contact")))

    return vec


# ═══════════════════════════════════════════════════════════════
#  헬퍼 함수: 아이템 효과 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_item_effects(item_id: str) -> np.ndarray:
    """ITEM_EFFECTS / _TYPE_BOOST_ITEMS → (12,) 효과 벡터.

    [offensive_mod, speed_mod, choice_lock, def_mod, spd_mod, sash,
     heal_pct, hazard_immune, recoil_pct, contact_punish,
     prevent_secondary, cure_status]
    """
    vec = np.zeros(N_ITEM_FX, dtype=np.float32)
    if not item_id:
        return vec

    fx = ITEM_EFFECTS.get(item_id, {})

    # 0: offensive_mod (CB 1.5→0.5, LO 1.3→0.3)
    vec[0] = fx.get("damage_mod", 1.0) - 1.0
    # 타입 강화 아이템
    if item_id in _TYPE_BOOST_ITEMS:
        _, boost = _TYPE_BOOST_ITEMS[item_id]
        vec[0] = max(vec[0], boost - 1.0)

    # 1: speed_mod (스카프 1.5→0.5)
    vec[1] = fx.get("speed_mod", 1.0) - 1.0

    # 2: choice_lock
    vec[2] = float(bool(fx.get("choice_lock")))

    # 3: def_mod (진화의휘석 1.5→0.5)
    vec[3] = fx.get("def_mod", 1.0) - 1.0

    # 4: spd_mod (돌격조끼 1.5→0.5)
    vec[4] = fx.get("spd_mod", 1.0) - 1.0

    # 5: sash
    vec[5] = float(bool(fx.get("sash")))

    # 6: heal_pct (먹밥 1/16 ≈ 0.0625)
    vec[6] = fx.get("end_turn_heal_pct", 0.0)

    # 7: hazard_immune (부츠)
    vec[7] = float(bool(fx.get("hazard_immune")))

    # 8: recoil_pct (LO 0.1)
    vec[8] = fx.get("recoil_pct", 0.0)

    # 9: contact_punish (울퉁불퉁멧 1/6)
    vec[9] = fx.get("contact_damage_pct", 0.0)

    # 10: prevent_secondary
    vec[10] = float(bool(fx.get("prevent_secondary")))

    # 11: cure_status
    vec[11] = float(bool(fx.get("cure_status")))

    return vec


# ═══════════════════════════════════════════════════════════════
#  헬퍼 함수: 특성 효과 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_ability_effects(ability_id: str) -> np.ndarray:
    """ABILITY_EFFECTS → (16,) 효과 벡터.

    [atk_mod, full_hp_shield, super_eff_reduce, foe_def_mod,
     foe_spd_mod, foe_atk_mod, foe_spa_mod, intimidate,
     ground_immune, fire_immune, speed_boost, switch_heal,
     mold_breaker, ignore_boosts, special_shield, sets_weather_terrain]
    """
    vec = np.zeros(N_ABILITY_FX, dtype=np.float32)
    if not ability_id:
        return vec

    fx = ABILITY_EFFECTS.get(ability_id, {})

    # 0: atk_mod (힘자랑 2.0→1.0)
    vec[0] = fx.get("atk_mod", 1.0) - 1.0

    # 1: full_hp_shield (멀티스케일 0.5→0.5)
    vec[1] = 1.0 - fx.get("full_hp_damage_mod", 1.0)

    # 2: super_eff_reduce (필터 0.75→0.25)
    vec[2] = 1.0 - fx.get("super_eff_mod", 1.0)

    # 3: foe_def_mod (재앙의검 0.75→0.25)
    vec[3] = 1.0 - fx.get("foe_def_mod", 1.0)

    # 4: foe_spd_mod (재앙의구슬그릇 0.75→0.25)
    vec[4] = 1.0 - fx.get("foe_spd_mod", 1.0)

    # 5: foe_atk_mod (재앙의서간 0.75→0.25)
    vec[5] = 1.0 - fx.get("foe_atk_mod", 1.0)

    # 6: foe_spa_mod (재앙의목간 0.75→0.25)
    vec[6] = 1.0 - fx.get("foe_spa_mod", 1.0)

    # 7: intimidate
    vec[7] = float(bool(fx.get("on_switch_atk_drop")))

    # 8: ground_immune (부유)
    vec[8] = float(bool(fx.get("ground_immune")))

    # 9: fire_immune (타오르는불꽃)
    vec[9] = float(bool(fx.get("fire_immune")))

    # 10: speed_boost (쓱쓱, 엽록소 등 — speed_mul - 1.0)
    vec[10] = fx.get("speed_mul", 1.0) - 1.0

    # 11: switch_heal (재생력 0.33)
    vec[11] = fx.get("switch_heal", 0.0)

    # 12: mold_breaker
    vec[12] = float(bool(fx.get("mold_breaker")))

    # 13: ignore_boosts (천진)
    vec[13] = float(bool(fx.get("ignore_boosts")))

    # 14: special_shield (아이스페이스 0.5→0.5)
    vec[14] = 1.0 - fx.get("special_damage_mod", 1.0)

    # 15: sets_weather_terrain
    vec[15] = float(bool(fx.get("set_weather") or fx.get("set_terrain")))

    return vec


# ═══════════════════════════════════════════════════════════════
#  헬퍼 함수: 상태이상 효과 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_status_effects(status: str, sleep_turns: int = 0,
                           tox_counter: int = 0) -> np.ndarray:
    """상태이상 → (5,) 효과 벡터.

    [phys_atk_penalty, speed_penalty, cant_move_chance, dot_damage_pct, turns_remaining]
    """
    vec = np.zeros(N_STATUS_FX, dtype=np.float32)
    if not status:
        return vec

    # 0: phys_atk_penalty (화상 = 물공 절반)
    if status == "brn":
        vec[0] = 0.5

    # 1: speed_penalty (마비 = 스피드 절반)
    if status == "par":
        vec[1] = 0.5

    # 2: cant_move_chance
    if status == "par":
        vec[2] = 0.25
    elif status == "slp":
        vec[2] = 1.0   # 깨어나기 전까지 100% 행동불능
    elif status == "frz":
        vec[2] = 0.80  # 20% 확률로 풀림

    # 3: dot_damage_pct
    if status == "psn":
        vec[3] = 0.125
    elif status == "brn":
        vec[3] = 0.0625
    elif status == "tox":
        vec[3] = min(tox_counter + 1, 15) / 16.0

    # 4: turns_remaining (수면)
    if status == "slp":
        vec[4] = max(0, 3 - sleep_turns) / 3.0

    return vec


# ═══════════════════════════════════════════════════════════════
#  헬퍼 함수: 경량 데미지 계산
# ═══════════════════════════════════════════════════════════════

def _calc_damage_fast(atk_stat: float, def_stat: float, power: float,
                      stab: float, type_eff: float,
                      weather_mod: float = 1.0,
                      is_burned: bool = False) -> float:
    """경량 데미지 추정 (Gen 9 공식 근사).

    Returns: 예상 데미지 (raw HP 단위)
    """
    if power <= 0 or type_eff <= 0:
        return 0.0
    # Level 50 damage formula: ((2*50/5+2) * power * atk/def) / 50 + 2
    base = ((22.0 * power * atk_stat / max(def_stat, 1)) / 50.0 + 2.0)
    dmg = base * stab * type_eff * weather_mod
    if is_burned:
        dmg *= 0.5
    # 평균 랜덤 보정 (0.85~1.0 → ~0.925)
    dmg *= 0.925
    return dmg


def _weather_mod_for_type(move_type: str, weather_str: str) -> float:
    """날씨에 따른 타입 보정."""
    if weather_str == "sun":
        if move_type == "Fire":
            return 1.5
        elif move_type == "Water":
            return 0.5
    elif weather_str == "rain":
        if move_type == "Water":
            return 1.5
        elif move_type == "Fire":
            return 0.5
    return 1.0

# ── TeamPreview 상수 ──────────────────────────────────────────
PREVIEW_POKEMON_DIM = 60   # type(18) + stats(6) + tera_type(18) + move_coverage(18)
PREVIEW_CROSS_DIM = 72     # 6×6 ally→opp + 6×6 opp→ally
PREVIEW_GLOBAL_DIM = 12
PREVIEW_TEAM_SIZE = 6
PREVIEW_STATE_DIM = (PREVIEW_POKEMON_DIM * PREVIEW_TEAM_SIZE * 2
                     + PREVIEW_CROSS_DIM + PREVIEW_GLOBAL_DIM)  # 804
NUM_COMBOS = 20            # C(6,3)
COMBO_TABLE = list(itertools.combinations(range(6), 3))  # 고정 순서

# ── TeamBuild 상수 ──────────────────────────────────────────
N_CANDIDATES = 50       # 후보 풀 크기 (Smogon Top-50)
BUILD_TEAM_SIZE = 6     # 빌드할 팀 크기
BUILD_STATE_DIM = 430   # 360 + 62 + 8


# ═══════════════════════════════════════════════════════════════
#  State 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_pokemon(poke, opp_active, gd: GameData,
                    tera_available: bool,
                    weather: str = "", terrain: str = "") -> np.ndarray:
    """단일 포켓몬 → (264,) float32.

    기본(63) + 기술(4×42=168) + 아이템효과(12) + 특성효과(16) + 매치업(5) = 264
    """
    vec = np.zeros(POKEMON_DIM, dtype=np.float32)
    i = 0

    # ── 기본 정보 (63) ──────────────────────────────────────────
    # HP%
    vec[i] = poke.hp_pct
    i += 1

    # Fainted
    vec[i] = float(poke.fainted)
    i += 1

    # Stats (/500 정규화)
    for k in STAT_KEYS:
        vec[i] = poke.stats.get(k, 100) / 500.0
        i += 1

    # Types one-hot (18)
    for t in poke.types:
        t_idx = TYPE_TO_IDX.get(t, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[i + t_idx] = 1.0
    i += NUM_TYPES

    # Status one-hot (7)
    s_idx = STATUS_TO_IDX.get(poke.status, 0) if poke.status else 0
    vec[i + s_idx] = 1.0
    i += NUM_STATUS

    # Status effects (5)
    status_fx = _encode_status_effects(
        poke.status,
        getattr(poke, 'sleep_turns', 0),
        getattr(poke, 'tox_counter', 0),
    )
    vec[i:i + N_STATUS_FX] = status_fx
    i += N_STATUS_FX

    # Boosts (/6 정규화)
    for k in STAT_KEYS[1:]:  # atk, def, spa, spd, spe
        vec[i] = poke.boosts.get(k, 0) / 6.0
        i += 1

    # Tera type one-hot (18)
    if poke.tera_type:
        t_idx = TYPE_TO_IDX.get(poke.tera_type, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[i + t_idx] = 1.0
    i += NUM_TYPES

    # Is terastallized
    vec[i] = float(poke.is_tera)
    i += 1

    # Tera available (테라 타입 있고 팀이 아직 미사용)
    vec[i] = float(tera_available and bool(poke.tera_type))
    i += 1

    # ── 기술 (4 × 42) ───────────────────────────────────────────
    poke_types = poke.types
    item_id = _to_id(poke.item) if poke.item else ""
    ability_id = _to_id(poke.ability) if poke.ability else ""
    is_burned = (poke.status == "brn")

    # 공격 스탯 계산 (부스트 포함)
    atk_stat = poke.stats.get("atk", 100)
    atk_boost = poke.boosts.get("atk", 0)
    if atk_boost > 0:
        atk_stat = atk_stat * (2 + atk_boost) / 2
    elif atk_boost < 0:
        atk_stat = atk_stat * 2 / (2 - atk_boost)

    spa_stat = poke.stats.get("spa", 100)
    spa_boost = poke.boosts.get("spa", 0)
    if spa_boost > 0:
        spa_stat = spa_stat * (2 + spa_boost) / 2
    elif spa_boost < 0:
        spa_stat = spa_stat * 2 / (2 - spa_boost)

    # 아이템 공격 보정
    item_fx_data = ITEM_EFFECTS.get(item_id, {})
    item_damage_mod = item_fx_data.get("damage_mod", 1.0)

    # 상대 방어 스탯
    if opp_active and not opp_active.fainted:
        opp_def = opp_active.stats.get("def", 100)
        def_boost = opp_active.boosts.get("def", 0)
        if def_boost > 0:
            opp_def = opp_def * (2 + def_boost) / 2
        elif def_boost < 0:
            opp_def = opp_def * 2 / (2 - def_boost)

        opp_spd = opp_active.stats.get("spd", 100)
        spd_boost = opp_active.boosts.get("spd", 0)
        if spd_boost > 0:
            opp_spd = opp_spd * (2 + spd_boost) / 2
        elif spd_boost < 0:
            opp_spd = opp_spd * 2 / (2 - spd_boost)

        opp_cur_hp = opp_active.cur_hp
        opp_max_hp = max(opp_active.max_hp, 1)
    else:
        opp_def = 100
        opp_spd = 100
        opp_cur_hp = 0
        opp_max_hp = 1

    for m_i in range(MOVES_PER_POKEMON):
        if m_i < len(poke.moves):
            move_data = gd.get_move(poke.moves[m_i])
            if move_data:
                # ── 기존 24 dim ──
                # Type one-hot (18)
                mt_name = move_data["type"]
                mt = TYPE_TO_IDX.get(mt_name, 0)
                vec[i + mt] = 1.0
                # Power / 250
                power = move_data["basePower"]
                vec[i + 18] = power / 250.0
                # Category one-hot (3): physical, special, status
                if move_data["is_physical"]:
                    vec[i + 19] = 1.0
                elif move_data["is_special"]:
                    vec[i + 20] = 1.0
                else:
                    vec[i + 21] = 1.0
                # Priority / 5
                vec[i + 22] = move_data["priority"] / 5.0
                # Effectiveness vs opponent (핵심 시그널)
                type_eff = 1.0
                if (opp_active and not opp_active.fainted
                        and not move_data["is_status"]):
                    type_eff = gd.effectiveness(mt_name, opp_active.types)
                    vec[i + 23] = type_eff / 4.0

                # ── 기술 스펙 15 dim ──
                move_fx = _encode_move_effects(move_data)
                vec[i + MOVE_BASE_DIM:i + MOVE_BASE_DIM + MOVE_EFFECTS_DIM] = move_fx

                # ── 지식 계산 3 dim ──
                knowledge_offset = MOVE_BASE_DIM + MOVE_EFFECTS_DIM
                if not move_data["is_status"] and opp_active and not opp_active.fainted:
                    # STAB
                    is_stab = mt_name in poke_types
                    stab_mod = 1.5 if is_stab else 1.0
                    vec[i + knowledge_offset + 1] = float(is_stab)

                    # damage_pct
                    if move_data["is_physical"]:
                        a_stat = atk_stat
                        d_stat = opp_def
                    else:
                        a_stat = spa_stat
                        d_stat = opp_spd

                    w_mod = _weather_mod_for_type(mt_name, weather)
                    est_dmg = _calc_damage_fast(
                        a_stat, d_stat, power, stab_mod, type_eff,
                        weather_mod=w_mod * item_damage_mod,
                        is_burned=(is_burned and move_data["is_physical"]),
                    )
                    dmg_pct = min(est_dmg / max(opp_max_hp, 1), 1.0)
                    vec[i + knowledge_offset] = dmg_pct

                    # can_ohko
                    vec[i + knowledge_offset + 2] = float(est_dmg >= opp_cur_hp)
        i += MOVE_DIM

    # ── 아이템 효과 (12) ──────────────────────────────────────
    vec[i:i + N_ITEM_FX] = _encode_item_effects(item_id)
    i += N_ITEM_FX

    # ── 특성 효과 (16) ───────────────────────────────────────
    vec[i:i + N_ABILITY_FX] = _encode_ability_effects(ability_id)
    i += N_ABILITY_FX

    # ── 매치업 피처 (5) ──────────────────────────────────────
    if opp_active and not opp_active.fainted and not poke.fainted:
        my_speed = poke.effective_speed(weather, terrain)
        opp_speed = opp_active.effective_speed(weather, terrain)

        # speed_advantage
        vec[i] = float(my_speed > opp_speed)
        # speed_ratio
        speed_sum = my_speed + opp_speed
        vec[i + 1] = my_speed / max(speed_sum, 1.0)

        # max_incoming_pct & can_be_ohko & defensive_eff
        max_inc = 0.0
        worst_eff = 0.0
        my_hp = max(poke.max_hp, 1)
        my_cur_hp = poke.cur_hp
        my_def = poke.stats.get("def", 100)
        my_def_boost = poke.boosts.get("def", 0)
        if my_def_boost > 0:
            my_def = my_def * (2 + my_def_boost) / 2
        elif my_def_boost < 0:
            my_def = my_def * 2 / (2 - my_def_boost)
        my_spd_stat = poke.stats.get("spd", 100)
        my_spd_boost = poke.boosts.get("spd", 0)
        if my_spd_boost > 0:
            my_spd_stat = my_spd_stat * (2 + my_spd_boost) / 2
        elif my_spd_boost < 0:
            my_spd_stat = my_spd_stat * 2 / (2 - my_spd_boost)

        opp_atk = opp_active.stats.get("atk", 100)
        opp_atk_boost = opp_active.boosts.get("atk", 0)
        if opp_atk_boost > 0:
            opp_atk = opp_atk * (2 + opp_atk_boost) / 2
        elif opp_atk_boost < 0:
            opp_atk = opp_atk * 2 / (2 - opp_atk_boost)

        opp_spa = opp_active.stats.get("spa", 100)
        opp_spa_boost = opp_active.boosts.get("spa", 0)
        if opp_spa_boost > 0:
            opp_spa = opp_spa * (2 + opp_spa_boost) / 2
        elif opp_spa_boost < 0:
            opp_spa = opp_spa * 2 / (2 - opp_spa_boost)

        opp_burned = (opp_active.status == "brn")

        for om_id in opp_active.moves:
            om = gd.get_move(om_id)
            if not om or om["is_status"]:
                continue
            o_eff = gd.effectiveness(om["type"], poke.types)
            worst_eff = max(worst_eff, o_eff)
            o_stab = 1.5 if om["type"] in opp_active.types else 1.0
            if om["is_physical"]:
                o_dmg = _calc_damage_fast(
                    opp_atk, my_def, om["basePower"], o_stab, o_eff,
                    weather_mod=_weather_mod_for_type(om["type"], weather),
                    is_burned=opp_burned)
            else:
                o_dmg = _calc_damage_fast(
                    opp_spa, my_spd_stat, om["basePower"], o_stab, o_eff,
                    weather_mod=_weather_mod_for_type(om["type"], weather))
            inc_pct = min(o_dmg / my_hp, 1.0)
            max_inc = max(max_inc, inc_pct)

        vec[i + 2] = max_inc                           # max_incoming_pct
        vec[i + 3] = float(max_inc * my_hp >= my_cur_hp)  # can_be_ohko
        vec[i + 4] = worst_eff / 4.0                   # defensive_eff
    i += MATCHUP_DIM

    return vec


def _encode_side(side) -> np.ndarray:
    """사이드 컨디션 → (8,) float32."""
    return np.array([
        float(side.stealth_rock),
        side.spikes / 3.0,
        side.toxic_spikes / 2.0,
        float(side.sticky_web),
        side.reflect_turns / 5.0,
        side.light_screen_turns / 5.0,
        side.tailwind_turns / 4.0,
        float(side.tera_used),
    ], dtype=np.float32)


def _encode_global(state: BattleState) -> np.ndarray:
    """글로벌 상태 → (14,) float32."""
    vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    i = 0
    # Weather one-hot (5: NONE, SUN, RAIN, SAND, SNOW)
    vec[i + int(state.weather)] = 1.0
    i += 5
    vec[i] = state.weather_turns / 5.0
    i += 1
    # Terrain one-hot (5: NONE, ELECTRIC, GRASSY, MISTY, PSYCHIC)
    vec[i + int(state.terrain)] = 1.0
    i += 5
    vec[i] = state.terrain_turns / 5.0
    i += 1
    # Trick room
    vec[i] = float(state.trick_room_turns > 0)
    i += 1
    # Turn (/50)
    vec[i] = min(state.turn / 50.0, 1.0)
    i += 1
    return vec


def encode_state(state: BattleState, player: int,
                 gd: GameData) -> torch.Tensor:
    """BattleState → (1614,) float32 텐서.

    player 관점 정규화: player 쪽이 항상 "아군".
    """
    ally = state.sides[player]
    opp = state.sides[1 - player]
    parts = []

    # 날씨/필드 문자열 추출
    weather_str = WEATHER_STR.get(state.weather, "")
    terrain_str = TERRAIN_STR.get(state.terrain, "")

    # 아군 포켓몬 ×3
    opp_act = opp.active
    for j in range(TEAM_SIZE):
        if j < len(ally.team):
            parts.append(_encode_pokemon(
                ally.team[j], opp_act, gd, not ally.tera_used,
                weather=weather_str, terrain=terrain_str))
        else:
            parts.append(np.zeros(POKEMON_DIM, dtype=np.float32))

    # 상대 포켓몬 ×3
    ally_act = ally.active
    for j in range(TEAM_SIZE):
        if j < len(opp.team):
            parts.append(_encode_pokemon(
                opp.team[j], ally_act, gd, not opp.tera_used,
                weather=weather_str, terrain=terrain_str))
        else:
            parts.append(np.zeros(POKEMON_DIM, dtype=np.float32))

    # 사이드 컨디션
    parts.append(_encode_side(ally))
    parts.append(_encode_side(opp))

    # 글로벌
    parts.append(_encode_global(state))

    return torch.from_numpy(np.concatenate(parts))


# ═══════════════════════════════════════════════════════════════
#  TeamPreview State 인코딩
# ═══════════════════════════════════════════════════════════════

def _encode_preview_pokemon(poke: Pokemon, gd: GameData) -> np.ndarray:
    """프리뷰용 단일 포켓몬 → (60,) float32.

    types(18) + stats_normalized(6) + tera_type(18) + move_type_coverage(18)
    """
    vec = np.zeros(PREVIEW_POKEMON_DIM, dtype=np.float32)
    i = 0

    # Types one-hot (18)
    for t in poke.types:
        t_idx = TYPE_TO_IDX.get(t, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[i + t_idx] = 1.0
    i += NUM_TYPES

    # Stats /500 정규화 (6)
    for k in STAT_KEYS:
        vec[i] = poke.stats.get(k, 100) / 500.0
        i += 1

    # Tera type one-hot (18)
    if poke.tera_type:
        t_idx = TYPE_TO_IDX.get(poke.tera_type, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[i + t_idx] = 1.0
    i += NUM_TYPES

    # Move type coverage: 보유 기술 타입 one-hot 합산 (18)
    for mid in poke.moves:
        move_data = gd.get_move(mid)
        if move_data and not move_data["is_status"]:
            mt = TYPE_TO_IDX.get(move_data["type"], -1)
            if 0 <= mt < NUM_TYPES:
                vec[i + mt] = 1.0
    i += NUM_TYPES

    return vec


def _best_move_effectiveness(attacker: Pokemon, defender: Pokemon,
                             gd: GameData) -> float:
    """attacker가 defender에게 가장 효과적인 기술의 상성값 / 4.0."""
    best = 0.0
    for mid in attacker.moves:
        move_data = gd.get_move(mid)
        if move_data and not move_data["is_status"]:
            eff = gd.effectiveness(move_data["type"], defender.types)
            if eff > best:
                best = eff
    return best / 4.0


def encode_preview_state(ally_team: list[Pokemon],
                         opp_team: list[Pokemon],
                         gd: GameData) -> torch.Tensor:
    """팀 프리뷰 상태 → (804,) float32 텐서.

    구성: 아군6(360) + 상대6(360) + 크로스매치업(72) + 글로벌(12) = 804
    """
    parts: list[np.ndarray] = []

    # 아군 6마리 (360)
    for j in range(PREVIEW_TEAM_SIZE):
        if j < len(ally_team):
            parts.append(_encode_preview_pokemon(ally_team[j], gd))
        else:
            parts.append(np.zeros(PREVIEW_POKEMON_DIM, dtype=np.float32))

    # 상대 6마리 (360)
    for j in range(PREVIEW_TEAM_SIZE):
        if j < len(opp_team):
            parts.append(_encode_preview_pokemon(opp_team[j], gd))
        else:
            parts.append(np.zeros(PREVIEW_POKEMON_DIM, dtype=np.float32))

    # 크로스 매치업 (72 = 36 + 36)
    cross = np.zeros(PREVIEW_CROSS_DIM, dtype=np.float32)
    idx = 0
    # ally→opp: 아군i가 상대j에게 가장 효과적인 기술 상성
    for i in range(PREVIEW_TEAM_SIZE):
        for j in range(PREVIEW_TEAM_SIZE):
            if i < len(ally_team) and j < len(opp_team):
                cross[idx] = _best_move_effectiveness(
                    ally_team[i], opp_team[j], gd)
            idx += 1
    # opp→ally: 상대j가 아군i에게 가장 효과적인 기술 상성
    for j in range(PREVIEW_TEAM_SIZE):
        for i in range(PREVIEW_TEAM_SIZE):
            if j < len(opp_team) and i < len(ally_team):
                cross[idx] = _best_move_effectiveness(
                    opp_team[j], ally_team[i], gd)
            idx += 1
    parts.append(cross)

    # 글로벌 (12)
    glb = np.zeros(PREVIEW_GLOBAL_DIM, dtype=np.float32)
    g_idx = 0
    # 아군 평균 스피드
    ally_speeds = [p.stats.get("spe", 100) for p in ally_team]
    glb[g_idx] = (sum(ally_speeds) / max(len(ally_speeds), 1)) / 500.0
    g_idx += 1
    # 상대 평균 스피드
    opp_speeds = [p.stats.get("spe", 100) for p in opp_team]
    glb[g_idx] = (sum(opp_speeds) / max(len(opp_speeds), 1)) / 500.0
    g_idx += 1
    # 아군 스피드 분산
    if len(ally_speeds) > 1:
        glb[g_idx] = float(np.std(ally_speeds)) / 200.0
    g_idx += 1
    # 상대 스피드 분산
    if len(opp_speeds) > 1:
        glb[g_idx] = float(np.std(opp_speeds)) / 200.0
    g_idx += 1
    # 아군 타입 다양성 (고유 타입 수 / 18)
    ally_types = set()
    for p in ally_team:
        ally_types.update(p.types)
    glb[g_idx] = len(ally_types) / NUM_TYPES
    g_idx += 1
    # 상대 타입 다양성
    opp_types = set()
    for p in opp_team:
        opp_types.update(p.types)
    glb[g_idx] = len(opp_types) / NUM_TYPES
    g_idx += 1
    # 아군 평균 HP 스탯
    ally_hp = [p.stats.get("hp", 100) for p in ally_team]
    glb[g_idx] = (sum(ally_hp) / max(len(ally_hp), 1)) / 500.0
    g_idx += 1
    # 상대 평균 HP 스탯
    opp_hp = [p.stats.get("hp", 100) for p in opp_team]
    glb[g_idx] = (sum(opp_hp) / max(len(opp_hp), 1)) / 500.0
    g_idx += 1
    # 아군 평균 공격 (max(atk, spa))
    ally_atk = [max(p.stats.get("atk", 100), p.stats.get("spa", 100))
                for p in ally_team]
    glb[g_idx] = (sum(ally_atk) / max(len(ally_atk), 1)) / 500.0
    g_idx += 1
    # 상대 평균 공격
    opp_atk = [max(p.stats.get("atk", 100), p.stats.get("spa", 100))
               for p in opp_team]
    glb[g_idx] = (sum(opp_atk) / max(len(opp_atk), 1)) / 500.0
    g_idx += 1
    # 아군 평균 방어 (min(def, spd))
    ally_def = [min(p.stats.get("def", 100), p.stats.get("spd", 100))
                for p in ally_team]
    glb[g_idx] = (sum(ally_def) / max(len(ally_def), 1)) / 500.0
    g_idx += 1
    # 상대 평균 방어
    opp_def = [min(p.stats.get("def", 100), p.stats.get("spd", 100))
               for p in opp_team]
    glb[g_idx] = (sum(opp_def) / max(len(opp_def), 1)) / 500.0
    g_idx += 1
    parts.append(glb)

    return torch.from_numpy(np.concatenate(parts))


# ═══════════════════════════════════════════════════════════════
#  TeamBuild State 인코딩
# ═══════════════════════════════════════════════════════════════

def encode_build_state(selected_pokemon: list[Pokemon], step: int,
                       gd: GameData) -> torch.Tensor:
    """팀 빌드 상태 → (430,) float32 텐서.

    구성: 선택된 팀(360) + 팀 집계(62) + 스텝 컨텍스트(8) = 430
    - 정규 순서: 선택된 포켓몬을 usage_pct 내림차순 정렬 후 인코딩 (순서 불변성)
    """
    parts: list[np.ndarray] = []

    # 정규 순서: usage_pct 내림차순 정렬
    if selected_pokemon:
        sorted_pokes = sorted(
            selected_pokemon,
            key=lambda p: (gd.get_stats(p.name) or {}).get("usage_pct", 0.0),
            reverse=True,
        )
    else:
        sorted_pokes = []

    # 선택된 팀 (360 = 6 × 60)
    for j in range(BUILD_TEAM_SIZE):
        if j < len(sorted_pokes):
            parts.append(_encode_preview_pokemon(sorted_pokes[j], gd))
        else:
            parts.append(np.zeros(PREVIEW_POKEMON_DIM, dtype=np.float32))

    # 팀 집계 (62)
    agg = np.zeros(62, dtype=np.float32)
    if selected_pokemon:
        # 공격 커버리지 (18): 각 타입에 최대 상성 / 4.0
        for t_idx in range(NUM_TYPES):
            best_eff = 0.0
            for poke in selected_pokemon:
                for mid in poke.moves:
                    move_data = gd.get_move(mid)
                    if move_data and not move_data["is_status"]:
                        eff = gd.effectiveness(move_data["type"], [TYPES[t_idx]])
                        best_eff = max(best_eff, eff)
            agg[t_idx] = best_eff / 4.0

        # 방어 커버리지 (18): 각 타입 공격에 팀 최소 피해 / 4.0
        for t_idx in range(NUM_TYPES):
            min_eff = 4.0
            for poke in selected_pokemon:
                eff = gd.effectiveness(TYPES[t_idx], poke.types)
                min_eff = min(min_eff, eff)
            agg[18 + t_idx] = min_eff / 4.0

        # 미커버 (18): 공격 커버리지 ≤ 1.0인 타입 binary
        for t_idx in range(NUM_TYPES):
            if agg[t_idx] * 4.0 <= 1.0:
                agg[36 + t_idx] = 1.0

        # 스탯 평균 (6)
        n = len(selected_pokemon)
        for s_idx, key in enumerate(STAT_KEYS):
            avg_stat = sum(p.stats.get(key, 100) for p in selected_pokemon) / n
            agg[54 + s_idx] = avg_stat / 500.0

        # 타입 다양성 (1)
        all_types = set()
        for p in selected_pokemon:
            all_types.update(p.types)
        agg[60] = len(all_types) / NUM_TYPES

        # 스피드 분산 (1)
        speeds = [p.stats.get("spe", 100) for p in selected_pokemon]
        if len(speeds) > 1:
            agg[61] = float(np.std(speeds)) / 200.0

    parts.append(agg)

    # 스텝 컨텍스트 (8)
    ctx = np.zeros(8, dtype=np.float32)
    ctx[0] = step / 6.0
    ctx[1] = (BUILD_TEAM_SIZE - step) / 6.0
    if step < BUILD_TEAM_SIZE:
        ctx[2 + step] = 1.0
    parts.append(ctx)

    return torch.from_numpy(np.concatenate(parts))


# ═══════════════════════════════════════════════════════════════
#  ResBlock
# ═══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear + residual."""

    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(x))
        out = self.fc1(out)
        out = F.relu(self.ln2(out))
        out = self.fc2(out)
        return out + residual


# ═══════════════════════════════════════════════════════════════
#  PokemonNet
# ═══════════════════════════════════════════════════════════════

class PokemonNet(nn.Module):
    """AlphaZero-style Policy + Value network (~2.2M params).

    Move Scoring Shortcut: active pokemon의 per-move 피처(42d)를
    해당 action logit에 직접 바이어스로 추가.
    → type_eff, damage_pct, can_ohko 등의 지식이 기술 선택에 즉시 반영됨.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 512,
                 n_res_blocks: int = 6, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.input_fc = nn.Linear(state_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_dim, 256)
        self.policy_fc2 = nn.Linear(256, n_actions)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Move scoring shortcut: per-move(42d) → 1 scalar bias
        # 기술의 type_eff, damage_pct, can_ohko가 해당 action logit에 직접 반영
        self.move_score = nn.Sequential(
            nn.Linear(MOVE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Value head 작은 초기값
        nn.init.uniform_(self.value_fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_fc2.bias)
        # Move shortcut 출력층 0 초기화 → 기존 체크포인트와 동일 동작
        nn.init.zeros_(self.move_score[2].weight)
        nn.init.zeros_(self.move_score[2].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, STATE_DIM) or (STATE_DIM,)
        Returns:
            policy_logits: (batch, NUM_ACTIONS) — raw logits (마스킹 전)
            value: (batch, 1) — tanh [-1, +1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # ── Move scoring shortcut ──
        # Active pokemon(아군 team[0])의 4개 기술 피처 추출 → 직접 점수화
        move_scores = []
        for m_i in range(MOVES_PER_POKEMON):
            start = POKEMON_BASE_DIM + m_i * MOVE_DIM   # 63 + i*42
            end = start + MOVE_DIM
            mf = x[:, start:end]                         # (batch, 42)
            move_scores.append(self.move_score(mf))      # (batch, 1)
        move_bias = torch.cat(move_scores, dim=1)        # (batch, 4)

        # ── Backbone ──
        h = F.relu(self.input_fc(x))
        h = self.res_blocks(h)
        h = self.ln(h)

        p = F.relu(self.policy_fc1(h))
        p = self.policy_fc2(p)

        # Move shortcut: 기술(0-3) + 테라기술(4-7)에 직접 바이어스
        bias = p.new_zeros(p.shape)
        bias[:, :4] = move_bias
        bias[:, 4:8] = move_bias          # 테라기술도 같은 기술
        p = p + bias

        v = F.relu(self.value_fc1(h))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        """기존 체크포인트 호환: move_score 키 없으면 0 초기화 유지."""
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        if result.missing_keys:
            print(f"  [PokemonNet] 새 레이어 (0 초기화 유지): "
                  f"{[k.split('.')[-2]+'.'+k.split('.')[-1] for k in result.missing_keys]}")
        return result

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════
#  TeamPreviewNet
# ═══════════════════════════════════════════════════════════════

class TeamPreviewNet(nn.Module):
    """팀 프리뷰용 Policy + Value network (~300K params).

    Input(804) → Linear(804→256) → 3× ResBlock(256) → LayerNorm
      ├─ Policy: Linear(256→128) → ReLU → Linear(128→20) → softmax
      └─ Value:  Linear(256→128) → ReLU → Linear(128→1)  → tanh
    """

    def __init__(self, state_dim: int = PREVIEW_STATE_DIM,
                 hidden_dim: int = 256, n_res_blocks: int = 3,
                 n_combos: int = NUM_COMBOS):
        super().__init__()
        self.input_fc = nn.Linear(state_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_dim, 128)
        self.policy_fc2 = nn.Linear(128, n_combos)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.uniform_(self.value_fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, PREVIEW_STATE_DIM) or (PREVIEW_STATE_DIM,)
        Returns:
            policy_logits: (batch, NUM_COMBOS) — raw logits
            value: (batch, 1) — tanh [-1, +1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = F.relu(self.input_fc(x))
        h = self.res_blocks(h)
        h = self.ln(h)

        p = F.relu(self.policy_fc1(h))
        p = self.policy_fc2(p)

        v = F.relu(self.value_fc1(h))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════
#  TeamBuildNet
# ═══════════════════════════════════════════════════════════════

class TeamBuildNet(nn.Module):
    """팀 빌드용 Policy + Value network (~711K params).

    Input(430) → Linear(430→256) → 4× ResBlock(256) → LayerNorm
      ├─ Policy: Linear(256→128) → ReLU → Linear(128→50) → [mask + softmax]
      └─ Value:  Linear(256→128) → ReLU → Linear(128→1)  → tanh
    """

    def __init__(self, state_dim: int = BUILD_STATE_DIM,
                 hidden_dim: int = 256, n_res_blocks: int = 4,
                 n_candidates: int = N_CANDIDATES):
        super().__init__()
        self.input_fc = nn.Linear(state_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_dim, 128)
        self.policy_fc2 = nn.Linear(128, n_candidates)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.uniform_(self.value_fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, BUILD_STATE_DIM) or (BUILD_STATE_DIM,)
        Returns:
            policy_logits: (batch, N_CANDIDATES) — raw logits
            value: (batch, 1) — tanh [-1, +1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = F.relu(self.input_fc(x))
        h = self.res_blocks(h)
        h = self.ln(h)

        p = F.relu(self.policy_fc1(h))
        p = self.policy_fc2(p)

        v = F.relu(self.value_fc1(h))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════
#  PreviewEvaluator (팀 프리뷰 래퍼)
# ═══════════════════════════════════════════════════════════════

class PreviewEvaluator:
    """TeamPreviewNet 래퍼: encode → forward → softmax → (combo_probs, value)."""

    def __init__(self, model: TeamPreviewNet, game_data: GameData,
                 device: torch.device | str = "cpu"):
        self.model = model
        self.gd = game_data
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, ally_team: list[Pokemon],
                 opp_team: list[Pokemon]
                 ) -> tuple[np.ndarray, float]:
        """팀 프리뷰 → (combo_probs[20], value).

        Returns:
            combo_probs: (20,) — C(6,3) 조합 확률
            value: float in [-1, +1]
        """
        x = encode_preview_state(ally_team, opp_team, self.gd).to(self.device)
        logits, v = self.model(x)
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=0).cpu().numpy()
        return probs, v.item()

    @torch.no_grad()
    def choose(self, ally_team: list[Pokemon],
               opp_team: list[Pokemon],
               temperature: float = 0.1) -> list[int]:
        """확률에서 콤보 샘플 → [idx1, idx2, idx3] 반환."""
        probs, _ = self.evaluate(ally_team, opp_team)

        if temperature < 0.01:
            combo_idx = int(np.argmax(probs))
        else:
            # temperature 적용
            logits = np.log(probs + 1e-8) / temperature
            logits -= logits.max()
            exp_logits = np.exp(logits)
            temp_probs = exp_logits / exp_logits.sum()
            combo_idx = np.random.choice(NUM_COMBOS, p=temp_probs)

        return list(COMBO_TABLE[combo_idx])


# ═══════════════════════════════════════════════════════════════
#  BuildEvaluator (팀 빌드 래퍼)
# ═══════════════════════════════════════════════════════════════

class BuildEvaluator:
    """TeamBuildNet 래퍼: autoregressive 6마리 팀 빌드."""

    def __init__(self, model: TeamBuildNet, game_data: GameData,
                 candidate_pool: list[str], format_name: str = "bss",
                 device: torch.device | str = "cpu"):
        self.model = model
        self.gd = game_data
        self.candidate_names = candidate_pool
        self.format_name = format_name
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 후보 풀 Pokemon 객체 미리 빌드하여 캐시
        self.candidate_pokemon: list[Pokemon | None] = []
        for name in candidate_pool:
            try:
                poke = make_pokemon_from_stats(game_data, name, format_name)
                self.candidate_pokemon.append(poke)
            except (ValueError, KeyError):
                self.candidate_pokemon.append(None)

    @torch.no_grad()
    def evaluate_step(self, selected: list[Pokemon], step: int,
                      selected_indices: list[int]
                      ) -> tuple[np.ndarray, float]:
        """현재 빌드 상태 평가 → (probs[50], value)."""
        x = encode_build_state(selected, step, self.gd).to(self.device)
        logits, v = self.model(x)
        logits = logits.squeeze(0)

        # 마스킹: 이미 선택된 후보 + 빌드 불가 후보
        mask = torch.zeros(N_CANDIDATES, device=self.device)
        for idx in selected_indices:
            mask[idx] = float('-inf')
        for idx, poke in enumerate(self.candidate_pokemon):
            if poke is None:
                mask[idx] = float('-inf')
        masked_logits = logits + mask

        probs = F.softmax(masked_logits, dim=0).cpu().numpy()
        return probs, v.item()

    @torch.no_grad()
    def build_team(self, temperature: float = 1.0
                   ) -> tuple[list[Pokemon], list]:
        """6단계 autoregressive 팀 빌드 → (team_6, build_examples).

        build_examples의 value는 0.0 (나중에 게임 결과로 채움).
        """
        from self_play import BuildTrainingExample

        self.model.eval()
        selected: list[Pokemon] = []
        selected_indices: list[int] = []
        build_examples: list[BuildTrainingExample] = []

        for step in range(BUILD_TEAM_SIZE):
            probs, value = self.evaluate_step(
                selected, step, selected_indices)

            state = encode_build_state(
                selected, step, self.gd).numpy()

            build_examples.append(BuildTrainingExample(
                state=state,
                policy=probs.copy(),
                value=0.0,
                step=step,
            ))

            # Temperature 적용 샘플링
            if temperature < 0.01:
                idx = int(np.argmax(probs))
            else:
                log_probs = np.log(probs + 1e-8) / temperature
                log_probs -= log_probs.max()
                exp_probs = np.exp(log_probs)
                temp_probs = exp_probs / exp_probs.sum()
                idx = int(np.random.choice(N_CANDIDATES, p=temp_probs))

            selected_indices.append(idx)
            selected.append(self.candidate_pokemon[idx])

        return selected, build_examples


# ═══════════════════════════════════════════════════════════════
#  NetworkEvaluator (MCTS 연결용 래퍼)
# ═══════════════════════════════════════════════════════════════

class NetworkEvaluator:
    """MCTS에서 사용: encode → forward → mask → softmax → (prior_dict, value)."""

    def __init__(self, model: PokemonNet, game_data: GameData,
                 device: torch.device | str = "cpu"):
        self.model = model
        self.gd = game_data
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, state: BattleState, player: int,
                 legal_actions: list[int]) -> tuple[dict[int, float], float]:
        """State → (prior_dict, value).

        Returns:
            priors: {action: probability} — 합계 1.0
            value: float in [-1, +1] — player 관점 형세
        """
        x = encode_state(state, player, self.gd).to(self.device)
        logits, v = self.model(x)
        logits = logits.squeeze(0)   # (NUM_ACTIONS,)
        value = v.item()

        # 마스킹: 불법 액션 -inf
        mask = torch.full((NUM_ACTIONS,), float('-inf'), device=self.device)
        for a in legal_actions:
            mask[a] = 0.0
        masked_logits = logits + mask

        probs = F.softmax(masked_logits, dim=0).cpu().numpy()
        priors = {a: float(probs[a]) for a in legal_actions}
        return priors, value

    @torch.no_grad()
    def evaluate_value(self, state: BattleState, player: int) -> float:
        """Value만 반환 (롤아웃 대체용)."""
        x = encode_state(state, player, self.gd).to(self.device)
        _, v = self.model(x)
        return v.item()

    @torch.no_grad()
    def evaluate_values_batch(self, states: list[BattleState],
                              player: int) -> np.ndarray:
        """N개 상태 배치 평가 → (N,) values in [-1,+1].

        Args:
            states: BattleState 리스트 (N개)
            player: 평가 관점 플레이어 (0 or 1)

        Returns:
            (N,) numpy array of values in [-1, +1]
        """
        if not states:
            return np.array([], dtype=np.float32)
        encodings = [encode_state(s, player, self.gd) for s in states]
        batch = torch.stack(encodings).to(self.device)
        _, values = self.model(batch)
        return values.squeeze(-1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """neural_net.py 검증."""
    print("=== Neural Net 검증 ===\n")

    gd = GameData(device="cpu")
    from battle_sim import BattleSimulator, make_pokemon_from_stats
    sim = BattleSimulator(gd)

    # 테스트 상태 생성
    team1 = [make_pokemon_from_stats(gd, n, "bss")
             for n in ["Koraidon", "Ting-Lu", "Flutter Mane"]]
    team2 = [make_pokemon_from_stats(gd, n, "bss")
             for n in ["Miraidon", "Gholdengo", "Great Tusk"]]
    state = sim.create_battle_state(team1, team2)

    # 인코딩 테스트
    x = encode_state(state, 0, gd)
    print(f"State 인코딩: shape={x.shape}, dtype={x.dtype}")
    print(f"  예상: ({STATE_DIM},) = "
          f"({POKEMON_DIM}×{TEAM_SIZE}×2 + {SIDE_DIM}×2 + {GLOBAL_DIM})")
    assert x.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {x.shape}"
    print(f"  min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")

    # Player 1 관점도 확인
    x1 = encode_state(state, 1, gd)
    assert x1.shape == (STATE_DIM,)
    print(f"  P1 관점: min={x1.min():.3f}, max={x1.max():.3f}")

    # 네트워크 테스트
    model = PokemonNet()
    print(f"\n네트워크 파라미터: {model.count_params():,}")

    logits, value = model(x)
    print(f"Policy logits: shape={logits.shape}")
    print(f"Value: {value.item():.4f}")

    # NetworkEvaluator 테스트
    evaluator = NetworkEvaluator(model, gd, device="cpu")
    legal = sim.get_legal_actions(state, 0)
    priors, val = evaluator.evaluate(state, 0, legal)
    print(f"\nNetworkEvaluator:")
    print(f"  Legal actions: {legal}")
    print(f"  Priors: { {a: f'{p:.3f}' for a, p in priors.items()} }")
    print(f"  Prior sum: {sum(priors.values()):.4f}")
    print(f"  Value: {val:.4f}")

    # Value-only 평가
    val_only = evaluator.evaluate_value(state, 0)
    print(f"  Value-only: {val_only:.4f}")

    # 배치 테스트
    batch = torch.stack([encode_state(state, 0, gd) for _ in range(8)])
    logits_b, value_b = model(batch)
    print(f"\n배치 테스트: input={batch.shape} → "
          f"logits={logits_b.shape}, value={value_b.shape}")

    # ── TeamPreviewNet 검증 ──
    print(f"\n{'='*40}")
    print("TeamPreviewNet 검증")
    print(f"{'='*40}")

    print(f"COMBO_TABLE: {NUM_COMBOS} combos, "
          f"first={COMBO_TABLE[0]}, last={COMBO_TABLE[-1]}")

    # 프리뷰 인코딩 테스트
    team_a = [make_pokemon_from_stats(gd, n, "bss")
              for n in ["Koraidon", "Ting-Lu", "Flutter Mane",
                         "Ho-Oh", "Chien-Pao", "Glimmora"]]
    team_b = [make_pokemon_from_stats(gd, n, "bss")
              for n in ["Miraidon", "Gholdengo", "Great Tusk",
                         "Kingambit", "Raging Bolt", "Iron Hands"]]

    px = encode_preview_state(team_a, team_b, gd)
    print(f"\nPreview 인코딩: shape={px.shape}, dtype={px.dtype}")
    print(f"  예상: ({PREVIEW_STATE_DIM},)")
    assert px.shape == (PREVIEW_STATE_DIM,), \
        f"Expected ({PREVIEW_STATE_DIM},), got {px.shape}"
    print(f"  min={px.min():.3f}, max={px.max():.3f}, mean={px.mean():.3f}")

    # 네트워크 테스트
    p_model = TeamPreviewNet()
    print(f"\nTeamPreviewNet 파라미터: {p_model.count_params():,}")
    p_logits, p_value = p_model(px)
    print(f"Policy logits: shape={p_logits.shape} (expected: (1, {NUM_COMBOS}))")
    print(f"Value: {p_value.item():.4f}")

    # PreviewEvaluator 테스트
    p_eval = PreviewEvaluator(p_model, gd, device="cpu")
    combo_probs, pv = p_eval.evaluate(team_a, team_b)
    print(f"\nPreviewEvaluator:")
    print(f"  combo_probs shape: {combo_probs.shape}, sum={combo_probs.sum():.4f}")
    print(f"  value: {pv:.4f}")
    top3 = np.argsort(combo_probs)[::-1][:3]
    for ci in top3:
        print(f"  combo {ci} {COMBO_TABLE[ci]}: {combo_probs[ci]:.3f} "
              f"→ {[team_a[i].name for i in COMBO_TABLE[ci]]}")

    chosen = p_eval.choose(team_a, team_b, temperature=0.1)
    print(f"\n선택된 콤보: {chosen} → {[team_a[i].name for i in chosen]}")

    # ── TeamBuildNet 검증 ──
    print(f"\n{'='*40}")
    print("TeamBuildNet 검증")
    print(f"{'='*40}")

    # encode_build_state 테스트
    # step=0 (빈 팀)
    bx0 = encode_build_state([], 0, gd)
    print(f"\nencode_build_state step=0: shape={bx0.shape}")
    assert bx0.shape == (BUILD_STATE_DIM,), \
        f"Expected ({BUILD_STATE_DIM},), got {bx0.shape}"

    # step=3 (3마리)
    bx3 = encode_build_state(team_a[:3], 3, gd)
    print(f"encode_build_state step=3: shape={bx3.shape}")
    assert bx3.shape == (BUILD_STATE_DIM,)

    # step=5 (5마리)
    bx5 = encode_build_state(team_a[:5], 5, gd)
    print(f"encode_build_state step=5: shape={bx5.shape}")
    assert bx5.shape == (BUILD_STATE_DIM,)

    # 정규 순서 확인: 같은 팀 다른 순서 → 동일 인코딩
    team_fwd = team_a[:3]
    team_rev = list(reversed(team_a[:3]))
    enc_fwd = encode_build_state(team_fwd, 3, gd)
    enc_rev = encode_build_state(team_rev, 3, gd)
    assert torch.allclose(enc_fwd, enc_rev), "정규 순서 위반!"
    print("정규 순서 검증: OK (다른 순서 → 동일 인코딩)")

    # TeamBuildNet 테스트
    b_model = TeamBuildNet()
    print(f"\nTeamBuildNet 파라미터: {b_model.count_params():,}")
    b_logits, b_value = b_model(bx0)
    print(f"Policy logits: shape={b_logits.shape} "
          f"(expected: (1, {N_CANDIDATES}))")
    print(f"Value: {b_value.item():.4f}")

    # BuildEvaluator 테스트
    from self_play import TeamSampler
    sampler = TeamSampler(gd, "bss", top_n=50)
    b_eval = BuildEvaluator(b_model, gd, sampler.names, "bss", device="cpu")
    built_team, build_exs = b_eval.build_team(temperature=1.0)
    print(f"\nBuildEvaluator.build_team:")
    print(f"  팀: {[p.name for p in built_team]}")
    print(f"  examples: {len(build_exs)}개")
    assert len(built_team) == BUILD_TEAM_SIZE
    assert len(build_exs) == BUILD_TEAM_SIZE

    # 마스킹 확인: 이미 선택된 후보 확률=0
    probs_step1, _ = b_eval.evaluate_step(
        [built_team[0]], 1, [0])  # index 0 마스킹
    print(f"  마스킹 확인: index 0 prob={probs_step1[0]:.6f} (expected ~0)")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
