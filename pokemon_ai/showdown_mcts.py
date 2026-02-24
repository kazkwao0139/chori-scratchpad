"""Showdown 엔진 기반 2-Player MCTS.

기존 mcts.py의 Max/Min 2-ply 구조를 따르되,
BattleState 대신 Showdown battle_id(Node.js 참조)를 사용.

핵심 차이:
- 상태 복제: bridge.fork(battle_id) — toJSON/fromJSON
- 턴 진행:  bridge.step(battle_id, p1c, p2c) — makeChoices
- 합법 수:  Showdown request에서 직접 파싱
- 평가:     request + state_info → encode_showdown_state → network
           (BattleState 중간 변환 없음)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import GameData, _to_id, TYPE_TO_IDX, STATUS_TO_IDX, ITEM_EFFECTS, _TYPE_BOOST_ITEMS, ABILITY_EFFECTS
from neural_net import (
    SHOWDOWN_POKEMON_DIM, SHOWDOWN_MOVE_DIM, SHOWDOWN_STATE_DIM,
    TEAM_SIZE, SIDE_DIM, GLOBAL_DIM,
    MOVE_BASE_DIM, MOVE_EFFECTS_DIM, MOVE_KNOWLEDGE_DIM,
    N_ITEM_FX, N_ABILITY_FX, N_STATUS_FX, MATCHUP_DIM, POKEMON_BASE_DIM,
    _encode_move_effects, _encode_item_effects, _encode_ability_effects,
    _encode_status_effects, _calc_damage_fast, _weather_mod_for_type,
)
from battle_sim import ActionType, NUM_ACTIONS
from showdown_bridge import ShowdownBridge

NUM_TYPES = len(TYPE_TO_IDX)    # 18
NUM_STATUS = len(STATUS_TO_IDX) # 7


# ═══════════════════════════════════════════════════════════════
#  Showdown → 텐서 인코딩 (BattleState 불필요)
# ═══════════════════════════════════════════════════════════════

WEATHER_ID_MAP = {
    '': 0, 'sunnyday': 1, 'raindance': 2,
    'sandstorm': 3, 'snow': 4, 'hail': 4,
}

TERRAIN_ID_MAP = {
    '': 0, 'electricterrain': 1, 'grassyterrain': 2,
    'mistyterrain': 3, 'psychicterrain': 4,
}


def _encode_sd_pokemon(pdata: dict, si_poke: dict,
                       opp_active_types: list, gd: GameData,
                       can_tera_team: bool,
                       opp_si_poke: dict | None = None,
                       opp_pdata: dict | None = None,
                       weather_str: str = "",
                       terrain_str: str = "") -> np.ndarray:
    """Showdown 데이터에서 단일 포켓몬 → (270,) float32.

    pdata: request.side.pokemon[i] 데이터
    si_poke: state_info.sides[s].pokemon[i] 데이터
    opp_si_poke: 상대 active pokemon state_info (매치업 계산용)
    opp_pdata: 상대 active pokemon request 데이터 (매치업 계산용)
    """
    vec = np.zeros(SHOWDOWN_POKEMON_DIM, dtype=np.float32)
    idx = 0

    # ── 기본 정보 (63 + 2 boosts extra = 65) ─────────────────
    # HP fraction (1)
    hp = si_poke.get('hp', 0)
    maxhp = si_poke.get('maxhp', 1)
    vec[idx] = hp / max(maxhp, 1)
    idx += 1

    # Fainted (1)
    vec[idx] = float(hp <= 0)
    idx += 1

    # Stats /500 (6): hp, atk, def, spa, spd, spe
    stats = pdata.get('stats', {})
    vec[idx] = max(maxhp, 1) / 500.0  # HP stat from maxhp
    idx += 1
    for k in ('atk', 'def', 'spa', 'spd', 'spe'):
        vec[idx] = stats.get(k, 100) / 500.0
        idx += 1

    # Types one-hot (18) — state_info에서 (테라 반영 현재 타입)
    poke_types = si_poke.get('types', [])
    for t in poke_types:
        t_idx = TYPE_TO_IDX.get(t, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[idx + t_idx] = 1.0
    idx += NUM_TYPES

    # Status one-hot (7)
    status = si_poke.get('status', '') or ''
    s_idx = STATUS_TO_IDX.get(status, 0) if status else 0
    vec[idx + s_idx] = 1.0
    idx += NUM_STATUS

    # Status effects (5)
    sleep_turns = si_poke.get('sleepTurns', 0)
    tox_counter = si_poke.get('toxicTurns', 0)
    status_fx = _encode_status_effects(status, sleep_turns, tox_counter)
    vec[idx:idx + N_STATUS_FX] = status_fx
    idx += N_STATUS_FX

    # Boosts /6 (5): atk, def, spa, spd, spe
    boosts = si_poke.get('boosts', {})
    for k in ('atk', 'def', 'spa', 'spd', 'spe'):
        vec[idx] = boosts.get(k, 0) / 6.0
        idx += 1

    # Tera type one-hot (18)
    tera_type = pdata.get('teraType', '')
    if tera_type:
        t_idx = TYPE_TO_IDX.get(tera_type, -1)
        if 0 <= t_idx < NUM_TYPES:
            vec[idx + t_idx] = 1.0
    idx += NUM_TYPES

    # is_tera (1)
    vec[idx] = float(bool(pdata.get('terastallized')))
    idx += 1

    # can_tera (1)
    vec[idx] = float(can_tera_team and bool(tera_type))
    idx += 1

    # Boosts extra (2): accuracy, evasion (Showdown 전용)
    vec[idx] = boosts.get('accuracy', 0) / 6.0
    idx += 1
    vec[idx] = boosts.get('evasion', 0) / 6.0
    idx += 1

    # ── 기술 (4 × 43) ────────────────────────────────────────
    moves = pdata.get('moves', [])
    move_slots = si_poke.get('moveSlots', [])

    # 미리 스탯 계산 (데미지 추정용)
    my_atk = stats.get('atk', 100)
    atk_boost = boosts.get('atk', 0)
    if atk_boost > 0:
        my_atk = my_atk * (2 + atk_boost) / 2
    elif atk_boost < 0:
        my_atk = my_atk * 2 / (2 - atk_boost)

    my_spa = stats.get('spa', 100)
    spa_boost = boosts.get('spa', 0)
    if spa_boost > 0:
        my_spa = my_spa * (2 + spa_boost) / 2
    elif spa_boost < 0:
        my_spa = my_spa * 2 / (2 - spa_boost)

    is_burned = (status == 'brn')
    item_id = _to_id(pdata.get('item', '')) if pdata.get('item') else ''
    item_fx_data = ITEM_EFFECTS.get(item_id, {})
    item_damage_mod = item_fx_data.get('damage_mod', 1.0)

    # 상대 방어 스탯
    opp_def_stat = 100
    opp_spd_stat = 100
    opp_cur_hp = 0
    opp_max_hp = 1
    if opp_si_poke:
        opp_stats = (opp_pdata or {}).get('stats', {})
        opp_boosts = opp_si_poke.get('boosts', {})
        opp_def_stat = opp_stats.get('def', 100)
        db = opp_boosts.get('def', 0)
        if db > 0:
            opp_def_stat = opp_def_stat * (2 + db) / 2
        elif db < 0:
            opp_def_stat = opp_def_stat * 2 / (2 - db)
        opp_spd_stat = opp_stats.get('spd', 100)
        sb = opp_boosts.get('spd', 0)
        if sb > 0:
            opp_spd_stat = opp_spd_stat * (2 + sb) / 2
        elif sb < 0:
            opp_spd_stat = opp_spd_stat * 2 / (2 - sb)
        opp_cur_hp = opp_si_poke.get('hp', 0)
        opp_max_hp = max(opp_si_poke.get('maxhp', 1), 1)

    for m_i in range(4):
        if m_i < len(moves):
            move_id = _to_id(moves[m_i])
            move_data = gd.get_move(move_id)
            if move_data:
                # ── 기존 base 24 dim ──
                mt_name = move_data['type']
                mt = TYPE_TO_IDX.get(mt_name, 0)
                vec[idx + mt] = 1.0
                power = move_data['basePower']
                vec[idx + 18] = power / 250.0
                if move_data['is_physical']:
                    vec[idx + 19] = 1.0
                elif move_data['is_special']:
                    vec[idx + 20] = 1.0
                else:
                    vec[idx + 21] = 1.0
                vec[idx + 22] = move_data['priority'] / 5.0
                type_eff = 1.0
                if opp_active_types and not move_data['is_status']:
                    type_eff = gd.effectiveness(mt_name, opp_active_types)
                    vec[idx + 23] = type_eff / 4.0

                # ── 기술 스펙 15 dim ──
                move_fx = _encode_move_effects(move_data)
                vec[idx + MOVE_BASE_DIM:idx + MOVE_BASE_DIM + MOVE_EFFECTS_DIM] = move_fx

                # ── 지식 계산 3 dim ──
                k_off = MOVE_BASE_DIM + MOVE_EFFECTS_DIM
                if not move_data['is_status'] and opp_active_types:
                    is_stab = mt_name in poke_types
                    stab_mod = 1.5 if is_stab else 1.0
                    vec[idx + k_off + 1] = float(is_stab)

                    if move_data['is_physical']:
                        a_stat, d_stat = my_atk, opp_def_stat
                    else:
                        a_stat, d_stat = my_spa, opp_spd_stat

                    w_mod = _weather_mod_for_type(mt_name, weather_str)
                    est_dmg = _calc_damage_fast(
                        a_stat, d_stat, power, stab_mod, type_eff,
                        weather_mod=w_mod * item_damage_mod,
                        is_burned=(is_burned and move_data['is_physical']),
                    )
                    dmg_pct = min(est_dmg / max(opp_max_hp, 1), 1.0)
                    vec[idx + k_off] = dmg_pct
                    vec[idx + k_off + 2] = float(est_dmg >= opp_cur_hp)

            # PP fraction (Showdown 전용, 마지막 1 dim)
            pp_offset = MOVE_BASE_DIM + MOVE_EFFECTS_DIM + MOVE_KNOWLEDGE_DIM
            if m_i < len(move_slots):
                ms = move_slots[m_i]
                maxpp = ms.get('maxpp', 1)
                vec[idx + pp_offset] = ms.get('pp', 0) / max(maxpp, 1)
        idx += SHOWDOWN_MOVE_DIM

    # ── 아이템 효과 (12) ──────────────────────────────────────
    vec[idx:idx + N_ITEM_FX] = _encode_item_effects(item_id)
    idx += N_ITEM_FX

    # ── 특성 효과 (16) ───────────────────────────────────────
    ability = pdata.get('ability', '')
    ability_id = _to_id(ability) if ability else ''
    vec[idx:idx + N_ABILITY_FX] = _encode_ability_effects(ability_id)
    idx += N_ABILITY_FX

    # ── 매치업 피처 (5) ──────────────────────────────────────
    if opp_si_poke and hp > 0 and opp_si_poke.get('hp', 0) > 0:
        # 실효 스피드 계산 (간이)
        my_spe = stats.get('spe', 100)
        spe_boost = boosts.get('spe', 0)
        if spe_boost > 0:
            my_spe = my_spe * (2 + spe_boost) / 2
        elif spe_boost < 0:
            my_spe = my_spe * 2 / (2 - spe_boost)
        if status == 'par':
            my_spe *= 0.5
        if item_fx_data.get('speed_mod'):
            my_spe *= item_fx_data['speed_mod']
        ability_fx = ABILITY_EFFECTS.get(ability_id, {})
        if ability_fx.get('speed_weather') == weather_str:
            my_spe *= ability_fx.get('speed_mul', 1.0)
        if ability_fx.get('speed_terrain') == terrain_str:
            my_spe *= ability_fx.get('speed_mul', 1.0)

        opp_stats_d = (opp_pdata or {}).get('stats', {})
        opp_boosts_d = opp_si_poke.get('boosts', {})
        opp_spe = opp_stats_d.get('spe', 100)
        opp_spe_b = opp_boosts_d.get('spe', 0)
        if opp_spe_b > 0:
            opp_spe = opp_spe * (2 + opp_spe_b) / 2
        elif opp_spe_b < 0:
            opp_spe = opp_spe * 2 / (2 - opp_spe_b)
        opp_status = opp_si_poke.get('status', '') or ''
        if opp_status == 'par':
            opp_spe *= 0.5
        opp_item_id = _to_id((opp_pdata or {}).get('item', '')) if (opp_pdata or {}).get('item') else ''
        opp_item_fx = ITEM_EFFECTS.get(opp_item_id, {})
        if opp_item_fx.get('speed_mod'):
            opp_spe *= opp_item_fx['speed_mod']
        opp_ability_id = _to_id((opp_pdata or {}).get('ability', '')) if (opp_pdata or {}).get('ability') else ''
        opp_ability_fx = ABILITY_EFFECTS.get(opp_ability_id, {})
        if opp_ability_fx.get('speed_weather') == weather_str:
            opp_spe *= opp_ability_fx.get('speed_mul', 1.0)
        if opp_ability_fx.get('speed_terrain') == terrain_str:
            opp_spe *= opp_ability_fx.get('speed_mul', 1.0)

        vec[idx] = float(my_spe > opp_spe)
        speed_sum = my_spe + opp_spe
        vec[idx + 1] = my_spe / max(speed_sum, 1.0)

        # max_incoming_pct, can_be_ohko, defensive_eff
        max_inc = 0.0
        worst_eff = 0.0
        my_hp_val = max(maxhp, 1)
        my_cur_hp_val = hp
        my_def_val = stats.get('def', 100)
        my_def_b = boosts.get('def', 0)
        if my_def_b > 0:
            my_def_val = my_def_val * (2 + my_def_b) / 2
        elif my_def_b < 0:
            my_def_val = my_def_val * 2 / (2 - my_def_b)
        my_spd_val = stats.get('spd', 100)
        my_spd_b = boosts.get('spd', 0)
        if my_spd_b > 0:
            my_spd_val = my_spd_val * (2 + my_spd_b) / 2
        elif my_spd_b < 0:
            my_spd_val = my_spd_val * 2 / (2 - my_spd_b)

        opp_atk = opp_stats_d.get('atk', 100)
        opp_atk_b = opp_boosts_d.get('atk', 0)
        if opp_atk_b > 0:
            opp_atk = opp_atk * (2 + opp_atk_b) / 2
        elif opp_atk_b < 0:
            opp_atk = opp_atk * 2 / (2 - opp_atk_b)
        opp_spa_v = opp_stats_d.get('spa', 100)
        opp_spa_b = opp_boosts_d.get('spa', 0)
        if opp_spa_b > 0:
            opp_spa_v = opp_spa_v * (2 + opp_spa_b) / 2
        elif opp_spa_b < 0:
            opp_spa_v = opp_spa_v * 2 / (2 - opp_spa_b)
        opp_burned = (opp_status == 'brn')
        opp_item_dmg_mod = opp_item_fx.get('damage_mod', 1.0)
        opp_types = opp_si_poke.get('types', [])

        opp_moves = (opp_pdata or {}).get('moves', [])
        for om_name in opp_moves:
            om = gd.get_move(_to_id(om_name))
            if not om or om['is_status']:
                continue
            o_eff = gd.effectiveness(om['type'], poke_types)
            worst_eff = max(worst_eff, o_eff)
            o_stab = 1.5 if om['type'] in opp_types else 1.0
            if om['is_physical']:
                o_dmg = _calc_damage_fast(
                    opp_atk, my_def_val, om['basePower'], o_stab, o_eff,
                    weather_mod=_weather_mod_for_type(om['type'], weather_str) * opp_item_dmg_mod,
                    is_burned=opp_burned)
            else:
                o_dmg = _calc_damage_fast(
                    opp_spa_v, my_spd_val, om['basePower'], o_stab, o_eff,
                    weather_mod=_weather_mod_for_type(om['type'], weather_str) * opp_item_dmg_mod)
            inc_pct = min(o_dmg / my_hp_val, 1.0)
            max_inc = max(max_inc, inc_pct)

        vec[idx + 2] = max_inc
        vec[idx + 3] = float(max_inc * my_hp_val >= my_cur_hp_val) if my_cur_hp_val > 0 else 0.0
        vec[idx + 4] = worst_eff / 4.0
    idx += MATCHUP_DIM

    return vec


def _encode_sd_side(side_info: dict, tera_used: bool) -> np.ndarray:
    """사이드 컨디션 → (8,) float32."""
    sc = side_info.get('sideConditions', {})
    return np.array([
        float('stealthrock' in sc),
        sc.get('spikes', 0) / 3.0,
        sc.get('toxicspikes', 0) / 2.0,
        float('stickyweb' in sc),
        sc.get('reflect', 0) / 5.0,
        sc.get('lightscreen', 0) / 5.0,
        sc.get('tailwind', 0) / 4.0,
        float(tera_used),
    ], dtype=np.float32)


def _encode_sd_global(field_info: dict, turn: int) -> np.ndarray:
    """글로벌 상태 → (14,) float32."""
    vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    i = 0

    # Weather one-hot (5)
    w = _to_id(field_info.get('weather', '')) if field_info.get('weather') else ''
    w_idx = WEATHER_ID_MAP.get(w, 0)
    vec[i + w_idx] = 1.0
    i += 5
    vec[i] = field_info.get('weatherTurns', 0) / 5.0
    i += 1

    # Terrain one-hot (5)
    t = _to_id(field_info.get('terrain', '')) if field_info.get('terrain') else ''
    t_idx = TERRAIN_ID_MAP.get(t, 0)
    vec[i + t_idx] = 1.0
    i += 5
    vec[i] = field_info.get('terrainTurns', 0) / 5.0
    i += 1

    # Trick room (1)
    vec[i] = float(field_info.get('trickRoom', False))
    i += 1

    # Turn / 50 (1)
    vec[i] = min(turn / 50.0, 1.0)
    i += 1

    return vec


def encode_showdown_state(p1_request: dict, p2_request: dict,
                          state_info: dict, player: int,
                          gd: GameData) -> torch.Tensor:
    """Showdown 데이터 → (1650,) float32 텐서. BattleState 불필요.

    Args:
        p1_request, p2_request: Showdown request JSON (양쪽)
        state_info: extractStateInfo() 결과 (field/sides/turn)
        player: 관점 (0=P1, 1=P2)
        gd: GameData (move type/power 조회용)
    """
    ally_req = p1_request if player == 0 else p2_request
    opp_req = p2_request if player == 0 else p1_request
    ally_side_idx = player
    opp_side_idx = 1 - player

    ally_side_info = state_info['sides'][ally_side_idx]
    opp_side_info = state_info['sides'][opp_side_idx]

    ally_pokemon_data = ally_req.get('side', {}).get('pokemon', [])
    opp_pokemon_data = opp_req.get('side', {}).get('pokemon', [])

    ally_si_pokemon = ally_side_info.get('pokemon', [])
    opp_si_pokemon = opp_side_info.get('pokemon', [])

    # tera_used 판정
    ally_tera_used = any(p.get('terastallized') for p in ally_pokemon_data)
    opp_tera_used = any(p.get('terastallized') for p in opp_pokemon_data)

    # 상대 active 타입 (effectiveness 계산용, index 0 = active)
    opp_active_types = opp_si_pokemon[0].get('types', []) if opp_si_pokemon else []
    ally_active_types = ally_si_pokemon[0].get('types', []) if ally_si_pokemon else []

    # 날씨/필드 문자열
    field_info = state_info.get('field', {})
    w = _to_id(field_info.get('weather', '')) if field_info.get('weather') else ''
    weather_str = w if w in ('sunnyday', 'raindance', 'sandstorm', 'snow', 'hail') else ''
    # sunnyday → sun, raindance → rain 등으로 변환 (battle_sim 호환)
    _w_map = {'sunnyday': 'sun', 'raindance': 'rain', 'sandstorm': 'sand', 'snow': 'snow', 'hail': 'snow'}
    weather_str = _w_map.get(weather_str, '')
    t = _to_id(field_info.get('terrain', '')) if field_info.get('terrain') else ''
    _t_map = {'electricterrain': 'electric', 'grassyterrain': 'grassy', 'mistyterrain': 'misty', 'psychicterrain': 'psychic'}
    terrain_str = _t_map.get(t, '')

    # 상대 active 정보 (매치업 계산용)
    opp_active_si = opp_si_pokemon[0] if opp_si_pokemon else None
    opp_active_pd = opp_pokemon_data[0] if opp_pokemon_data else None
    ally_active_si = ally_si_pokemon[0] if ally_si_pokemon else None
    ally_active_pd = ally_pokemon_data[0] if ally_pokemon_data else None

    parts = []

    # 아군 포켓몬 ×3
    for j in range(TEAM_SIZE):
        if j < len(ally_pokemon_data) and j < len(ally_si_pokemon):
            parts.append(_encode_sd_pokemon(
                ally_pokemon_data[j], ally_si_pokemon[j],
                opp_active_types, gd, not ally_tera_used,
                opp_si_poke=opp_active_si, opp_pdata=opp_active_pd,
                weather_str=weather_str, terrain_str=terrain_str))
        else:
            parts.append(np.zeros(SHOWDOWN_POKEMON_DIM, dtype=np.float32))

    # 상대 포켓몬 ×3
    for j in range(TEAM_SIZE):
        if j < len(opp_pokemon_data) and j < len(opp_si_pokemon):
            parts.append(_encode_sd_pokemon(
                opp_pokemon_data[j], opp_si_pokemon[j],
                ally_active_types, gd, not opp_tera_used,
                opp_si_poke=ally_active_si, opp_pdata=ally_active_pd,
                weather_str=weather_str, terrain_str=terrain_str))
        else:
            parts.append(np.zeros(SHOWDOWN_POKEMON_DIM, dtype=np.float32))

    # 사이드 컨디션
    parts.append(_encode_sd_side(ally_side_info, ally_tera_used))
    parts.append(_encode_sd_side(opp_side_info, opp_tera_used))

    # 글로벌
    parts.append(_encode_sd_global(field_info, state_info.get('turn', 0)))

    return torch.from_numpy(np.concatenate(parts))


# ═══════════════════════════════════════════════════════════════
#  ShowdownEvaluator — 네트워크 평가 래퍼
# ═══════════════════════════════════════════════════════════════

class ShowdownEvaluator:
    """Showdown 데이터 직접 → (priors, value). BattleState 불필요."""

    def __init__(self, model, game_data: GameData,
                 device: torch.device | str = "cpu"):
        self.model = model
        self.gd = game_data
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, p1_req: dict, p2_req: dict, state_info: dict,
                 player: int,
                 legal_actions_str: list[str]) -> tuple[dict[str, float], float]:
        """Showdown 데이터 → (priors, value).

        Returns:
            priors: {action_str: probability} — 합계 1.0
            value: float in [-1, +1]
        """
        x = encode_showdown_state(
            p1_req, p2_req, state_info, player, self.gd)
        logits, v = self.model(x.to(self.device))
        logits = logits.squeeze(0)
        value = v.item()

        # 마스킹: 불법 액션 -inf
        mask = torch.full((NUM_ACTIONS,), float('-inf'), device=self.device)
        str_to_int = {}
        for s in legal_actions_str:
            int_a = str_to_action_type(s)
            if int_a is not None and int_a < NUM_ACTIONS:
                mask[int_a] = 0.0
                str_to_int[s] = int_a

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=0).cpu().numpy()

        priors = {}
        for s, int_a in str_to_int.items():
            priors[s] = float(probs[int_a])
        # 매핑 안 된 액션은 균등
        for s in legal_actions_str:
            if s not in priors:
                priors[s] = 1.0 / len(legal_actions_str)

        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        return priors, value

    @torch.no_grad()
    def evaluate_value(self, p1_req: dict, p2_req: dict,
                       state_info: dict, player: int) -> float:
        """Value만 반환."""
        x = encode_showdown_state(
            p1_req, p2_req, state_info, player, self.gd)
        _, v = self.model(x.to(self.device))
        return v.item()


# ═══════════════════════════════════════════════════════════════
#  액션 변환 (standalone 함수)
# ═══════════════════════════════════════════════════════════════

def str_to_action_type(action_str: str) -> int | None:
    """Showdown 액션 문자열 → ActionType int.

    "move 1" → 0, "move 2" → 1, ..., "switch 1" → 4, ...
    "move 1 terastallize" → 9, ...
    """
    parts = action_str.split()
    if not parts:
        return None
    if parts[0] == "move" and len(parts) >= 2:
        try:
            idx = int(parts[1]) - 1
        except ValueError:
            return None
        if "terastallize" in parts:
            return ActionType.TERA_MOVE1 + idx if idx < 4 else None
        return ActionType.MOVE1 + idx if idx < 4 else None
    elif parts[0] == "switch" and len(parts) >= 2:
        try:
            idx = int(parts[1]) - 1
        except ValueError:
            return None
        return ActionType.SWITCH1 + idx if idx < 5 else None
    return None


def action_type_to_str(action: int) -> str:
    """ActionType int → Showdown 액션 문자열."""
    if action >= ActionType.TERA_MOVE1:
        idx = action - ActionType.TERA_MOVE1
        return f"move {idx + 1} terastallize"
    elif action >= ActionType.SWITCH1:
        idx = action - ActionType.SWITCH1
        return f"switch {idx + 1}"
    else:
        idx = action - ActionType.MOVE1
        return f"move {idx + 1}"


# ═══════════════════════════════════════════════════════════════
#  Showdown MCTS 노드
# ═══════════════════════════════════════════════════════════════

@dataclass
class ShowdownMCTSNode:
    """Showdown MCTS 트리 노드.

    Max 노드: root_player가 행동 → UCB 최대화
    Min 노드: 상대가 행동 → UCB 최소화
    """
    battle_id: str
    acting_player: int
    is_max: bool
    parent: Optional[ShowdownMCTSNode] = None
    action_from_parent: str = ""
    children: dict[str, ShowdownMCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0

    # 2-ply: Max → Min으로 넘어갈 때 Max의 선택을 기억
    pending_action: str = ""

    # 캐시
    _p1_request: dict = field(default=None, repr=False)
    _p2_request: dict = field(default=None, repr=False)
    _state_info: dict = field(default=None, repr=False)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_max(self, c: float = 1.414, pw: float = 2.0) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        return (self.q_value
                + c * math.sqrt(math.log(parent_n) / self.visit_count)
                + pw * self.prior / (1 + self.visit_count))

    def ucb1_min(self, c: float = 1.414, pw: float = 2.0) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        return (-self.q_value
                + c * math.sqrt(math.log(parent_n) / self.visit_count)
                + pw * self.prior / (1 + self.visit_count))


# ═══════════════════════════════════════════════════════════════
#  Showdown MCTS 엔진
# ═══════════════════════════════════════════════════════════════

class ShowdownMCTS:
    """Showdown 엔진 기반 2-Player MCTS."""

    def __init__(
        self,
        bridge: ShowdownBridge,
        game_data: GameData,
        network_evaluator=None,
        n_simulations: int = 200,
        exploration_weight: float = 1.414,
        prior_weight: float = 2.0,
        max_opp_branches: int = 5,
        dirichlet_alpha: float = 0.0,
        dirichlet_weight: float = 0.25,
        format_name: str = "bss",
    ):
        self.bridge = bridge
        self.gd = game_data
        self.network = network_evaluator  # ShowdownEvaluator or None
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        self.prior_weight = prior_weight
        self.max_opp_branches = max_opp_branches
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.format_name = format_name

        self.root_player = 0
        self._forked_ids: list[str] = []

    # ─── 메인 탐색 ───────────────────────────────────────────

    def search(self, root_battle_id: str, player: int,
               p1_request: dict = None, p2_request: dict = None,
               state_info: dict = None,
               n_simulations: int | None = None) -> dict[str, int]:
        """MCTS 탐색 → {action_str: visit_count}.

        Args:
            root_battle_id: 현재 배틀 ID (원본 유지, fork 안 함)
            player: 탐색 주체 (0 or 1)
            p1_request: 캐시된 p1 request
            p2_request: 캐시된 p2 request
            state_info: extractStateInfo 결과 (인코딩용)
            n_simulations: 시뮬레이션 횟수
        """
        n_sims = n_simulations or self.n_simulations
        self.root_player = player
        self._forked_ids = []

        root = ShowdownMCTSNode(
            battle_id=root_battle_id,
            acting_player=player,
            is_max=True,
        )
        if p1_request:
            root._p1_request = p1_request
        if p2_request:
            root._p2_request = p2_request
        if state_info:
            root._state_info = state_info

        self._expand(root)

        # Dirichlet 노이즈
        if self.dirichlet_alpha > 0 and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children))
            eps = self.dirichlet_weight
            for child, eta in zip(root.children.values(), noise):
                child.prior = (1 - eps) * child.prior + eps * eta

        for _ in range(n_sims):
            node = self._select(root)
            if not node.is_terminal and not node.is_expanded:
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))
            value = self._simulate(node)
            self._backpropagate(node, value)

        result = {a: c.visit_count for a, c in root.children.items()}
        self._cleanup_forks()
        return result

    def get_action_probs(self, root_battle_id: str, player: int,
                         temperature: float = 1.0,
                         **kwargs) -> dict[str, float]:
        """탐색 → action별 확률 분포 (학습용)."""
        visits = self.search(root_battle_id, player, **kwargs)
        if not visits:
            return {}
        if temperature == 0:
            best = max(visits, key=visits.get)
            return {a: 1.0 if a == best else 0.0 for a in visits}
        total = sum(v ** (1.0 / temperature) for v in visits.values())
        if total == 0:
            return {a: 1.0 / len(visits) for a in visits}
        return {a: (v ** (1.0 / temperature)) / total
                for a, v in visits.items()}

    # ─── Selection ───────────────────────────────────────────

    def _select(self, node: ShowdownMCTSNode) -> ShowdownMCTSNode:
        c = self.exploration_weight
        pw = self.prior_weight
        while node.is_expanded and node.children and not node.is_terminal:
            if node.is_max:
                node = max(node.children.values(),
                           key=lambda n: n.ucb1_max(c, pw))
            else:
                node = max(node.children.values(),
                           key=lambda n: n.ucb1_min(c, pw))
        return node

    # ─── Expansion ───────────────────────────────────────────

    def _expand(self, node: ShowdownMCTSNode):
        """2-ply 확장.

        Max 노드 → 아군 액션별 자식 (Min 노드, 같은 battle_id)
        Min 노드 → 상대 액션별 자식 (fork + step, 새 battle_id)
        """
        if node.is_terminal:
            node.is_expanded = True
            return

        request = self._get_request(node, node.acting_player)
        if not request or request.get("wait"):
            node.is_expanded = True
            return

        if request.get("forceSwitch"):
            legal = self._parse_force_switch(request)
        else:
            legal = self._parse_legal_actions(request)

        if not legal:
            node.is_expanded = True
            return

        priors = self._compute_priors(node, legal)

        if node.is_max:
            for action in legal:
                child = ShowdownMCTSNode(
                    battle_id=node.battle_id,
                    acting_player=1 - self.root_player,
                    is_max=False,
                    parent=node,
                    action_from_parent=action,
                    prior=priors.get(action, 1.0 / len(legal)),
                    pending_action=action,
                )
                child._p1_request = node._p1_request
                child._p2_request = node._p2_request
                child._state_info = node._state_info
                node.children[action] = child
        else:
            our_action = node.pending_action
            sorted_actions = sorted(
                legal, key=lambda a: priors.get(a, 0), reverse=True)
            actions_to_use = sorted_actions[:self.max_opp_branches]

            for opp_action in actions_to_use:
                fork_resp = self.bridge.fork(node.battle_id)
                if "error" in fork_resp:
                    continue
                new_id = fork_resp["new_battle_id"]
                self._forked_ids.append(new_id)

                if self.root_player == 0:
                    p1c, p2c = our_action, opp_action
                else:
                    p1c, p2c = opp_action, our_action

                step_resp = self.bridge.step(new_id, p1c, p2c)

                child = ShowdownMCTSNode(
                    battle_id=new_id,
                    acting_player=self.root_player,
                    is_max=True,
                    parent=node,
                    action_from_parent=opp_action,
                    prior=priors.get(opp_action, 1.0 / len(legal)),
                )
                child._p1_request = step_resp.get("p1request")
                child._p2_request = step_resp.get("p2request")
                child._state_info = step_resp.get("state_info")

                if step_resp.get("ended"):
                    child.is_terminal = True
                    winner = step_resp.get("winner", "")
                    if winner == f"Player {self.root_player + 1}":
                        child.terminal_value = 1.0
                    elif winner:
                        child.terminal_value = 0.0
                    else:
                        child.terminal_value = 0.5

                # forceSwitch 자동 처리
                my_req = step_resp.get(
                    f"p{self.root_player + 1}request")
                if my_req and my_req.get("forceSwitch") and not child.is_terminal:
                    fs_resp = self._handle_force_switch(
                        new_id, my_req, self.root_player)
                    child._p1_request = fs_resp.get("p1request")
                    child._p2_request = fs_resp.get("p2request")
                    child._state_info = fs_resp.get("state_info")

                node.children[opp_action] = child

        node.is_expanded = True

    def _handle_force_switch(self, battle_id: str, request: dict,
                             player: int) -> dict:
        """forceSwitch 자동 처리 → step 응답 반환."""
        switch_action = self._pick_force_switch(request)
        opp_req = self.bridge.get_request(battle_id, 1 - player)

        if opp_req and opp_req.get("forceSwitch"):
            opp_action = self._pick_force_switch(opp_req)
        else:
            opp_action = "default"

        if player == 0:
            return self.bridge.step(battle_id, switch_action, opp_action)
        else:
            return self.bridge.step(battle_id, opp_action, switch_action)

    def _pick_force_switch(self, request: dict) -> str:
        side = request.get("side", {})
        for i, poke in enumerate(side.get("pokemon", [])):
            if not poke.get("active") and poke.get("condition", "0 fnt") != "0 fnt":
                return f"switch {i + 1}"
        return "switch 1"

    # ─── Simulation (Value 평가) ─────────────────────────────

    def _simulate(self, node: ShowdownMCTSNode) -> float:
        """노드 value 평가 → root_player 관점 [0, 1]."""
        if node.is_terminal:
            return node.terminal_value

        if self.network and node._state_info:
            p1r = self._get_request(node, 0)
            p2r = self._get_request(node, 1)
            if p1r and p2r:
                value = self.network.evaluate_value(
                    p1r, p2r, node._state_info, self.root_player)
                return (value + 1.0) / 2.0  # [-1,+1] → [0,1]

        return self._heuristic_eval(node)

    def _heuristic_eval(self, node: ShowdownMCTSNode) -> float:
        """HP 기반 간이 평가 (네트워크 없을 때 폴백)."""
        my_req = self._get_request(node, self.root_player)
        opp_req = self._get_request(node, 1 - self.root_player)
        if not my_req or not opp_req:
            return 0.5

        my_hp = self._total_hp_fraction(my_req)
        opp_hp = self._total_hp_fraction(opp_req)
        diff = my_hp - opp_hp
        return max(0.0, min(1.0, 0.5 + diff / 12.0))

    def _total_hp_fraction(self, request: dict) -> float:
        total = 0.0
        side = request.get("side", {})
        for poke in side.get("pokemon", []):
            cond = poke.get("condition", "0 fnt")
            if cond == "0 fnt":
                continue
            parts = cond.split("/")
            if len(parts) == 2:
                cur = int(parts[0].split()[0])
                maxhp = int(parts[1].split()[0])
                total += cur / max(maxhp, 1)
        return total

    # ─── Backpropagation ─────────────────────────────────────

    def _backpropagate(self, node: ShowdownMCTSNode, value: float):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    # ─── Prior 계산 ──────────────────────────────────────────

    def _compute_priors(self, node: ShowdownMCTSNode,
                        legal_actions: list[str]) -> dict[str, float]:
        """Network 있으면 Policy Net, 없으면 균등."""
        if self.network and node._state_info:
            p1r = self._get_request(node, 0)
            p2r = self._get_request(node, 1)
            if p1r and p2r:
                priors, _ = self.network.evaluate(
                    p1r, p2r, node._state_info,
                    node.acting_player, legal_actions)
                return priors

        return {a: 1.0 / len(legal_actions) for a in legal_actions}

    # ─── Request 파싱 ────────────────────────────────────────

    def _parse_legal_actions(self, request: dict) -> list[str]:
        actions = []
        if "active" in request:
            active = request["active"][0]
            for i, move in enumerate(active.get("moves", [])):
                if not move.get("disabled"):
                    actions.append(f"move {i + 1}")
            if active.get("canTerastallize"):
                for i, move in enumerate(active.get("moves", [])):
                    if not move.get("disabled"):
                        actions.append(f"move {i + 1} terastallize")

        if "side" in request:
            for i, poke in enumerate(request["side"]["pokemon"]):
                if not poke.get("active") and \
                   poke.get("condition", "0 fnt") != "0 fnt":
                    actions.append(f"switch {i + 1}")

        return actions if actions else ["default"]

    def _parse_force_switch(self, request: dict) -> list[str]:
        actions = []
        if "side" in request:
            for i, poke in enumerate(request["side"]["pokemon"]):
                if not poke.get("active") and \
                   poke.get("condition", "0 fnt") != "0 fnt":
                    actions.append(f"switch {i + 1}")
        return actions if actions else ["switch 1"]

    def _get_request(self, node: ShowdownMCTSNode,
                     player: int) -> dict | None:
        if player == 0 and node._p1_request:
            return node._p1_request
        if player == 1 and node._p2_request:
            return node._p2_request

        req = self.bridge.get_request(node.battle_id, player)
        if player == 0:
            node._p1_request = req
        else:
            node._p2_request = req
        return req

    # ─── 메모리 관리 ─────────────────────────────────────────

    def _cleanup_forks(self):
        for bid in self._forked_ids:
            try:
                self.bridge.destroy(bid)
            except Exception:
                pass
        self._forked_ids.clear()


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """ShowdownMCTS + encode_showdown_state 검증."""
    print("=== ShowdownMCTS 검증 ===\n")

    gd = GameData(device="cpu")
    bridge = ShowdownBridge()

    # 1. 인코딩 테스트
    print("--- 테스트 1: encode_showdown_state ---")
    resp = bridge.init_battle(format_id="gen9randombattle")
    bid = resp["battle_id"]
    si = resp.get("state_info")
    p1r = resp.get("p1request")
    p2r = resp.get("p2request")

    x0 = encode_showdown_state(p1r, p2r, si, 0, gd)
    x1 = encode_showdown_state(p1r, p2r, si, 1, gd)
    print(f"P0 관점: shape={x0.shape}, min={x0.min():.3f}, "
          f"max={x0.max():.3f}, mean={x0.mean():.3f}")
    print(f"P1 관점: shape={x1.shape}, min={x1.min():.3f}, "
          f"max={x1.max():.3f}, mean={x1.mean():.3f}")
    assert x0.shape == (SHOWDOWN_STATE_DIM,), \
        f"Expected ({SHOWDOWN_STATE_DIM},), got {x0.shape}"
    bridge.destroy(bid)

    # 2. MCTS 탐색 (heuristic 평가)
    print("\n--- 테스트 2: MCTS 탐색 (50 sims, heuristic) ---")
    mcts = ShowdownMCTS(
        bridge=bridge, game_data=gd,
        network_evaluator=None,
        n_simulations=50,
        max_opp_branches=3,
    )
    resp = bridge.init_battle(format_id="gen9randombattle")
    bid = resp["battle_id"]
    visits = mcts.search(
        bid, player=0,
        p1_request=resp["p1request"],
        p2_request=resp["p2request"],
        state_info=resp.get("state_info"),
    )
    total_v = sum(visits.values())
    for action, count in sorted(visits.items(), key=lambda x: -x[1]):
        print(f"  {action:30s} | {count:4d} ({count/total_v:.1%})")
    bridge.destroy(bid)

    # 3. 전체 게임 (MCTS vs first-move)
    print("\n--- 테스트 3: MCTS vs First-Move 게임 ---")
    resp = bridge.init_battle(format_id="gen9randombattle")
    game_id = resp["battle_id"]
    turns = 0

    while not resp.get("ended", False) and turns < 50:
        p1r = resp.get("p1request") or bridge.get_request(game_id, 0)
        p2r = resp.get("p2request") or bridge.get_request(game_id, 1)
        si = resp.get("state_info")

        if not p1r or not p2r:
            break

        if p1r.get("forceSwitch"):
            p1c = _pick_first_valid_switch(p1r)
        elif p1r.get("wait"):
            p1c = "default"
        else:
            visits = mcts.search(game_id, player=0,
                                 p1_request=p1r, p2_request=p2r,
                                 state_info=si,
                                 n_simulations=30)
            p1c = max(visits, key=visits.get) if visits else "default"

        if p2r.get("forceSwitch"):
            p2c = _pick_first_valid_switch(p2r)
        elif p2r.get("wait"):
            p2c = "default"
        else:
            p2c = _pick_first_action(p2r)

        resp = bridge.step(game_id, p1c, p2c)
        turns += 1

    print(f"게임 결과: {turns}턴, ended={resp.get('ended')}, "
          f"winner='{resp.get('winner', '')}'")
    bridge.destroy(game_id)
    bridge.close()
    print("\n검증 완료!")


def _pick_first_action(request: dict) -> str:
    if "active" in request:
        for i, m in enumerate(request["active"][0].get("moves", [])):
            if not m.get("disabled"):
                return f"move {i + 1}"
    return "default"


def _pick_first_valid_switch(request: dict) -> str:
    side = request.get("side", {})
    for i, poke in enumerate(side.get("pokemon", [])):
        if not poke.get("active") and poke.get("condition", "0 fnt") != "0 fnt":
            return f"switch {i + 1}"
    return "switch 1"


if __name__ == "__main__":
    verify()
