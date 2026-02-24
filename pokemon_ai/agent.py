"""poke-env MCTSPlayer — Showdown 서버 연동.

패킷 흐름:
  Showdown 서버 ←WS→ poke-env (패킷 파싱 → Battle 객체)
                         ↓
                  choose_move(battle)
                         ↓
              _battle_to_state() → 내부 BattleState
                         ↓
                    MCTS search (자체 경량 시뮬레이터)
                         ↓
              _action_to_order() → BattleOrder
                         ↓
                  poke-env → Showdown 서버
"""

from __future__ import annotations

import random
import time
from typing import Optional, Union

import numpy as np
import torch

# ── poke-env 임포트 (없으면 오프라인 모드) ───────────────────
try:
    from poke_env.player import Player
    from poke_env.player.battle_order import BattleOrder
    from poke_env.battle import (
        Battle, Pokemon as PEPokemon, Move as PEMove,
    )
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status as PEStatus
    from poke_env.battle.weather import Weather as PEWeather
    from poke_env.battle.field import Field as PEField
    from poke_env.battle.side_condition import SideCondition as PESideCondition
    from poke_env.battle.effect import Effect as PEEffect
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    class Player:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def create_order(self, *args, **kwargs): return "/choose default"
        def choose_default_move(self, *args, **kwargs): return "/choose default"

from data_loader import GameData, _to_id, STAT_KEYS, TYPE_TO_IDX, NATURES
from damage_calc import DamageCalculator
from battle_sim import (
    BattleSimulator, BattleState, Side, Pokemon, ActionType,
    Weather, Terrain, make_pokemon_from_stats, make_pokemon_from_set_dict,
)
from mcts import MCTS
from team_builder import TeamBuilder
from set_db import SetInferencer


# ═══════════════════════════════════════════════════════════════
#  poke-env enum → 내부 enum 매핑
# ═══════════════════════════════════════════════════════════════

if POKE_ENV_AVAILABLE:
    _PE_WEATHER_MAP = {
        PEWeather.SUNNYDAY: Weather.SUN,
        PEWeather.DESOLATELAND: Weather.SUN,
        PEWeather.RAINDANCE: Weather.RAIN,
        PEWeather.PRIMORDIALSEA: Weather.RAIN,
        PEWeather.SANDSTORM: Weather.SAND,
    }
    # snow/hail은 버전마다 이름이 다를 수 있음
    for _name in ("SNOW", "SNOWSCAPE", "HAIL"):
        if hasattr(PEWeather, _name):
            _PE_WEATHER_MAP[getattr(PEWeather, _name)] = Weather.SNOW

    _PE_TERRAIN_MAP = {
        PEField.ELECTRIC_TERRAIN: Terrain.ELECTRIC,
        PEField.GRASSY_TERRAIN: Terrain.GRASSY,
        PEField.MISTY_TERRAIN: Terrain.MISTY,
        PEField.PSYCHIC_TERRAIN: Terrain.PSYCHIC,
    }

    _PE_STATUS_MAP = {
        PEStatus.BRN: "brn",
        PEStatus.PAR: "par",
        PEStatus.PSN: "psn",
        PEStatus.TOX: "tox",
        PEStatus.SLP: "slp",
        PEStatus.FRZ: "frz",
    }

    # PokemonType.FIRE → "Fire"
    def _pe_type_to_str(t: PokemonType) -> str:
        return t.name.capitalize()
else:
    _PE_WEATHER_MAP = {}
    _PE_TERRAIN_MAP = {}
    _PE_STATUS_MAP = {}
    def _pe_type_to_str(t) -> str:
        return str(t).split(".")[-1].capitalize()


# ═══════════════════════════════════════════════════════════════
#  팀 → Showdown paste 포맷 변환
# ═══════════════════════════════════════════════════════════════

def team_to_showdown_paste(gd: GameData, species_list: list[str],
                           format_name: str = "bss", level: int = 50) -> str:
    """Smogon 통계 기반 세트를 Showdown 팀 문자열로 변환."""
    lines = []
    used_items: set[str] = set()  # Item Clause: 아이템 중복 방지

    for species in species_list:
        pid = _to_id(species)
        usage = gd.get_stats(species, format_name)
        dex = gd.get_pokemon(species)
        if not dex:
            continue

        # 아이템 (중복 방지)
        item = ""
        if usage:
            items = sorted(usage.get("items", {}).items(), key=lambda x: -x[1])
            for item_id, _ in items:
                item_name = _item_id_to_name(gd, item_id)
                if item_id not in used_items:
                    item = item_name
                    used_items.add(item_id)
                    break

        # 특성 — dex 이름 우선 (Showdown이 인식하는 형태)
        ability = dex["abilities"][0] if dex["abilities"] else ""
        if usage:
            abs_sorted = sorted(usage.get("abilities", {}).items(), key=lambda x: -x[1])
            if abs_sorted:
                # dex에 있는 정식 이름 중 id 매칭되는 것 사용
                best_ab_id = abs_sorted[0][0]
                matched = [a for a in dex["abilities"]
                           if _to_id(a) == best_ab_id]
                if matched:
                    ability = matched[0]
                else:
                    ability = _ability_id_to_name(gd, best_ab_id) or ability

        # 성격/EV
        nature = "Adamant"
        evs = {k: 0 for k in STAT_KEYS}
        if usage and usage.get("spreads"):
            best = max(usage["spreads"], key=lambda x: x["weight"])
            nature = best["nature"]
            evs = best["evs"]

        # 기술 (빈 문자열/None 필터링)
        moves = []
        if usage:
            m_sorted = sorted(usage.get("moves", {}).items(), key=lambda x: -x[1])
            for mid, _ in m_sorted:
                if not mid or mid.strip() == "":
                    continue
                move_data = gd.get_move(mid)
                if move_data and move_data.get("name"):
                    moves.append(move_data["name"])
                elif mid.strip():
                    moves.append(mid)
                if len(moves) >= 4:
                    break

        # 기술 없으면 이 포켓몬 건너뜀 (poke-env 파싱 에러 방지)
        if not moves:
            continue

        # 테라 타입: 특수 포켓몬은 고정 테라타입
        _FIXED_TERA = {
            "terapagos": "Stellar",
            "ogerponwellspring": "Water",
            "ogerponhearthflame": "Fire",
            "ogerponcornerstone": "Rock",
            "ogerpon": "Grass",
        }
        fixed_tera = _FIXED_TERA.get(_to_id(dex["name"]))
        if fixed_tera:
            tera_type = fixed_tera
        elif usage and usage.get("tera_types"):
            # Smogon 통계에서 가장 인기 있는 테라 타입 사용
            top_tera = sorted(usage["tera_types"].items(),
                              key=lambda x: -x[1])
            if top_tera:
                tera_type = top_tera[0][0].capitalize()
            else:
                tera_type = dex["types"][0] if dex["types"] else "Normal"
        else:
            tera_type = dex["types"][0] if dex["types"] else "Normal"

        # Showdown paste 형식
        header = f"{dex['name']} @ {item}" if item else dex['name']
        lines.append(header)
        lines.append(f"Ability: {ability}")
        lines.append(f"Tera Type: {tera_type}")
        lines.append(f"Level: {level}")

        ev_parts = []
        for stat_name, stat_key in [("HP", "hp"), ("Atk", "atk"), ("Def", "def"),
                                     ("SpA", "spa"), ("SpD", "spd"), ("Spe", "spe")]:
            val = evs.get(stat_key, 0)
            if val > 0:
                ev_parts.append(f"{val} {stat_name}")
        if ev_parts:
            lines.append(f"EVs: {' / '.join(ev_parts)}")

        lines.append(f"{nature} Nature")
        lines.append(f"IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe")

        for m in moves:
            lines.append(f"- {m}")
        lines.append("")  # 빈 줄로 구분

    return "\n".join(lines)


def _item_id_to_name(gd: GameData, item_id: str) -> str:
    """아이템 ID → 표시 이름. 간단한 매핑."""
    _common = {
        "choicescarf": "Choice Scarf", "choiceband": "Choice Band",
        "choicespecs": "Choice Specs", "lifeorb": "Life Orb",
        "leftovers": "Leftovers", "focussash": "Focus Sash",
        "assaultvest": "Assault Vest", "heavydutyboots": "Heavy-Duty Boots",
        "rockyhelmet": "Rocky Helmet", "boosterenergy": "Booster Energy",
        "sitrusberry": "Sitrus Berry", "lumberry": "Lum Berry",
        "eviolite": "Eviolite", "expertbelt": "Expert Belt",
        "weaknesspolicy": "Weakness Policy", "clearamulet": "Clear Amulet",
        "covertcloak": "Covert Cloak", "safetygoggles": "Safety Goggles",
        "mentalherb": "Mental Herb", "redcard": "Red Card",
        "ejectbutton": "Eject Button", "airballoon": "Air Balloon",
        "scopelens": "Scope Lens", "widelens": "Wide Lens",
        "loadeddice": "Loaded Dice", "mirrorherb": "Mirror Herb",
        "throatspray": "Throat Spray", "shellbell": "Shell Bell",
        "blacksludge": "Black Sludge", "punchingglove": "Punching Glove",
        "fairyfeather": "Fairy Feather", "custapberry": "Custap Berry",
        "figyberry": "Figy Berry", "wiseglasses": "Wise Glasses",
        "pixieplate": "Pixie Plate", "spelltag": "Spell Tag",
        "grassyseed": "Grassy Seed", "electricseed": "Electric Seed",
        "aguavberry": "Aguav Berry", "iapapaberry": "Iapapa Berry",
        "ironball": "Iron Ball", "roomservice": "Room Service",
        "lightclay": "Light Clay", "terrainextender": "Terrain Extender",
        "protectivepads": "Protective Pads", "utilityumbrella": "Utility Umbrella",
        "muscleband": "Muscle Band", "chopleberry": "Chople Berry",
        "babiriberry": "Babiri Berry", "marangaberry": "Maranga Berry",
        "chestoberry": "Chesto Berry", "oranberry": "Oran Berry",
    }
    return _common.get(item_id, item_id.replace("_", " ").title())


def _ability_id_to_name(gd: GameData, ability_id: str) -> str:
    """특성 ID → 표시 이름."""
    _common = {
        "orichalcumpulse": "Orichalcum Pulse",
        "hadronengine": "Hadron Engine",
        "protosynthesis": "Protosynthesis",
        "quarkdrive": "Quark Drive",
        "vesselofruin": "Vessel of Ruin",
        "tabletsofruin": "Tablets of Ruin",
        "swordofruin": "Sword of Ruin",
        "beadsofruin": "Beads of Ruin",
        "intimidate": "Intimidate",
        "regenerator": "Regenerator",
        "multiscale": "Multiscale",
        "adaptability": "Adaptability",
        "hugepower": "Huge Power",
        "levitate": "Levitate",
        "flashfire": "Flash Fire",
        "drizzle": "Drizzle",
        "drought": "Drought",
        "sandstream": "Sand Stream",
        "snowwarning": "Snow Warning",
        "electricsurge": "Electric Surge",
        "grassysurge": "Grassy Surge",
        "mistysurge": "Misty Surge",
        "psychicsurge": "Psychic Surge",
        "goodasgold": "Good as Gold",
        "unaware": "Unaware",
        "magicbounce": "Magic Bounce",
        "moldbreaker": "Mold Breaker",
        "sturdy": "Sturdy",
        "naturalcure": "Natural Cure",
        "magicguard": "Magic Guard",
        "thickfat": "Thick Fat",
        "icescales": "Ice Scales",
        "serenegrace": "Serene Grace",
        "technician": "Technician",
        "shedskin": "Shed Skin",
    }
    return _common.get(ability_id, ability_id.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════
#  MCTSPlayer (Showdown 연동)
# ═══════════════════════════════════════════════════════════════

class MCTSPlayer(Player):
    """MCTS 기반 포켓몬 배틀 AI — Showdown 패킷을 받아 MCTS 돌림.

    흐름:
      1. Showdown이 WS 패킷 전송
      2. poke-env가 파싱 → Battle 객체로 구조화
      3. choose_move()가 호출됨
      4. Battle → 내부 BattleState 변환
      5. MCTS 탐색 (자체 경량 시뮬레이터에서 수백 회 롤아웃)
      6. 최적 수 → BattleOrder로 변환 → Showdown에 전송
    """

    def __init__(
        self,
        game_data: GameData,
        format_name: str = "bss",
        n_simulations: int = 400,
        time_limit: float = 10.0,
        log_decisions: bool = True,
        preview_checkpoint_path: str | None = None,
        network_evaluator=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gd = game_data
        self.format_name = format_name
        self.sim = BattleSimulator(game_data)
        self.mcts = MCTS(
            game_data=game_data,
            simulator=self.sim,
            n_simulations=n_simulations,
            rollout_depth=30,
            n_rollout_per_leaf=2,
            use_parallel=True,
            n_workers=4,
            format_name=format_name,
            max_opp_branches=5,
            network_evaluator=network_evaluator,
        )
        self.tb = TeamBuilder(game_data, format_name)
        self.set_inferencer = SetInferencer(game_data)
        self.time_limit = time_limit
        self.log_decisions = log_decisions
        self.decision_log: list[dict] = []

        # 프리뷰 네트워크 (선택적)
        self.preview_eval = None
        if preview_checkpoint_path:
            try:
                from neural_net import TeamPreviewNet, PreviewEvaluator
                p_model = TeamPreviewNet()
                p_ckpt = torch.load(
                    preview_checkpoint_path, map_location="cpu",
                    weights_only=False)
                if "model_state_dict" in p_ckpt:
                    p_model.load_state_dict(p_ckpt["model_state_dict"])
                else:
                    p_model.load_state_dict(p_ckpt)
                p_model.eval()
                self.preview_eval = PreviewEvaluator(
                    p_model, game_data, device="cpu")
                if self.log_decisions:
                    print(f"[프리뷰] 신경망 로드: {preview_checkpoint_path}")
            except Exception as e:
                if self.log_decisions:
                    print(f"[프리뷰] 신경망 로드 실패: {e}, 룰 기반 사용")

    # ─── 팀 프리뷰 ──────────────────────────────────────────
    def teampreview(self, battle: Battle) -> str:
        """팀 프리뷰 — 상대 보고 최적 선출 결정.

        preview_eval이 있으면 신경망 사용, 없으면 룰 기반 폴백.
        """
        # 새 배틀 시작 시 세트 추론기 초기화
        self.set_inferencer.reset()

        my_team = list(battle.team.values())
        sim_team = [self._pe_pokemon_to_sim(p) for p in my_team]

        opp_names = [p.species for p in battle.opponent_team.values()]

        # 신경망 프리뷰 시도
        if self.preview_eval and len(sim_team) == 6:
            opp_sim = []
            for name in opp_names:
                try:
                    poke = make_pokemon_from_stats(
                        self.gd, name, self.format_name)
                    opp_sim.append(poke)
                except (ValueError, KeyError):
                    continue

            if len(opp_sim) >= 6:
                leads = self.preview_eval.choose(
                    sim_team, opp_sim, temperature=0.1)
                pick_size = 3 if self.format_name == "bss" else 4
                order = leads[:pick_size]
                remaining = [i for i in range(len(sim_team))
                             if i not in order]
                full_order = order + remaining

                if self.log_decisions:
                    print(f"[팀프리뷰/NN] 상대: {opp_names}")
                    print(f"[팀프리뷰/NN] 선출: "
                          f"{[sim_team[i].name for i in order]}")

                return "/team " + "".join(str(i + 1) for i in full_order)

        # 룰 기반 폴백
        leads = self.tb.choose_leads(sim_team, opp_names)
        pick_size = 3 if self.format_name == "bss" else 4
        order = leads[:pick_size]
        remaining = [i for i in range(len(sim_team)) if i not in order]
        full_order = order + remaining

        if self.log_decisions:
            print(f"[팀프리뷰] 상대: {opp_names}")
            print(f"[팀프리뷰] 선출: {[sim_team[i].name for i in order]}")

        return "/team " + "".join(str(i + 1) for i in full_order)

    # ─── 매 턴 행동 ──────────────────────────────────────────
    def choose_move(self, battle: Battle) -> BattleOrder:
        """Showdown 패킷 → 내부 상태 → MCTS → BattleOrder."""
        t0 = time.time()

        # 1. poke-env Battle → 내부 BattleState
        state = self._battle_to_state(battle)

        # 2. MCTS 탐색
        search_info = self.mcts.get_search_info(
            state, 0,
            n_simulations=self.mcts.n_simulations,
            time_limit=self.time_limit,
        )

        # 3. 최적 액션 선택
        if not search_info["actions"]:
            return self.choose_default_move(battle)

        best = max(search_info["actions"], key=lambda x: x["visits"])
        elapsed = time.time() - t0

        # 4. 로깅
        if self.log_decisions:
            print(f"[턴 {battle.turn:2d}] {search_info['active_pokemon']:15s} "
                  f"→ {best['name']:25s} ({best['probability']:.0%}, "
                  f"{best['visits']}회, {elapsed:.1f}s)")
            for a in search_info["actions"][:5]:
                print(f"         {a['name']:25s} {a['visits']:4d}회 "
                      f"({a['probability']:.0%})")

        self.decision_log.append({
            "turn": battle.turn,
            "pokemon": search_info["active_pokemon"],
            "action": best["name"],
            "action_id": best["action"],
            "visits": best["visits"],
            "probability": best["probability"],
            "elapsed": elapsed,
        })

        # 5. 내부 액션 → poke-env BattleOrder
        order = self._action_to_order(battle, state, best["action"])
        return order

    # ═══════════════════════════════════════════════════════════
    #  Battle → BattleState 변환
    # ═══════════════════════════════════════════════════════════

    def _battle_to_state(self, battle: Battle) -> BattleState:
        """poke-env Battle 패킷 파싱 결과 → 내부 BattleState."""

        # ── poke-env가 teraType을 파싱하지 않으므로 request에서 직접 추출
        tera_map: dict[str, str] = {}  # ident → teraType name
        last_req = getattr(battle, '_last_request', None)
        if last_req and 'side' in last_req:
            for poke_req in last_req['side'].get('pokemon', []):
                ident = poke_req.get('ident', '')
                tera = poke_req.get('teraType', '')
                if tera:
                    tera_map[ident] = tera

        # ── 아군 팀 (배틀에 실제 참가한 포켓몬만) ─────────────
        # BSS 3v3: 6마리 중 3마리만 참가. battle.team에는 6마리 전부 있음.
        # 실제 참가 = active + available_switches + 기절한 참가자
        in_battle = set()
        if battle.active_pokemon:
            in_battle.add(battle.active_pokemon)
        for sw in battle.available_switches:
            in_battle.add(sw)
        for pe_poke in battle.team.values():
            if pe_poke.fainted:
                in_battle.add(pe_poke)

        my_pokemon_list = [p for p in battle.team.values() if p in in_battle]
        my_team = []
        active_idx = 0
        for i, pe_poke in enumerate(my_pokemon_list):
            # request의 teraType을 fallback으로 전달
            req_tera = tera_map.get(
                f"{battle.player_role}: {pe_poke.species}", "")
            if not req_tera:
                # ident 형식이 다를 수 있으므로 team key로도 시도
                for ident, poke in battle.team.items():
                    if poke is pe_poke and ident in tera_map:
                        req_tera = tera_map[ident]
                        break
            sim_poke = self._pe_pokemon_to_sim(pe_poke, req_tera)
            if pe_poke == battle.active_pokemon:
                active_idx = i
                # 활성 포켓몬은 가용 기술 반영
                sim_poke.moves = self._get_available_move_ids(battle)
            my_team.append(sim_poke)

        # ── 상대 팀 (불완전 정보 → Smogon prior 보충) ────────
        opp_pokemon_list = list(battle.opponent_team.values())
        opp_team = []
        opp_active_idx = 0
        for i, pe_poke in enumerate(opp_pokemon_list):
            sim_poke = self._pe_pokemon_to_sim_opponent(pe_poke)
            if pe_poke == battle.opponent_active_pokemon:
                opp_active_idx = i
            opp_team.append(sim_poke)

        if not opp_team:
            opp_team = [self._make_dummy_pokemon()]

        # ── 날씨 ─────────────────────────────────────────────
        weather = Weather.NONE
        for pe_w in battle.weather:
            if pe_w in _PE_WEATHER_MAP:
                weather = _PE_WEATHER_MAP[pe_w]
                break

        # ── 필드/지형 ────────────────────────────────────────
        terrain = Terrain.NONE
        trick_room = 0
        for pe_f, turn in battle.fields.items():
            if pe_f in _PE_TERRAIN_MAP:
                terrain = _PE_TERRAIN_MAP[pe_f]
            elif pe_f == PEField.TRICK_ROOM:
                trick_room = 5  # 남은 턴 정확히 모름 → 보수적

        # ── 사이드 컨디션 ────────────────────────────────────
        my_side = Side(team=my_team, active_idx=active_idx)
        opp_side = Side(team=opp_team, active_idx=opp_active_idx)

        self._parse_side_conditions(battle.side_conditions, my_side)
        self._parse_side_conditions(battle.opponent_side_conditions, opp_side)

        # ── 테라 사용 여부 ───────────────────────────────────
        # 아군 중 테라 쓴 포켓몬이 있으면 tera_used = True
        for pe_poke in my_pokemon_list:
            if pe_poke.is_terastallized:
                my_side.tera_used = True
                break

        return BattleState(
            sides=[my_side, opp_side],
            weather=weather,
            terrain=terrain,
            trick_room_turns=trick_room,
            turn=battle.turn,
        )

    def _parse_side_conditions(self, conditions: dict, side: Side):
        """SideCondition dict → Side 필드 설정."""
        if not POKE_ENV_AVAILABLE:
            return
        for cond, val in conditions.items():
            if cond == PESideCondition.STEALTH_ROCK:
                side.stealth_rock = True
            elif cond == PESideCondition.SPIKES:
                side.spikes = min(3, val)
            elif cond == PESideCondition.TOXIC_SPIKES:
                side.toxic_spikes = min(2, val)
            elif cond == PESideCondition.STICKY_WEB:
                side.sticky_web = True
            elif cond == PESideCondition.REFLECT:
                side.reflect_turns = max(1, 5 - val)
            elif cond == PESideCondition.LIGHT_SCREEN:
                side.light_screen_turns = max(1, 5 - val)
            elif cond == PESideCondition.TAILWIND:
                side.tailwind_turns = max(1, 4 - val)

    # ─── poke-env Pokemon → 내부 Pokemon ─────────────────────

    def _pe_pokemon_to_sim(self, pe: PEPokemon,
                           req_tera: str = "") -> Pokemon:
        """아군 포켓몬 변환 — 완전한 정보.

        req_tera: Showdown request에서 읽은 teraType (poke-env 파싱 누락 보완).
        """
        types = [_pe_type_to_str(t) for t in pe.types if t is not None]

        # 스탯: poke-env가 계산한 값 사용
        stats = {}
        if pe.stats:
            stats = {k: pe.stats.get(k, 100) or 100 for k in STAT_KEYS}
        else:
            base = pe.base_stats or {k: 80 for k in STAT_KEYS}
            stats = {k: base.get(k, 80) + 50 for k in STAT_KEYS}

        # 기술 + PP
        moves = [_to_id(m.id) for m in pe.moves.values()]
        pp_list = []
        max_pp_list = []
        for m in pe.moves.values():
            pp_list.append(m.current_pp if m.current_pp is not None else 5)
            max_pp_list.append(m.max_pp if m.max_pp is not None else 5)

        # 부스트
        boosts = {k: 0 for k in STAT_KEYS[1:]}
        if pe.boosts:
            for k, v in pe.boosts.items():
                if k in boosts:
                    boosts[k] = v

        # 상태이상
        status = ""
        if pe.status and pe.status in _PE_STATUS_MAP:
            status = _PE_STATUS_MAP[pe.status]

        # HP
        cur_hp = int(pe.current_hp) if pe.current_hp else 0
        max_hp = int(pe.max_hp) if pe.max_hp else stats.get("hp", 100)

        # 테라 타입: poke-env → request fallback → 첫 번째 타입 fallback
        tera_type = ""
        if pe.tera_type is not None:
            tera_type = _pe_type_to_str(pe.tera_type)
        elif req_tera:
            tera_type = req_tera
        elif types:
            tera_type = types[0]  # 최후 fallback: 주 타입

        # 이미 테라한 포켓몬: 수비 타입이 테라 타입으로 변경
        is_tera = pe.is_terastallized
        if is_tera and tera_type:
            types = [tera_type]

        # volatile 상태 읽기
        encore_move = ""
        encore_turns = 0
        taunt_turns = 0
        disabled_move = ""
        disable_turns = 0
        torment = False
        if POKE_ENV_AVAILABLE and hasattr(pe, 'effects'):
            for eff, counter in pe.effects.items():
                if eff == PEEffect.ENCORE:
                    encore_turns = max(counter, 1)
                    # poke-env의 last move를 앵콜 기술로
                    if hasattr(pe, 'moved') and pe.moved:
                        encore_move = _to_id(pe.moved.id) if hasattr(pe.moved, 'id') else ""
                elif eff == PEEffect.TAUNT:
                    taunt_turns = max(counter, 1)
                elif eff == PEEffect.DISABLE:
                    disable_turns = max(counter, 1)
                elif eff == PEEffect.TORMENT:
                    torment = True

        return Pokemon(
            species_id=_to_id(pe.species),
            name=pe.species,
            types=types,
            base_stats=pe.base_stats or {k: 80 for k in STAT_KEYS},
            stats=stats,
            ability=pe.ability or "",
            item=pe.item or "",
            moves=moves[:4],
            pp=pp_list[:4],
            max_pp=max_pp_list[:4],
            cur_hp=cur_hp,
            max_hp=max_hp,
            status=status,
            boosts=boosts,
            tera_type=tera_type,
            is_tera=is_tera,
            fainted=pe.fainted,
            encore_move=encore_move,
            encore_turns=encore_turns,
            taunt_turns=taunt_turns,
            disabled_move=disabled_move,
            disable_turns=disable_turns,
            torment=torment,
        )

    def _pe_pokemon_to_sim_opponent(self, pe: PEPokemon) -> Pokemon:
        """상대 포켓몬 변환 — 세트 추론 우선, fallback으로 Smogon prior."""
        types = [_pe_type_to_str(t) for t in pe.types if t is not None]

        species_id = _to_id(pe.species)
        dex = self.gd.get_pokemon(species_id)
        usage = self.gd.get_stats(species_id, self.format_name)

        base_stats = dex["baseStats"] if dex else (
            pe.base_stats or {k: 80 for k in STAT_KEYS})

        # ── 관측 정보 수집 ────────────────────────────────────
        observed_item = pe.item or ""
        observed_moves = [_to_id(m.id) for m in pe.moves.values()]
        observed_ability = pe.ability or ""
        observed_tera = (_pe_type_to_str(pe.tera_type)
                         if pe.tera_type is not None else "")

        # 확정 아이템 기록 (Item Clause 추적)
        if observed_item:
            self.set_inferencer.update_confirmed_item(species_id, observed_item)

        # ── 세트 추론 시도 ────────────────────────────────────
        inferred = self.set_inferencer.infer(
            species_id, observed_item, observed_moves,
            observed_ability, observed_tera,
        )

        if inferred and inferred["confidence"] > 0.3:
            # 세트 기반 보충: 관측된 정보는 절대 덮어쓰지 않음
            # 아이템
            item = observed_item or inferred["item"]
            # 특성
            ability = observed_ability or inferred["ability"]
            # 기술: 관측된 것 + 세트에서 미확인 보충
            moves = list(observed_moves)
            for mv in inferred["moves"]:
                if mv not in moves and len(moves) < 4:
                    moves.append(mv)
            # 테라 타입
            tera_type = observed_tera or inferred["tera_type"]
            # 성격/노력치로 스탯 계산
            nature = inferred["nature"]
            evs = inferred["evs"]
            stats = DamageCalculator.calc_stats_from_spread(
                base_stats, nature, evs)
        else:
            # ── 기존 Smogon 개별 통계 fallback ────────────────
            # 스탯 추정: Smogon 최인기 스프레드
            if usage and usage.get("spreads"):
                best = max(usage["spreads"], key=lambda x: x["weight"])
                stats = DamageCalculator.calc_stats_from_spread(
                    base_stats, best["nature"], best["evs"]
                )
            else:
                stats = {k: int(base_stats.get(k, 80) * 1.1) + 5
                         for k in STAT_KEYS}

            # 기술: 확인된 것 + Smogon TOP으로 보충
            moves = list(observed_moves)
            if usage:
                top_moves = sorted(usage.get("moves", {}).items(),
                                   key=lambda x: -x[1])
                for mid, _ in top_moves:
                    if mid not in moves and len(moves) < 4:
                        moves.append(mid)

            # 특성: 확인된 것 or Smogon 1등
            ability = observed_ability
            if not ability and usage:
                top_ab = sorted(usage.get("abilities", {}).items(),
                                key=lambda x: -x[1])
                ability = top_ab[0][0] if top_ab else ""

            # 아이템: 확인된 것 or Smogon 1등
            item = observed_item
            if not item and usage:
                top_items = sorted(usage.get("items", {}).items(),
                                   key=lambda x: -x[1])
                item = top_items[0][0] if top_items else ""

            # 테라 타입: Smogon 1등 or fallback
            tera_type = observed_tera
            if not tera_type and usage and usage.get("tera_types"):
                top_tera = sorted(usage["tera_types"].items(),
                                  key=lambda x: -x[1])
                if top_tera:
                    tera_type = top_tera[0][0].capitalize()

        # ── 공통 필드 ────────────────────────────────────────
        # 부스트
        boosts = {k: 0 for k in STAT_KEYS[1:]}
        if pe.boosts:
            for k, v in pe.boosts.items():
                if k in boosts:
                    boosts[k] = v

        # 상태이상
        status = ""
        if pe.status and pe.status in _PE_STATUS_MAP:
            status = _PE_STATUS_MAP[pe.status]

        # HP (상대는 비율만 알 수 있음)
        hp_frac = pe.current_hp_fraction
        max_hp = stats.get("hp", 100)
        cur_hp = int(max_hp * hp_frac)

        # 상대 PP: 정확히 모름 → 각 기술의 기본 PP 사용
        opp_pp = []
        for mid in moves[:4]:
            md = self.gd.get_move(mid)
            opp_pp.append(md["pp"] if md and "pp" in md else 5)

        # 이미 테라한 포켓몬: 수비 타입이 테라 타입으로 변경
        is_tera = pe.is_terastallized
        if is_tera and tera_type:
            types = [tera_type]

        return Pokemon(
            species_id=species_id,
            name=pe.species,
            types=types,
            base_stats=base_stats,
            stats=stats,
            ability=ability,
            item=item,
            moves=moves[:4],
            pp=opp_pp,
            max_pp=list(opp_pp),
            cur_hp=cur_hp,
            max_hp=max_hp,
            status=status,
            boosts=boosts,
            tera_type=tera_type,
            is_tera=is_tera,
            fainted=pe.fainted,
        )

    def _make_dummy_pokemon(self) -> Pokemon:
        return Pokemon(
            species_id="unknown", name="Unknown",
            types=["Normal"], base_stats={k: 80 for k in STAT_KEYS},
            stats={k: 100 for k in STAT_KEYS},
            ability="", item="",
            moves=["tackle"], pp=[35], max_pp=[35],
            cur_hp=200, max_hp=200,
        )

    def _get_available_move_ids(self, battle: Battle) -> list[str]:
        """현재 사용 가능한 기술 ID 리스트 (PP 0인 기술 제외)."""
        return [_to_id(m.id) for m in battle.available_moves]

    # ═══════════════════════════════════════════════════════════
    #  내부 액션 → BattleOrder 변환
    # ═══════════════════════════════════════════════════════════

    def _action_to_order(self, battle: Battle, state: BattleState,
                         action: int) -> BattleOrder:
        """MCTS가 선택한 내부 액션 → poke-env BattleOrder.

        핵심: 내부 시뮬레이터의 기술/포켓몬 인덱스를 poke-env의
        available_moves / available_switches 순서와 정확히 매칭.
        """
        avail_moves = battle.available_moves
        avail_switches = battle.available_switches
        side = state.sides[0]
        active = side.active

        if action >= ActionType.TERA_MOVE1:
            # 테라스탈 기술 — 실패 시 일반 기술로 폴백
            move_idx = action - ActionType.TERA_MOVE1
            target_move_id = active.moves[move_idx] if move_idx < len(active.moves) else None
            pe_move = self._find_pe_move(avail_moves, target_move_id)
            if pe_move:
                can_tera = getattr(battle, 'can_tera', False)
                already_tera = any(
                    p.is_terastallized for p in battle.team.values()
                )
                if can_tera and not already_tera:
                    return self.create_order(pe_move, terastallize=True)
                else:
                    # 테라 불가 → 일반 기술로 폴백
                    return self.create_order(pe_move)

        elif action >= ActionType.SWITCH1:
            # 교체
            target_idx = action - ActionType.SWITCH1
            target_species = (side.team[target_idx].species_id
                              if target_idx < len(side.team) else None)
            pe_switch = self._find_pe_switch(avail_switches, target_species)
            if pe_switch:
                return self.create_order(pe_switch)

        else:
            # 일반 기술
            move_idx = action - ActionType.MOVE1
            target_move_id = active.moves[move_idx] if move_idx < len(active.moves) else None
            pe_move = self._find_pe_move(avail_moves, target_move_id)
            if pe_move:
                return self.create_order(pe_move)

        # 폴백
        if avail_moves:
            return self.create_order(avail_moves[0])
        if avail_switches:
            return self.create_order(avail_switches[0])
        return self.choose_default_move(battle)

    def _find_pe_move(self, avail_moves: list, target_id: str | None):
        """available_moves에서 내부 move_id와 매칭되는 Move 찾기."""
        if target_id is None:
            return None
        for m in avail_moves:
            if _to_id(m.id) == target_id:
                return m
        return None

    def _find_pe_switch(self, avail_switches: list, target_species: str | None):
        """available_switches에서 species_id로 매칭."""
        if target_species is None:
            return None
        for p in avail_switches:
            if _to_id(p.species) == target_species:
                return p
        return None


# ═══════════════════════════════════════════════════════════════
#  TrainingNetworkPlayer (Showdown 셀프플레이 학습용 — MCTS 없음)
# ═══════════════════════════════════════════════════════════════

class TrainingNetworkPlayer(MCTSPlayer):
    """Showdown 셀프플레이 학습용 — 경량 시뮬레이터 완전 배제.

    MCTS 없이 neural network policy + Dirichlet 노이즈로 직접 플레이.
    모든 게임 로직은 Showdown 엔진이 처리.
    매 턴 (state, policy, value) 수집 → 학습 데이터.

    흐름:
      Showdown 패킷 → Battle 객체 → BattleState 인코딩
      → 네트워크 forward → policy logits → mask + softmax + Dirichlet
      → 액션 샘플 → BattleOrder → Showdown
    """

    def __init__(self, *args,
                 temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_weight: float = 0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._battle_examples: dict[str, list] = {}
        self.training_examples: list = []
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        # NetworkEvaluator 직접 참조 (MCTS 경유하지 않음)
        self._net_eval = self.mcts.network

    def choose_move(self, battle):
        """Showdown 패킷 → 네트워크 policy → 학습 데이터 수집 + BattleOrder.

        MCTS/경량시뮬 없이 네트워크만 사용. 탐색 노이즈로 다양성 확보.
        """
        import torch
        import torch.nn.functional as F
        from neural_net import encode_state, NUM_ACTIONS
        from battle_sim import ActionType

        # 1. Battle → BattleState → 인코딩
        state = self._battle_to_state(battle)
        encoded = encode_state(state, 0, self.gd).numpy()

        # 2. 합법 액션 파악 (Showdown이 제공하는 정보 기반)
        legal_actions = self._get_legal_actions_from_battle(battle, state)
        if not legal_actions:
            return self.choose_default_move(battle)

        # 3. 네트워크 forward (경량 시뮬 사용 안함)
        net_eval = self._net_eval
        if net_eval is None:
            return self.choose_default_move(battle)

        x = torch.from_numpy(encoded).to(net_eval.device)
        with torch.no_grad():
            logits, value = net_eval.model(x)
        logits = logits.squeeze(0)  # (NUM_ACTIONS,)

        # 4. 불법 액션 마스킹
        mask = torch.full((NUM_ACTIONS,), float('-inf'), device=logits.device)
        for a in legal_actions:
            mask[a] = 0.0
        masked_logits = logits + mask

        # 5. Temperature + Dirichlet 노이즈
        if self.temperature > 0.01:
            scaled_logits = masked_logits / self.temperature
        else:
            scaled_logits = masked_logits

        raw_probs = F.softmax(scaled_logits, dim=0).cpu().numpy()

        # 합법 액션만 추출 → 재정규화 (부동소수점 안전)
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in legal_actions:
            probs[a] = max(raw_probs[a], 0.0)

        # Dirichlet 노이즈: 탐색 다양성
        if self.dirichlet_alpha > 0 and len(legal_actions) > 1:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                probs[a] = ((1 - self.dirichlet_weight) * probs[a]
                            + self.dirichlet_weight * noise[i])

        # 명시적 재정규화 (항상 수행)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # 폴백: 균등 분포
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)

        # 6. policy 벡터 저장 (학습 타겟)
        policy = probs.copy()

        tag = battle.battle_tag
        if tag not in self._battle_examples:
            self._battle_examples[tag] = []
        self._battle_examples[tag].append((encoded, policy))

        # 7. 액션 샘플링
        action = np.random.choice(NUM_ACTIONS, p=probs)

        # 8. 로깅
        if self.log_decisions:
            side = state.sides[0]
            active_name = side.active.name if side.active else "?"
            action_name = self._action_name(state, action)
            print(f"[턴 {battle.turn:2d}] {active_name:15s} "
                  f"→ {action_name:25s} ({probs[action]:.0%})")

        # 9. BattleOrder 반환 (Showdown에 전송)
        return self._action_to_order(battle, state, action)

    def _get_legal_actions_from_battle(self, battle, state) -> list[int]:
        """poke-env Battle 객체에서 합법 액션 인덱스 추출.

        Showdown이 제공하는 available_moves/available_switches만 사용.
        경량 시뮬레이터의 get_legal_actions() 아님.
        """
        from battle_sim import ActionType

        legal = []
        side = state.sides[0]
        active = side.active

        # 기술
        for i, pe_move in enumerate(battle.available_moves):
            move_id = _to_id(pe_move.id)
            for j, sim_move in enumerate(active.moves):
                if sim_move == move_id and j < 4:
                    legal.append(ActionType.MOVE1 + j)
                    break

        # 교체
        for pe_poke in battle.available_switches:
            switch_species = _to_id(pe_poke.species)
            for j, sim_poke in enumerate(side.team):
                if sim_poke.species_id == switch_species and j != side.active_idx:
                    legal.append(ActionType.SWITCH1 + j)
                    break

        # 테라스탈 기술 — 안전 검사 강화
        can_tera = getattr(battle, 'can_tera', False)
        if can_tera:
            # 이미 테라 사용한 포켓몬이 있으면 불가
            already_tera = any(
                p.is_terastallized for p in battle.team.values()
            )
            if not already_tera and not battle.active_pokemon.is_terastallized:
                for i, pe_move in enumerate(battle.available_moves):
                    move_id = _to_id(pe_move.id)
                    for j, sim_move in enumerate(active.moves):
                        if sim_move == move_id and j < 4:
                            legal.append(ActionType.TERA_MOVE1 + j)
                            break

        return legal if legal else [0]

    def _action_name(self, state, action: int) -> str:
        """디버그용 액션 이름."""
        from battle_sim import ActionType
        side = state.sides[0]
        active = side.active

        if action >= ActionType.TERA_MOVE1:
            idx = action - ActionType.TERA_MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return f"Tera+{move['name']}" if move else f"Tera Move {idx}"
        elif action >= ActionType.SWITCH1:
            idx = action - ActionType.SWITCH1
            if idx < len(side.team):
                return f"Switch: {side.team[idx].name}"
        else:
            idx = action - ActionType.MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return f"Move: {move['name']}" if move else f"Move {idx}"
        return f"Action {action}"

    def _battle_finished_callback(self, battle):
        """배틀 종료 시 승패 결과를 value로 할당."""
        from self_play import TrainingExample

        tag = battle.battle_tag
        if battle.won:
            value = 1.0
        elif battle.lost:
            value = -1.0
        else:
            value = 0.0

        if tag in self._battle_examples:
            for encoded, policy in self._battle_examples[tag]:
                self.training_examples.append(
                    TrainingExample(
                        state=encoded,
                        policy=policy,
                        value=value,
                    ))
            del self._battle_examples[tag]


# ═══════════════════════════════════════════════════════════════
#  오프라인 플레이어 (poke-env 없이 내부 시뮬레이터 전용)
# ═══════════════════════════════════════════════════════════════

class OfflineMCTSPlayer:
    """poke-env 없이 내부 시뮬레이터만으로 동작하는 AI."""

    def __init__(self, game_data: GameData, format_name: str = "bss",
                 n_simulations: int = 400):
        self.gd = game_data
        self.format_name = format_name
        self.sim = BattleSimulator(game_data)
        self.mcts = MCTS(
            game_data=game_data, simulator=self.sim,
            n_simulations=n_simulations, rollout_depth=30,
            n_rollout_per_leaf=2, use_parallel=True, n_workers=4,
            format_name=format_name, max_opp_branches=5,
        )
        self.decision_log: list[dict] = []

    def choose_action(self, state: BattleState, player: int) -> int:
        info = self.mcts.get_search_info(state, player)
        if not info["actions"]:
            legal = self.sim.get_legal_actions(state, player)
            return random.choice(legal) if legal else 0
        best = max(info["actions"], key=lambda x: x["visits"])
        self.decision_log.append({
            "turn": state.turn + 1,
            "pokemon": info["active_pokemon"],
            "action": best["name"],
            "probability": best["probability"],
        })
        return best["action"]


class NashPlayer(MCTSPlayer):
    """Nash 균형 기반 Showdown 대전 AI.

    MCTSPlayer를 상속하되, choose_move()만 NashSolver로 교체.
    _battle_to_state(), _action_to_order(), teampreview() 등은 그대로 재사용.
    """

    def __init__(
        self,
        game_data: GameData,
        format_name: str = "bss",
        log_decisions: bool = True,
        nash_solver=None,
        preview_checkpoint_path: str | None = None,
        network_evaluator=None,
        **kwargs,
    ):
        # MCTSPlayer.__init__이 MCTS를 만들지만, 우리는 쓰지 않음
        super().__init__(
            game_data=game_data,
            format_name=format_name,
            n_simulations=1,  # MCTS 사용 안함 — 최소값
            time_limit=1.0,
            log_decisions=log_decisions,
            network_evaluator=network_evaluator,
            preview_checkpoint_path=preview_checkpoint_path,
            **kwargs,
        )
        self.nash = nash_solver
        # poke-env는 protect_count를 안 줌 → 내부 추적
        self._used_protect_last: dict[str, bool] = {}  # species → True/False
        # 구애류 추론 상태
        self._opp_choice_infer: dict[str, dict] = {}
        # species → {"item": "choicescarf"|"choiceband"|"choicespecs"|"",
        #            "locked_move": str, "confidence": float}

    def _infer_choice_item(self, battle):
        """턴 로그에서 상대 구애 아이템 추론.

        - 스카프: 상대가 우리보다 선공인데 기본 스피드가 더 낮음 → 스카프
        - 띠/안경: 데미지가 예상보다 ~1.4배 이상 → 물리면 띠, 특수면 안경
        - 같은 기술만 반복 사용 → 구애 확정도 올림
        """
        if not POKE_ENV_AVAILABLE or battle.turn < 2:
            return

        opp_active = battle.opponent_active_pokemon
        if opp_active is None or opp_active.fainted:
            return

        sp_id = _to_id(opp_active.species)

        # 이미 아이템 확정이면 스킵
        if opp_active.item and opp_active.item != "unknown_item":
            return

        info = self._opp_choice_infer.get(sp_id, {
            "item": "", "locked_move": "", "confidence": 0.0,
            "last_move": "", "same_move_count": 0,
            "moved_first_count": 0, "turns_seen": 0,
        })

        # 이전 턴 events 분석
        prev_obs = battle.observations.get(battle.turn - 1)
        if not prev_obs or not prev_obs.events:
            self._opp_choice_infer[sp_id] = info
            return

        opp_role = "p2" if battle.player_role == "p1" else "p1"
        my_role = battle.player_role or "p1"

        # 이벤트에서 move 순서 + 기술명 추출
        move_order = []  # [(role, move_id), ...]
        for event in prev_obs.events:
            if len(event) >= 3 and event[0] == "move":
                ident = event[1]  # "p1a: Miraidon"
                move_name = event[2]
                role = ident.split(":")[0].replace("a", "").replace("b", "")
                move_order.append((role, _to_id(move_name)))

        # 상대가 사용한 기술
        opp_move = ""
        opp_moved_first = False
        for i, (role, mid) in enumerate(move_order):
            if role == opp_role:
                opp_move = mid
                if i == 0:
                    opp_moved_first = True
                break

        if not opp_move:
            self._opp_choice_infer[sp_id] = info
            return

        info["turns_seen"] += 1

        # ── 같은 기술 반복 체크 ──
        if info["last_move"] and info["last_move"] == opp_move:
            info["same_move_count"] += 1
        else:
            info["same_move_count"] = 0
        info["last_move"] = opp_move

        # ── 스피드 기반 스카프 추론 ──
        if opp_moved_first:
            info["moved_first_count"] += 1
            # 내 스피드와 비교
            my_active = battle.active_pokemon
            if my_active and my_active.stats and opp_active.base_stats:
                my_spe = my_active.stats.get("spe", 100)
                # 상대 예상 스피드: base stat 기반 추정 (무보정)
                opp_base_spe = opp_active.base_stats.get("spe", 80)
                opp_est_spe = int(opp_base_spe * 0.9 + 52)  # lv50 대략 추정
                # 선공했는데 기본 스피드가 내 실스탯보다 확실히 낮으면 스카프 의심
                if opp_est_spe < my_spe * 0.85:
                    info["item"] = "choicescarf"
                    info["confidence"] = min(1.0, info["confidence"] + 0.4)
                    info["locked_move"] = opp_move

        # ── 같은 기술 2회 이상 → 구애 확정도 올림 ──
        if info["same_move_count"] >= 1:
            info["confidence"] = min(1.0, info["confidence"] + 0.2)
            info["locked_move"] = opp_move
            # 아직 아이템 미확정이면 기술 카테고리로 추론
            if not info["item"]:
                move_data = self.gd.get_move(opp_move)
                if move_data:
                    if move_data["is_physical"]:
                        info["item"] = "choiceband"
                    elif move_data["is_special"]:
                        info["item"] = "choicespecs"

        # ── 데미지 역산 (HP 변화 기반) ──
        # 내 포켓몬의 HP 변화로 데미지 추정
        my_obs = prev_obs.active_pokemon
        curr_obs = battle.current_observation.active_pokemon if battle.current_observation else None
        if my_obs and curr_obs and hasattr(my_obs, 'current_hp_fraction') and hasattr(curr_obs, 'current_hp_fraction'):
            hp_before = my_obs.current_hp_fraction
            hp_after = curr_obs.current_hp_fraction
            hp_lost = hp_before - hp_after
            if hp_lost > 0 and opp_move and battle.active_pokemon:
                # 예상 데미지 계산 (구애 없이)
                move_data = self.gd.get_move(opp_move)
                if move_data and not move_data["is_status"]:
                    opp_sim = self._pe_pokemon_to_sim_opponent(opp_active)
                    my_sim = self._pe_pokemon_to_sim(battle.active_pokemon)
                    atk_d = {"types": opp_sim.types, "stats": opp_sim.stats,
                             "ability": opp_sim.ability, "item": "",
                             "status": opp_sim.status, "boosts": {}}
                    def_d = {"types": my_sim.types, "stats": my_sim.stats,
                             "ability": my_sim.ability, "item": my_sim.item,
                             "status": my_sim.status, "boosts": {},
                             "cur_hp": my_sim.cur_hp}
                    _, _, expected = self.dc.calc_damage(atk_d, def_d, move_data)
                    if expected > 0 and my_sim.max_hp > 0:
                        actual_dmg = hp_lost * my_sim.max_hp
                        ratio = actual_dmg / expected
                        # 1.35배 이상이면 구애 의심
                        if ratio > 1.35:
                            info["confidence"] = min(1.0, info["confidence"] + 0.3)
                            info["locked_move"] = opp_move
                            if not info["item"]:
                                if move_data["is_physical"]:
                                    info["item"] = "choiceband"
                                elif move_data["is_special"]:
                                    info["item"] = "choicespecs"

        self._opp_choice_infer[sp_id] = info

        # 로깅
        if info["confidence"] >= 0.4 and info["item"] and self.log_decisions:
            _ITEM_KR = {"choicescarf": "구애스카프",
                        "choiceband": "구애머리띠",
                        "choicespecs": "구애안경"}
            print(f"[구애추론] {opp_active.species}: "
                  f"{_ITEM_KR.get(info['item'], info['item'])} "
                  f"(확신: {info['confidence']:.0%}, "
                  f"잠금: {info['locked_move']})")

    def _apply_choice_inference(self, state: BattleState, battle):
        """추론된 구애 정보를 상대 상태에 반영."""
        opp_active = battle.opponent_active_pokemon
        if opp_active is None:
            return

        sp_id = _to_id(opp_active.species)
        info = self._opp_choice_infer.get(sp_id)
        if not info or info["confidence"] < 0.5:
            return

        # 이미 아이템 확정이면 스킵
        if opp_active.item and opp_active.item != "unknown_item":
            return

        opp_sim = state.sides[1].active
        if info["item"] and not opp_sim.item:
            opp_sim.item = info["item"]
        if info["locked_move"]:
            opp_sim.choice_locked_move = info["locked_move"]

    def _build_opponent_variants(self, battle, base_state: BattleState,
                                 max_variants: int = 3
                                 ) -> list[tuple[BattleState, float]]:
        """상대 액티브 포켓몬의 세트 분포 → K개 BattleState 변형.

        Args:
            battle: poke-env Battle 객체
            base_state: 기본 BattleState (내 쪽 + primary 추론 상대)
            max_variants: 최대 변형 수

        Returns:
            [(variant_state, weight), ...] 가중치 합=1.0
        """
        if not POKE_ENV_AVAILABLE:
            return [(base_state, 1.0)]

        opp_active = battle.opponent_active_pokemon
        if opp_active is None or opp_active.fainted:
            return [(base_state, 1.0)]

        species_id = _to_id(opp_active.species)

        # 관측 정보 수집
        observed_item = opp_active.item or ""
        observed_moves = [_to_id(m.id) for m in opp_active.moves.values()]
        observed_ability = opp_active.ability or ""
        observed_tera = (_pe_type_to_str(opp_active.tera_type)
                         if opp_active.tera_type is not None else "")

        # 세트 분포 추론
        dist = self.set_inferencer.infer_distribution(
            species_id, observed_item, observed_moves,
            observed_ability, observed_tera,
            max_sets=max_variants)

        if not dist or len(dist) <= 1:
            return [(base_state, 1.0)]

        # confidence가 매우 높으면 (>0.9) 단일 상태로 유지 (성능 절약)
        if dist[0]["weight"] > 0.9:
            return [(base_state, 1.0)]

        variants = []
        opp_observed = base_state.sides[1].active  # 관측된 런타임 상태 참조

        for set_dict in dist:
            variant_state = base_state.clone()

            # 상대 액티브만 교체 (벤치는 primary 유지)
            variant_pokemon = make_pokemon_from_set_dict(
                self.gd, species_id, set_dict,
                observed_pokemon=opp_observed)
            active_idx = variant_state.sides[1].active_idx
            variant_state.sides[1].team[active_idx] = variant_pokemon

            variants.append((variant_state, set_dict["weight"]))

        return variants

    def choose_move(self, battle) -> BattleOrder:
        """Showdown 패킷 → 내부 상태 → Nash 균형 → BattleOrder."""
        t0 = time.time()

        # 1. poke-env Battle → 내부 BattleState
        state = self._battle_to_state(battle)

        # 1.3. 구애류 추론 + 반영
        self._infer_choice_item(battle)
        self._apply_choice_inference(state, battle)

        # 1.5. protect_count 패치 — poke-env가 안 줌, 내부 추적
        active_sp = state.sides[0].active.name
        if self._used_protect_last.get(active_sp, False):
            state.sides[0].active.protect_count = 1

        # 2. 상대 세트 변형 생성
        variants = self._build_opponent_variants(battle, state)

        # 3. Nash 균형 계산 (P1 관점, variants 전달)
        p1_dict, p2_dict, gv, _, _, _, _ = self.nash.get_nash_strategies_both(
            state, add_noise=False, opp_variants=variants)

        # 4. 최적 액션 선택 (greedy: 최대 확률 액션)
        if p1_dict:
            best_action = max(p1_dict, key=p1_dict.get)
        else:
            # 폴백: 랜덤
            legal = self.sim.get_legal_actions(state, 0)
            best_action = random.choice(legal) if legal else 0

        elapsed = time.time() - t0

        # 5. 로깅
        if self.log_decisions:
            # 첫 턴에 매크로 전략 표시
            if battle.turn <= 1 and getattr(self, 'active_macro', None):
                _PAT_KR = {"setup_sweep": "세팅>스윕",
                           "break_clean": "브레이크>클린",
                           "cycle": "사이클"}
                pn, pi = self.active_macro
                print(f"[매크로 실행] {_PAT_KR.get(pn, pn)}: {pi}")
            side = state.sides[0]
            active = side.active
            action_name = self._nash_action_name(state, best_action)
            # 변형 정보 로깅
            if len(variants) > 1:
                variant_info = ", ".join(
                    f"{v[0].sides[1].active.item}({v[1]:.0%})"
                    for v in variants)
                print(f"[턴 {battle.turn:2d}] {active.name:15s} "
                      f"→ {action_name:25s} "
                      f"(GV={gv:+.3f}, K={len(variants)}, {elapsed:.2f}s)")
                print(f"         [변형] {variant_info}")
            else:
                print(f"[턴 {battle.turn:2d}] {active.name:15s} "
                      f"→ {action_name:25s} (GV={gv:+.3f}, {elapsed:.2f}s)")
            if p1_dict:
                top = sorted(p1_dict.items(), key=lambda x: -x[1])[:5]
                for a, p in top:
                    aname = self._nash_action_name(state, a)
                    print(f"         {aname:25s} {p:.0%}")

        # 6. 방어류 사용 추적 — 다음 턴 프루닝용
        _PROTECT_IDS = {"protect", "detect", "banefulbunker",
                        "spikyshield", "kingsshield", "obstruct",
                        "silktrap", "burningbulwark"}
        used_protect = False
        act = best_action
        if act < 4 or act >= 9:  # 기술 사용 (교체 아님)
            midx = act - 9 if act >= 9 else act
            moves = state.sides[0].active.moves
            if midx < len(moves) and moves[midx] in _PROTECT_IDS:
                used_protect = True
        # 교체면 해당 포켓몬의 방어 기록 리셋 (교체 후 첫 턴)
        if 4 <= act <= 8:
            sw_idx = act - 4
            if sw_idx < len(state.sides[0].team):
                sw_name = state.sides[0].team[sw_idx].name
                self._used_protect_last[sw_name] = False
        self._used_protect_last[active_sp] = used_protect

        # 7. 내부 액션 → poke-env BattleOrder
        return self._action_to_order(battle, state, best_action)

    def teampreview(self, battle: Battle) -> str:
        """팀 프리뷰 — 20×20 매치업 시뮬레이션 기반 최적 선출.

        1단계: 내 6C3=20 × 상대 6C3=20 = 400 초기 상태를 value net 배치 평가
        2단계: 20×20 보상행렬 → Nash 균형으로 최적 선출 결정
        3단계: 상위 후보에서 리드 순서 최적화
        """
        from itertools import combinations

        t0 = time.time()

        # 새 배틀 시작 시 초기화
        self.set_inferencer.reset()
        self._used_protect_last.clear()
        self._opp_choice_infer.clear()

        my_team_pe = list(battle.team.values())
        my_team = [self._pe_pokemon_to_sim(p) for p in my_team_pe]

        opp_names = [p.species for p in battle.opponent_team.values()]
        opp_team = []
        for name in opp_names:
            try:
                poke = make_pokemon_from_stats(
                    self.gd, name, self.format_name)
                opp_team.append(poke)
            except (ValueError, KeyError):
                continue

        # 상대 팀 불완전하면 부모 클래스 폴백
        if len(my_team) < 6 or len(opp_team) < 6:
            return super().teampreview(battle)

        evaluator = self.nash.evaluator if self.nash else None
        if evaluator is None:
            return super().teampreview(battle)

        # ── 1단계: 400 매치업 배치 평가 ──
        my_combos = list(combinations(range(6), 3))   # 20
        opp_combos = list(combinations(range(6), 3))  # 20

        states = []
        for my_idx in my_combos:
            for opp_idx in opp_combos:
                mt = [my_team[i].clone() for i in my_idx]
                ot = [opp_team[i].clone() for i in opp_idx]
                st = self.sim.create_battle_state(mt, ot)
                states.append(st)

        values = evaluator.evaluate_values_batch(states, 0)
        M = np.array(values, dtype=np.float64).reshape(20, 20)

        # ── 2단계: 각 내 선출의 worst-case (minimax) ──
        minimax_scores = M.min(axis=1)  # 상대가 최적으로 골랐을 때
        top5_idx = np.argsort(minimax_scores)[::-1][:5]

        if self.log_decisions:
            print(f"[선출/심층] 상대: {opp_names}")
            print(f"[선출/심층] 20×20 행렬 평가 완료 ({time.time()-t0:.1f}s)")
            for rank, ci in enumerate(top5_idx):
                names = [my_team[i].name for i in my_combos[ci]]
                print(f"  #{rank+1} {names}  "
                      f"minimax={minimax_scores[ci]:+.3f}  "
                      f"avg={M[ci].mean():+.3f}")

        # ── 3단계: 상위 후보 리드 순서 최적화 ──
        # 각 상위 후보의 3가지 리드 순서를 평가
        best_score = -2.0
        best_order = list(my_combos[top5_idx[0]])

        # 상대 예상 선출: 상대 minimax 기준 상위 3개 가중 평균
        opp_minimax = (-M).min(axis=0)  # 상대 관점 minimax
        opp_top3 = np.argsort(opp_minimax)[::-1][:3]

        for ci in top5_idx:
            combo = list(my_combos[ci])
            for lead_pos in range(3):
                # lead_pos번째를 선발로
                ordered = [combo[lead_pos]] + [combo[j] for j in range(3)
                                                if j != lead_pos]
                lead_states = []
                for oi in opp_top3:
                    opp_c = list(opp_combos[oi])
                    mt = [my_team[i].clone() for i in ordered]
                    ot = [opp_team[i].clone() for i in opp_c]
                    lead_states.append(
                        self.sim.create_battle_state(mt, ot))

                lead_vals = evaluator.evaluate_values_batch(lead_states, 0)
                score = float(min(lead_vals))  # worst-case

                if score > best_score:
                    best_score = score
                    best_order = ordered

        elapsed = time.time() - t0

        if self.log_decisions:
            names = [my_team[i].name for i in best_order]
            print(f"[선출/심층] 최종: {names}  "
                  f"score={best_score:+.3f}  ({elapsed:.1f}s)")

        # ── 4단계: 선출된 3마리로 매크로 탐색 + evaluator 설정 ──
        try:
            from macro_search import search_macros
            picked_team = [my_team[i] for i in best_order[:3]]
            macros = search_macros(picked_team, top_per_pattern=1)
            if macros:
                best_pat = max(macros.items(), key=lambda x: x[1][0][1])
                pat_name = best_pat[0]
                pat_info = best_pat[1][0][2]  # detail dict
                self.active_macro = (pat_name, pat_info)
                if hasattr(evaluator, 'set_macro'):
                    evaluator.set_macro(pat_name, pat_info)
                if self.log_decisions:
                    _PAT_KR = {"setup_sweep": "세팅>스윕",
                               "break_clean": "브레이크>클린",
                               "cycle": "사이클"}
                    print(f"[매크로] {_PAT_KR.get(pat_name, pat_name)}: "
                          f"{pat_info}")
            else:
                self.active_macro = None
                if hasattr(evaluator, 'set_macro'):
                    evaluator.set_macro(None)
        except ImportError:
            self.active_macro = None

        # best_order는 my_team 인덱스 (0~5)
        pick = best_order[:3]
        remaining = [i for i in range(len(my_team)) if i not in pick]
        full_order = pick + remaining

        return "/team " + "".join(str(i + 1) for i in full_order)

    def _nash_action_name(self, state, action: int) -> str:
        """디버그용 액션 이름."""
        side = state.sides[0]
        active = side.active
        if action >= ActionType.TERA_MOVE1:
            idx = action - ActionType.TERA_MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return f"Tera+{move['name']}" if move else f"Tera Move {idx}"
        elif action >= ActionType.SWITCH1:
            idx = action - ActionType.SWITCH1
            if idx < len(side.team):
                return f"Switch>{side.team[idx].name}"
        else:
            idx = action - ActionType.MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return move['name'] if move else f"Move {idx}"
        return f"Action {action}"


class RandomPlayer:
    """랜덤 플레이어 (벤치마크용)."""

    def __init__(self, game_data: GameData):
        self.sim = BattleSimulator(game_data)

    def choose_action(self, state: BattleState, player: int) -> int:
        legal = self.sim.get_legal_actions(state, player)
        return random.choice(legal) if legal else 0


# ═══════════════════════════════════════════════════════════════
#  AlphaZeroPlayer (체크포인트 로드 → NetworkEvaluator → MCTS)
# ═══════════════════════════════════════════════════════════════

class AlphaZeroPlayer:
    """AlphaZero 체크포인트를 사용하는 AI 플레이어.

    기존 MCTSPlayer/OfflineMCTSPlayer와 동일한 choose_action() API.
    네트워크가 Smogon prior + rollout을 대체.
    preview_checkpoint_path가 있으면 TeamPreviewNet도 로드.
    """

    def __init__(self, game_data: GameData, checkpoint_path: str,
                 format_name: str = "bss", n_simulations: int = 400,
                 device: str = "cpu",
                 preview_checkpoint_path: str | None = None):
        from neural_net import (
            PokemonNet, NetworkEvaluator,
            TeamPreviewNet, PreviewEvaluator,
        )

        self.gd = game_data
        self.format_name = format_name
        self.sim = BattleSimulator(game_data)
        self.tb = TeamBuilder(game_data, format_name)

        # 배틀 모델 로드
        self.model = PokemonNet()
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.evaluator = NetworkEvaluator(
            self.model, game_data, device=device)
        self.mcts = MCTS(
            game_data=game_data, simulator=self.sim,
            n_simulations=n_simulations, rollout_depth=30,
            n_rollout_per_leaf=2, use_parallel=True, n_workers=4,
            format_name=format_name, max_opp_branches=5,
            network_evaluator=self.evaluator,
        )
        self.decision_log: list[dict] = []

        # 프리뷰 모델 로드
        self.preview_eval = None
        if preview_checkpoint_path:
            try:
                p_model = TeamPreviewNet()
                p_ckpt = torch.load(
                    preview_checkpoint_path, map_location=device,
                    weights_only=False)
                if "model_state_dict" in p_ckpt:
                    p_model.load_state_dict(p_ckpt["model_state_dict"])
                else:
                    p_model.load_state_dict(p_ckpt)
                p_model.eval()
                self.preview_eval = PreviewEvaluator(
                    p_model, game_data, device=device)
            except Exception as e:
                print(f"[경고] 프리뷰 모델 로드 실패: {e}, 룰 기반 폴백")

    def choose_action(self, state: BattleState, player: int) -> int:
        """MCTS+Network 탐색 → 최적 수 반환."""
        info = self.mcts.get_search_info(state, player)
        if not info["actions"]:
            legal = self.sim.get_legal_actions(state, player)
            return random.choice(legal) if legal else 0

        best = max(info["actions"], key=lambda x: x["visits"])
        self.decision_log.append({
            "turn": state.turn + 1,
            "pokemon": info["active_pokemon"],
            "action": best["name"],
            "probability": best["probability"],
        })
        return best["action"]

    def choose_leads(self, team6: list[Pokemon],
                     opp_names: list[str]) -> list[int]:
        """6마리 중 선출할 3마리 인덱스 선택.

        PreviewEvaluator가 있으면 신경망 사용, 없으면 룰 기반 폴백.
        """
        if self.preview_eval and len(team6) == 6:
            # 상대 팀을 Pokemon 객체로 변환
            opp_team = []
            for name in opp_names:
                try:
                    poke = make_pokemon_from_stats(
                        self.gd, name, self.format_name)
                    opp_team.append(poke)
                except (ValueError, KeyError):
                    continue

            if len(opp_team) >= 6:
                return self.preview_eval.choose(
                    team6, opp_team, temperature=0.1)

        # 폴백: 룰 기반
        return self.tb.choose_leads(team6, opp_names)


# ═══════════════════════════════════════════════════════════════
#  ShowdownMCTSPlayer (Showdown 엔진 MCTS로 수 선택)
# ═══════════════════════════════════════════════════════════════

class ShowdownMCTSPlayer(TrainingNetworkPlayer):
    """Showdown 엔진 MCTS로 수를 선택하는 플레이어.

    셀프플레이 학습에서 사용: MCTS 탐색 → 방문 분포를 policy 타겟으로 수집.
    실전 대전에서도 사용 가능 (shadow battle 동기화 필요).

    흐름:
      1. poke-env Battle → BattleState → 내부 sim 상태가 아니라,
         Showdown request를 직접 사용해서 MCTS 탐색
      2. 방문 횟수 기반 수 선택
      3. (state, mcts_policy, value) 학습 데이터 수집
    """

    def __init__(self, *args,
                 n_mcts_sims: int = 200,
                 mcts_temperature: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mcts_sims = n_mcts_sims
        self.mcts_temperature = mcts_temperature
        # ShowdownBridge와 ShowdownMCTS는 지연 초기화
        self._showdown_bridge = None
        self._showdown_mcts = None

    def _ensure_mcts(self):
        """ShowdownBridge/ShowdownMCTS 지연 초기화."""
        if self._showdown_mcts is None:
            from showdown_bridge import ShowdownBridge
            from showdown_mcts import ShowdownMCTS
            self._showdown_bridge = ShowdownBridge()
            self._showdown_mcts = ShowdownMCTS(
                bridge=self._showdown_bridge,
                game_data=self.gd,
                network_evaluator=self._net_eval,
                n_simulations=self.n_mcts_sims,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_weight=self.dirichlet_weight,
                format_name=self.format_name,
            )

    def choose_move(self, battle):
        """Showdown 패킷 → shadow battle MCTS → BattleOrder.

        MCTS 탐색이 실패하면 부모 클래스(network policy)로 폴백.
        """
        from neural_net import encode_state, NUM_ACTIONS
        from showdown_mcts import ShowdownMCTS

        self._ensure_mcts()
        bridge = self._showdown_bridge
        mcts = self._showdown_mcts

        # 1. Battle → BattleState (network 인코딩용)
        state = self._battle_to_state(battle)
        encoded = encode_state(state, 0, self.gd).numpy()

        # 2. Shadow battle 생성
        # poke-env battle에서 현재까지의 request를 추출해서
        # init + 현재 상태 재현은 어렵지만,
        # 현재 BattleState를 MCTS에 직접 전달하는 방식 사용
        try:
            shadow_id = self._create_shadow_from_battle(battle)
        except Exception as e:
            if self.log_decisions:
                print(f"[MCTS] shadow 생성 실패: {e}, network 폴백")
            return super().choose_move(battle)

        # 3. MCTS 탐색
        try:
            visits = mcts.search(shadow_id, player=0,
                                 n_simulations=self.n_mcts_sims)
        except Exception as e:
            if self.log_decisions:
                print(f"[MCTS] 탐색 실패: {e}, network 폴백")
            bridge.destroy(shadow_id)
            return super().choose_move(battle)

        # 4. 방문 분포 → policy 벡터 (학습용)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        total_v = sum(visits.values()) or 1
        for action_str, count in visits.items():
            int_action = mcts._str_to_action_type(action_str)
            if int_action is not None and int_action < NUM_ACTIONS:
                policy[int_action] = count / total_v

        # 학습 데이터 수집
        tag = battle.battle_tag
        if tag not in self._battle_examples:
            self._battle_examples[tag] = []
        self._battle_examples[tag].append((encoded, policy))

        # 5. 수 선택 (temperature 적용)
        if not visits:
            bridge.destroy(shadow_id)
            return super().choose_move(battle)

        if self.mcts_temperature <= 0.01:
            best_action = max(visits, key=visits.get)
        else:
            actions = list(visits.keys())
            counts = np.array([visits[a] for a in actions], dtype=np.float64)
            counts = counts ** (1.0 / self.mcts_temperature)
            probs = counts / counts.sum()
            best_action = np.random.choice(actions, p=probs)

        # 6. Showdown 액션 → BattleOrder
        order = self._showdown_action_to_order(battle, state, best_action)

        # 7. 로깅
        if self.log_decisions:
            side = state.sides[0]
            active_name = side.active.name if side.active else "?"
            print(f"[턴 {battle.turn:2d}] {active_name:15s} "
                  f"→ {best_action:25s} (MCTS {total_v}sims)")
            for a, c in sorted(visits.items(), key=lambda x: -x[1])[:5]:
                print(f"         {a:25s} {c:4d}회 ({c/total_v:.0%})")

        bridge.destroy(shadow_id)
        return order

    def _create_shadow_from_battle(self, battle) -> str:
        """poke-env Battle → shadow battle 생성.

        현재 상태를 완벽하게 재현하기 어렵기 때문에,
        random battle 형식으로 새 배틀을 생성하고
        현재 팀 정보를 최대한 반영.

        셀프플레이에서는 양쪽 정보를 모두 알기 때문에
        trainer_showdown.py에서 직접 bridge를 사용하는 것이 더 정확.
        """
        bridge = self._showdown_bridge

        # 아군 팀 → packed team string
        my_paste = team_to_showdown_paste(
            self.gd,
            [p.species for p in battle.team.values()],
            self.format_name,
        )

        # 상대 팀 (알려진 정보)
        opp_paste = team_to_showdown_paste(
            self.gd,
            [p.species for p in battle.opponent_team.values()],
            self.format_name,
        )

        resp = bridge.init_battle(my_paste, opp_paste, "gen9bssregj")
        return resp["battle_id"]

    def _showdown_action_to_order(self, battle, state, action_str: str):
        """Showdown 액션 문자열 → BattleOrder."""
        parts = action_str.split()
        if not parts:
            return self.choose_default_move(battle)

        avail_moves = battle.available_moves
        avail_switches = battle.available_switches

        if parts[0] == "move" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                return self.choose_default_move(battle)

            terastallize = "terastallize" in parts
            # idx번째 기술 찾기 (request 기준이므로 available_moves 순서)
            if idx < len(avail_moves):
                pe_move = avail_moves[idx]
                if terastallize:
                    can_tera = getattr(battle, 'can_tera', False)
                    already_tera = any(
                        p.is_terastallized for p in battle.team.values())
                    if can_tera and not already_tera:
                        return self.create_order(pe_move, terastallize=True)
                return self.create_order(pe_move)

        elif parts[0] == "switch" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
            except ValueError:
                return self.choose_default_move(battle)
            # switch idx는 팀 내 위치 (1-indexed)
            # available_switches에서 매칭
            team_list = list(battle.team.values())
            if idx < len(team_list):
                target_species = _to_id(team_list[idx].species)
                for sw in avail_switches:
                    if _to_id(sw.species) == target_species:
                        return self.create_order(sw)

        # 폴백
        if avail_moves:
            return self.create_order(avail_moves[0])
        if avail_switches:
            return self.create_order(avail_switches[0])
        return self.choose_default_move(battle)

    def close(self):
        """리소스 정리."""
        if self._showdown_bridge:
            self._showdown_bridge.close()
            self._showdown_bridge = None
            self._showdown_mcts = None


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """에이전트 검증."""
    import time
    print("=== Agent 검증 ===\n")

    gd = GameData(device="cpu")

    # 팀 페이스트 생성 테스트
    species = ["Koraidon", "Ting-Lu", "Flutter Mane",
               "Ho-Oh", "Chien-Pao", "Glimmora"]
    paste = team_to_showdown_paste(gd, species, "bss")
    print("--- Showdown 팀 페이스트 ---")
    print(paste[:500])
    print("...\n")

    # 오프라인 대전
    mcts_player = OfflineMCTSPlayer(gd, "bss", n_simulations=200)
    rand_player = RandomPlayer(gd)
    sim = BattleSimulator(gd)

    print("MCTS vs Random: 10게임\n")
    mcts_wins = 0
    t0 = time.time()

    for g in range(10):
        team1 = [make_pokemon_from_stats(gd, n, "bss")
                 for n in ["Koraidon", "Ting-Lu", "Flutter Mane"]]
        team2 = [make_pokemon_from_stats(gd, n, "bss")
                 for n in ["Miraidon", "Gholdengo", "Great Tusk"]]
        state = sim.create_battle_state(team1, team2)

        for _ in range(50):
            if state.is_terminal:
                break
            a1 = mcts_player.choose_action(state, 0)
            a2 = rand_player.choose_action(state, 1)
            state = sim.step(state, a1, a2)

        if state.winner == 0:
            mcts_wins += 1
        print(f"  게임 {g+1}: {'MCTS' if state.winner==0 else 'Random'} 승 (턴 {state.turn})")

    print(f"\nMCTS 승률: {mcts_wins}/10 ({time.time()-t0:.1f}s)")
    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
