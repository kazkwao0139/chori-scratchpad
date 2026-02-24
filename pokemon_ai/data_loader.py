"""Showdown 데이터 파싱 — pokedex, moves, typechart, learnsets, stats, items, abilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

DATA_DIR = Path(__file__).parent / "data"

# ── 18 타입 순서 (텐서 인덱싱용) ─────────────────────────────
TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
TYPE_TO_IDX_LOWER = {t.lower(): i for i, t in enumerate(TYPES)}
NUM_TYPES = len(TYPES)

# ── 상태이상 ──────────────────────────────────────────────────
STATUS = ["none", "brn", "par", "psn", "tox", "slp", "frz"]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUS)}

# ── 성격 보정 ─────────────────────────────────────────────────
NATURES: dict[str, tuple[str | None, str | None]] = {
    "Hardy": (None, None), "Lonely": ("atk", "def"), "Brave": ("atk", "spe"),
    "Adamant": ("atk", "spa"), "Naughty": ("atk", "spd"),
    "Bold": ("def", "atk"), "Docile": (None, None), "Relaxed": ("def", "spe"),
    "Impish": ("def", "spa"), "Lax": ("def", "spd"),
    "Timid": ("spe", "atk"), "Hasty": ("spe", "def"), "Serious": (None, None),
    "Jolly": ("spe", "spa"), "Naive": ("spe", "spd"),
    "Modest": ("spa", "atk"), "Mild": ("spa", "def"), "Quiet": ("spa", "spe"),
    "Bashful": (None, None), "Rash": ("spa", "spd"),
    "Calm": ("spd", "atk"), "Gentle": ("spd", "def"), "Sassy": ("spd", "spe"),
    "Careful": ("spd", "spa"), "Quirky": (None, None),
}

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]


def _to_id(name: str) -> str:
    """Showdown ID 변환: 'Life Orb' → 'lifeorb'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ═══════════════════════════════════════════════════════════════
#  1. Pokédex
# ═══════════════════════════════════════════════════════════════

def load_pokedex(path: Path | None = None) -> dict[str, dict]:
    """pokedex.json → {id: {name, types, baseStats, abilities, num, ...}}"""
    path = path or DATA_DIR / "pokedex.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    dex: dict[str, dict] = {}
    for pid, info in raw.items():
        # Gen 9 이외 & 비표준 제외
        ns = info.get("isNonstandard")
        if ns in ("Past", "Future", "LGPE", "CAP", "Custom"):
            if info.get("num", 0) > 0 and ns == "Past":
                pass  # Mega 등은 일단 포함 (나중에 포맷별 필터)
            else:
                continue
        if info.get("num", 0) <= 0:
            continue
        base = info.get("baseStats", {})
        dex[pid] = {
            "name": info["name"],
            "num": info["num"],
            "types": [t for t in info.get("types", [])],
            "type_ids": [TYPE_TO_IDX.get(t, 0) for t in info.get("types", [])],
            "baseStats": {k: base.get(k, 0) for k in STAT_KEYS},
            "abilities": list(info.get("abilities", {}).values()),
            "weightkg": info.get("weightkg", 0),
            "tier": info.get("tier", ""),
        }
    return dex


# ═══════════════════════════════════════════════════════════════
#  2. Moves
# ═══════════════════════════════════════════════════════════════

def load_moves(path: Path | None = None) -> dict[str, dict]:
    """moves.json → {id: {name, basePower, type, category, accuracy, priority, flags, ...}}"""
    path = path or DATA_DIR / "moves.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    moves: dict[str, dict] = {}
    for mid, info in raw.items():
        ns = info.get("isNonstandard")
        if ns in ("Past", "Future", "LGPE", "CAP"):
            continue
        if info.get("num", 0) <= 0:
            continue
        cat = info.get("category", "Status")
        moves[mid] = {
            "name": info.get("name", mid),
            "num": info.get("num", 0),
            "basePower": info.get("basePower", 0),
            "type": info.get("type", "Normal"),
            "type_id": TYPE_TO_IDX.get(info.get("type", "Normal"), 0),
            "category": cat,  # Physical / Special / Status
            "is_physical": cat == "Physical",
            "is_special": cat == "Special",
            "is_status": cat == "Status",
            "accuracy": info.get("accuracy", 100),  # True → always hits
            "pp": info.get("pp", 5),
            "priority": info.get("priority", 0),
            "flags": info.get("flags", {}),
            "drain": info.get("drain"),        # [numerator, denominator]
            "recoil": info.get("recoil"),      # [numerator, denominator]
            "secondary": info.get("secondary"),
            "target": info.get("target", "normal"),
            "critRatio": info.get("critRatio", 1),
            "multihit": info.get("multihit"),
            # 추가 필드 (게임 지식 인코딩용)
            "status": info.get("status"),            # 'brn', 'par', etc. (직접 상태이상)
            "boosts": info.get("boosts"),             # 변화기의 자기 부스트 (e.g. 칼춤 {'atk': 2})
            "self": info.get("self"),                 # self.boosts (e.g. 인파 {'boosts': {'def': -1, 'spd': -1}})
            "selfSwitch": info.get("selfSwitch"),     # U-turn, Volt Switch 등
            "forceSwitch": info.get("forceSwitch"),   # 울부짖기, 회오리 등
            "sideCondition": info.get("sideCondition"),  # 스텔스록, 독압정 등
            "heal": info.get("heal"),                 # [1, 2] → 50% 회복
        }
    return moves


# ═══════════════════════════════════════════════════════════════
#  3. Type Chart → 18×18 effectiveness matrix
# ═══════════════════════════════════════════════════════════════

def load_typechart(path: Path | None = None) -> torch.Tensor:
    """typechart.js → (18, 18) tensor.  chart[atk_type, def_type] = multiplier.

    Showdown damageTaken 코드: 0 = 1×, 1 = 2× (super effective),
    2 = 0.5× (not very effective), 3 = 0× (immune).
    """
    path = path or DATA_DIR / "typechart.js"
    text = path.read_text(encoding="utf-8")
    # JS → JSON: 'exports.BattleTypeChart = {...};' 형태
    text = re.sub(r"^exports\.BattleTypeChart\s*=\s*", "", text.strip())
    text = text.rstrip(";")
    # JS object → JSON: 프로퍼티 키 따옴표 추가
    text = re.sub(r"(\w+)\s*:", r'"\1":', text)
    data: dict[str, Any] = json.loads(text)

    # damageTaken 맵핑: value → multiplier
    dmg_map = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

    chart = torch.ones(NUM_TYPES, NUM_TYPES)  # default 1×
    for def_type_name, info in data.items():
        def_idx = TYPE_TO_IDX_LOWER.get(def_type_name.lower())
        if def_idx is None:
            continue
        dt = info.get("damageTaken", {})
        for atk_type_name, val in dt.items():
            atk_idx = TYPE_TO_IDX.get(atk_type_name)
            if atk_idx is None:
                continue
            chart[atk_idx, def_idx] = dmg_map.get(val, 1.0)
    return chart


def type_effectiveness(chart: torch.Tensor, atk_type: int,
                       def_type1: int, def_type2: int = -1) -> float:
    """단일 상성 계산."""
    eff = chart[atk_type, def_type1].item()
    if def_type2 >= 0:
        eff *= chart[atk_type, def_type2].item()
    return eff


# ═══════════════════════════════════════════════════════════════
#  4. Learnsets (Gen 9 필터)
# ═══════════════════════════════════════════════════════════════

def load_learnsets(path: Path | None = None) -> dict[str, set[str]]:
    """learnsets.json → {pokemon_id: set of move_ids} (Gen 9만)."""
    path = path or DATA_DIR / "learnsets.json"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for poke_id, info in raw.items():
        ls = info.get("learnset", {})
        gen9_moves: set[str] = set()
        for move_id, sources in ls.items():
            for src in sources:
                # Gen 9 소스: '9L1', '9M', '9T', '9E', '9S0', etc.
                if src.startswith("9"):
                    gen9_moves.add(move_id)
                    break
        if gen9_moves:
            result[poke_id] = gen9_moves
    return result


# ═══════════════════════════════════════════════════════════════
#  5. Smogon Usage Stats (BSS / VGC)
# ═══════════════════════════════════════════════════════════════

def load_usage_stats(path: Path | None = None,
                     format_name: str = "bss") -> dict[str, dict]:
    """bss_stats.json / vgc_stats.json → 포켓몬별 사용률, 기술/아이템/EV 확률분포.

    Returns:
        {pokemon_id: {
            raw_count, abilities: {id: weight},
            items: {id: weight}, spreads: {spread_str: weight},
            moves: {id: weight}, teammates: {id: weight},
            counters: [(id, pct, std_dev, koed_pct)]
        }}
    """
    if path is None:
        fname = "bss_stats.json" if format_name == "bss" else "vgc_stats.json"
        path = DATA_DIR / fname
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    meta = raw.get("info", {})
    data = raw.get("data", {})

    stats: dict[str, dict] = {}
    total_battles = meta.get("number of battles", 1)

    for name, info in data.items():
        pid = _to_id(name)
        raw_count = info.get("Raw count", 0)
        usage_pct = raw_count / max(total_battles, 1)

        # 확률 분포로 정규화
        def _normalize(d: dict) -> dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return {}
            return {_to_id(k): v / total for k, v in d.items()}

        abilities = _normalize(info.get("Abilities", {}))
        items = _normalize(info.get("Items", {}))
        spreads_raw = info.get("Spreads", {})
        moves_raw = info.get("Moves", {})
        teammates_raw = info.get("Teammates", {})

        # Spreads: "Adamant:252/0/4/0/0/252" → 파싱
        spreads: list[dict] = []
        sp_total = sum(spreads_raw.values()) if spreads_raw else 1
        for spread_str, weight in spreads_raw.items():
            parts = spread_str.split(":")
            if len(parts) == 2:
                nature = parts[0]
                evs_str = parts[1].split("/")
                if len(evs_str) == 6:
                    evs = {k: int(v) for k, v in zip(STAT_KEYS, evs_str)}
                    spreads.append({
                        "nature": nature,
                        "evs": evs,
                        "weight": weight / sp_total,
                    })

        # Moves 정규화
        moves = _normalize(moves_raw)

        # Tera Types 정규화
        tera_types_raw = info.get("Tera Types", {})
        tera_types = _normalize(tera_types_raw)

        # Teammates 정규화 (음수값 있을 수 있음 → 클리핑)
        tm = {}
        if teammates_raw:
            tm_total = sum(max(0, v) for v in teammates_raw.values())
            if tm_total > 0:
                tm = {_to_id(k): max(0, v) / tm_total
                      for k, v in teammates_raw.items()}

        stats[pid] = {
            "name": name,
            "raw_count": raw_count,
            "usage_pct": usage_pct,
            "abilities": abilities,
            "items": items,
            "spreads": spreads,
            "moves": moves,
            "tera_types": tera_types,
            "teammates": tm,
        }
    return stats


# ═══════════════════════════════════════════════════════════════
#  6. Items (JS → 주요 배틀 효과)
# ═══════════════════════════════════════════════════════════════

# 데미지/배틀에 영향을 주는 핵심 아이템만 매핑
ITEM_EFFECTS: dict[str, dict] = {
    "lifeorb":       {"damage_mod": 1.3, "recoil_pct": 0.1},
    "choiceband":    {"damage_mod": 1.5, "choice_lock": True, "stat": "atk"},
    "choicespecs":   {"damage_mod": 1.5, "choice_lock": True, "stat": "spa"},
    "choicescarf":   {"speed_mod": 1.5, "choice_lock": True},
    "expertbelt":    {"super_eff_mod": 1.2},
    "assaultvest":   {"spd_mod": 1.5},
    "eviolite":      {"def_mod": 1.5, "spd_mod": 1.5},
    "leftovers":     {"end_turn_heal_pct": 1/16},
    "blacksludge":   {"end_turn_heal_pct": 1/16, "poison_only": True},
    "sitrusberry":   {"heal_at_half": 0.25},
    "focussash":     {"sash": True},
    "rockyhelmet":   {"contact_damage_pct": 1/6},
    "boosterenergy": {"booster": True},
    "clearamulet":   {"prevent_stat_drop": True},
    "covertcloak":   {"prevent_secondary": True},
    "safetygoggles": {"weather_immune": True, "powder_immune": True},
    "heavydutyboots":{"hazard_immune": True},
    "weaknesspolicy":{"wp": True},
    "throatspray":   {"ts": True},
    "lumberry":      {"cure_status": True},
    "mentalherb":    {"cure_infatuation": True},
    "mirrorherb":    {"copy_boosts": True},
    "loadeddice":    {"multihit_min": 4},
    "widelens":      {"accuracy_mod": 1.1},
    "scopelens":     {"crit_stage": 1},
    "redcard":       {"force_switch": True},
    "ejectbutton":   {"eject": True},
    "airballoon":    {"levitate": True},
}

# 타입 강화 아이템
_TYPE_BOOST_ITEMS: dict[str, tuple[str, float]] = {
    "silkscarf": ("Normal", 1.2), "charcoal": ("Fire", 1.2),
    "mysticwater": ("Water", 1.2), "magnet": ("Electric", 1.2),
    "miracleseed": ("Grass", 1.2), "nevermeltice": ("Ice", 1.2),
    "blackbelt": ("Fighting", 1.2), "poisonbarb": ("Poison", 1.2),
    "softsand": ("Ground", 1.2), "sharpbeak": ("Flying", 1.2),
    "twistedspoon": ("Psychic", 1.2), "silverpowder": ("Bug", 1.2),
    "hardstone": ("Rock", 1.2), "spelltag": ("Ghost", 1.2),
    "dragonfang": ("Dragon", 1.2), "blackglasses": ("Dark", 1.2),
    "metalcoat": ("Steel", 1.2),
    # Gen 9 feathers (1.2×)
    "punchingglove": ("Fighting", 1.1),  # 접촉 제거 + 펀치기술 1.1
    "fairyfeather": ("Fairy", 1.2),
}

# 플레이트 (1.2×)
for _t in TYPES:
    _plate_name = _to_id(_t + "plate") if _t != "Normal" else None
    # Arceus plates 등, 실전에선 거의 미사용 → 생략

def load_items_js(path: Path | None = None) -> dict[str, dict]:
    """items.js에서 주요 아이템 정보를 추출 (fling power, gen 등)."""
    path = path or DATA_DIR / "items.js"
    text = path.read_text(encoding="utf-8")
    # 간단한 파싱: 이름과 gen 정보 추출
    text = re.sub(r"^exports\.BattleItems\s*=\s*", "", text.strip())
    text = text.rstrip(";")
    # JS → JSON 변환
    text = re.sub(r"(\w+)\s*:", r'"\1":', text)
    # boolean
    text = text.replace(":true", ":true").replace(":false", ":false")
    text = text.replace(":null", ":null")
    # 함수 제거 (onXxx 콜백은 무시)
    # 간단 접근: JSON 파싱 시도, 실패하면 기본 매핑 사용
    items: dict[str, dict] = {}
    try:
        data = json.loads(text)
        for iid, info in data.items():
            gen = info.get("gen", 0)
            if gen > 9:
                continue
            items[iid] = {
                "name": info.get("name", iid),
                "gen": gen,
                "fling_power": info.get("fling", {}).get("basePower", 0)
                               if isinstance(info.get("fling"), dict) else 0,
            }
    except json.JSONDecodeError:
        # JS 파싱 실패 → 최소 매핑
        pass
    return items


# ═══════════════════════════════════════════════════════════════
#  7. Abilities (JS → 주요 배틀 효과)
# ═══════════════════════════════════════════════════════════════

ABILITY_EFFECTS: dict[str, dict] = {
    "adaptability":   {"stab_mod": 2.0},
    "aerilate":       {"normal_to_type": "Flying", "type_mod": 1.2},
    "pixilate":       {"normal_to_type": "Fairy", "type_mod": 1.2},
    "refrigerate":    {"normal_to_type": "Ice", "type_mod": 1.2},
    "galvanize":      {"normal_to_type": "Electric", "type_mod": 1.2},
    "hugepower":      {"atk_mod": 2.0},
    "purepower":      {"atk_mod": 2.0},
    "hustle":         {"atk_mod": 1.5, "accuracy_mod": 0.8},
    "guts":           {"atk_mod_if_status": 1.5, "ignore_burn": True},
    "marvelscale":    {"def_mod_if_status": 1.5},
    "intimidate":     {"on_switch_atk_drop": 1},
    "drought":        {"set_weather": "sun"},
    "drizzle":        {"set_weather": "rain"},
    "snowwarning":    {"set_weather": "snow"},
    "sandstream":     {"set_weather": "sand"},
    "electricsurge":  {"set_terrain": "electric"},
    "grassysurge":    {"set_terrain": "grassy"},
    "mistysurge":     {"set_terrain": "misty"},
    "psychicsurge":   {"set_terrain": "psychic"},
    "protosynthesis": {"booster": "sun"},
    "quarkdrive":     {"booster": "electric"},
    "orichalcumpulse":{"set_weather": "sun", "atk_boost_sun": 1.3333},
    "hadronengine":   {"set_terrain": "electric", "spa_boost_terrain": 1.3333},
    "swordofruin":    {"foe_def_mod": 0.75},
    "tabletsofruin":  {"foe_atk_mod": 0.75},
    "vesselofruin":   {"foe_spd_mod": 0.75},
    "beadsofruin":    {"foe_spa_mod": 0.75},
    "multiscale":     {"full_hp_damage_mod": 0.5},
    "shadowshield":   {"full_hp_damage_mod": 0.5},
    "sturdy":         {"sash_like": True},
    "levitate":       {"ground_immune": True},
    "flashfire":      {"fire_immune": True, "fire_boost": 1.5},
    "voltabsorb":     {"electric_immune": True, "heal": True},
    "waterabsorb":    {"water_immune": True, "heal": True},
    "stormdrain":     {"water_immune": True, "spa_boost": 1},
    "lightningrod":   {"electric_immune": True, "spa_boost": 1},
    "sapsipper":      {"grass_immune": True, "atk_boost": 1},
    "thickfat":       {"fire_resist": 0.5, "ice_resist": 0.5},
    "heatproof":      {"fire_resist": 0.5},
    "furcoat":        {"def_mod": 2.0},
    "icescales":      {"special_damage_mod": 0.5},
    "filter":         {"super_eff_mod": 0.75},
    "solidrock":      {"super_eff_mod": 0.75},
    "prismarmor":     {"super_eff_mod": 0.75},
    "tintedlens":     {"not_very_eff_mod": 2.0},
    "technician":     {"low_bp_mod": 1.5, "bp_threshold": 60},
    "sheerforce":     {"secondary_mod": 1.3, "remove_secondary": True},
    "ironfist":       {"punch_mod": 1.2},
    "strongjaw":      {"bite_mod": 1.5},
    "megalauncher":   {"pulse_mod": 1.5},
    "toughclaws":     {"contact_mod": 1.3},
    "sandforce":      {"sand_mod": 1.3, "sand_types": ["Rock", "Ground", "Steel"]},
    "moldbreaker":    {"mold_breaker": True},
    "teravolt":       {"mold_breaker": True},
    "turboblaze":     {"mold_breaker": True},
    "unaware":        {"ignore_boosts": True},
    "contrary":       {"invert_boosts": True},
    "magicbounce":    {"reflect_status": True},
    "magicguard":     {"indirect_immune": True},
    "regenerator":    {"switch_heal": 1/3},
    "naturalcure":    {"switch_cure": True},
    "serenegrace":    {"secondary_chance_mod": 2.0},
    "skilllink":      {"multihit_max": True},
    "compoundeyes":   {"accuracy_mod": 1.3},
    "noguard":        {"always_hit": True},
    "supremeoverlord":{"boost_per_faint": 0.1},
    "commanderally":  {},  # 더블 전용, 특수 처리 필요
    "swiftswim":      {"speed_weather": "rain", "speed_mul": 2.0},
    "chlorophyll":    {"speed_weather": "sun", "speed_mul": 2.0},
    "sandrush":       {"speed_weather": "sand", "speed_mul": 2.0},
    "slushrush":      {"speed_weather": "snow", "speed_mul": 2.0},
    "surgesurfer":    {"speed_terrain": "electric", "speed_mul": 2.0},
}


def load_abilities_js(path: Path | None = None) -> dict[str, dict]:
    """abilities.js에서 기본 정보 추출."""
    path = path or DATA_DIR / "abilities.js"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^exports\.BattleAbilities\s*=\s*", "", text.strip())
    text = text.rstrip(";")
    text = re.sub(r"(\w+)\s*:", r'"\1":', text)
    text = text.replace(":true", ":true").replace(":false", ":false")
    text = text.replace(":null", ":null")
    abilities: dict[str, dict] = {}
    try:
        data = json.loads(text)
        for aid, info in data.items():
            abilities[aid] = {
                "name": info.get("name", aid),
                "rating": info.get("rating", 0),
                "desc": info.get("shortDesc", ""),
            }
    except json.JSONDecodeError:
        pass
    return abilities


# ═══════════════════════════════════════════════════════════════
#  통합 로더
# ═══════════════════════════════════════════════════════════════

class GameData:
    """모든 게임 데이터를 한 번에 로드하고 보관."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pokedex = load_pokedex()
        self.moves = load_moves()
        self.type_chart = load_typechart().to(self.device)
        self.learnsets = load_learnsets()
        self.bss_stats = load_usage_stats(format_name="bss")
        self.vgc_stats = load_usage_stats(format_name="vgc")
        self.item_effects = ITEM_EFFECTS
        self.type_boost_items = _TYPE_BOOST_ITEMS
        self.ability_effects = ABILITY_EFFECTS

    def get_pokemon(self, name: str) -> dict | None:
        pid = _to_id(name)
        return self.pokedex.get(pid)

    def get_move(self, name: str) -> dict | None:
        mid = _to_id(name)
        return self.moves.get(mid)

    def get_legal_moves(self, pokemon: str) -> set[str]:
        pid = _to_id(pokemon)
        return self.learnsets.get(pid, set())

    def effectiveness(self, atk_type: str | int,
                      def_types: list[str | int]) -> float:
        if isinstance(atk_type, str):
            atk_type = TYPE_TO_IDX.get(atk_type, 0)
        eff = 1.0
        for dt in def_types:
            if isinstance(dt, str):
                dt = TYPE_TO_IDX.get(dt, 0)
            eff *= self.type_chart[atk_type, dt].item()
        return eff

    def get_stats(self, pokemon: str, format_name: str = "bss") -> dict | None:
        pid = _to_id(pokemon)
        stats = self.bss_stats if format_name == "bss" else self.vgc_stats
        return stats.get(pid)


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """기본 데이터 로드 검증."""
    print("=== Data Loader 검증 ===\n")

    gd = GameData(device="cpu")

    # 포켓몬 수
    print(f"Pokédex: {len(gd.pokedex)} 포켓몬 로드")

    # 코라이돈 확인
    koraidon = gd.get_pokemon("Koraidon")
    if koraidon:
        print(f"코라이돈 종족값: {koraidon['baseStats']}")
        print(f"코라이돈 타입: {koraidon['types']}")
    else:
        print("WARNING: 코라이돈을 찾을 수 없습니다")

    # 기술 수
    print(f"\nMoves: {len(gd.moves)} 기술 로드")
    eq = gd.get_move("Earthquake")
    if eq:
        print(f"지진 위력: {eq['basePower']}, 타입: {eq['type']}, 명중: {eq['accuracy']}")

    # 타입 상성
    print(f"\n타입 차트: {gd.type_chart.shape}")
    eff = gd.effectiveness("Fire", ["Grass"])
    print(f"불 → 풀: {eff}×")
    eff = gd.effectiveness("Water", ["Fire", "Ground"])
    print(f"물 → 불/땅: {eff}×")
    eff = gd.effectiveness("Normal", ["Ghost"])
    print(f"노말 → 고스트: {eff}×")

    # 배울 수 있는 기술
    print(f"\nLearnsets: {len(gd.learnsets)} 포켓몬 로드")
    kl = gd.get_legal_moves("Koraidon")
    print(f"코라이돈 Gen9 기술 수: {len(kl)}")

    # 사용률 통계
    print(f"\nBSS stats: {len(gd.bss_stats)} 포켓몬")
    print(f"VGC stats: {len(gd.vgc_stats)} 포켓몬")
    tinglu = gd.get_stats("Ting-Lu", "bss")
    if tinglu:
        print(f"딩루 BSS 사용률: {tinglu['usage_pct']:.2%}")
        top_moves = sorted(tinglu["moves"].items(), key=lambda x: -x[1])[:5]
        print(f"딩루 인기 기술 TOP5: {top_moves}")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
