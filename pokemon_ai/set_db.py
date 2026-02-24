"""상대 세트 추론 시스템 — Smogon 통계 기반 베이지안 세트 추론.

관측 정보(아이템, 기술, 특성, 테라타입)가 드러날 때마다
세트 단위로 상대를 추론하여 MCTS의 상대 시뮬레이션 정확도를 높임.
"""

from __future__ import annotations

from typing import Any

from data_loader import GameData, _to_id, STAT_KEYS, NATURES, TYPE_TO_IDX


# ═══════════════════════════════════════════════════════════════
#  SET_DB: BSS Top 30 포켓몬별 경쟁 세트 정의
#  데이터 소스: bss_stats.json 아이템/기술 분포 클러스터링 + Smogon 분석
# ═══════════════════════════════════════════════════════════════

SET_DB: dict[str, list[dict]] = {
    # ── 1. Koraidon ───────────────────────────────────────────
    "koraidon": [
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["flareblitz", "closecombat", "outrage", "uturn"],
            "item": "choicescarf",
            "ability": "orichalcumpulse",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Fire",
            "weight": 0.293,
        },
        {
            "name": "Loaded Dice",
            "key_item": "loadeddice",
            "key_moves": {"scaleshot"},
            "moves": ["scaleshot", "closecombat", "flamecharge", "swordsdance"],
            "item": "loadeddice",
            "ability": "orichalcumpulse",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Fire",
            "weight": 0.233,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"drainpunch"},
            "moves": ["flareblitz", "closecombat", "drainpunch", "uturn"],
            "item": "assaultvest",
            "ability": "orichalcumpulse",
            "nature": "Jolly",
            "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Steel",
            "weight": 0.180,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": {"collisioncourse"},
            "moves": ["flareblitz", "closecombat", "collisioncourse", "uturn"],
            "item": "choiceband",
            "ability": "orichalcumpulse",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Fire",
            "weight": 0.170,
        },
    ],
    # ── 2. Ting-Lu ────────────────────────────────────────────
    "tinglu": [
        {
            "name": "Sitrus Berry Wall",
            "key_item": "sitrusberry",
            "key_moves": {"whirlwind", "stealthrock"},
            "moves": ["earthquake", "whirlwind", "stealthrock", "ruination"],
            "item": "sitrusberry",
            "ability": "vesselofruin",
            "nature": "Impish",
            "evs": {"hp": 244, "atk": 4, "def": 36, "spa": 0, "spd": 204, "spe": 20},
            "tera_type": "Poison",
            "weight": 0.519,
        },
        {
            "name": "Leftovers Wall",
            "key_item": "leftovers",
            "key_moves": {"spikes", "rest"},
            "moves": ["earthquake", "spikes", "rest", "ruination"],
            "item": "leftovers",
            "ability": "vesselofruin",
            "nature": "Impish",
            "evs": {"hp": 244, "atk": 4, "def": 116, "spa": 0, "spd": 132, "spe": 12},
            "tera_type": "Fairy",
            "weight": 0.218,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"heavyslam"},
            "moves": ["earthquake", "heavyslam", "ruination", "bulldoze"],
            "item": "assaultvest",
            "ability": "vesselofruin",
            "nature": "Impish",
            "evs": {"hp": 244, "atk": 4, "def": 36, "spa": 0, "spd": 204, "spe": 20},
            "tera_type": "Steel",
            "weight": 0.190,
        },
    ],
    # ── 3. Calyrex-Shadow ─────────────────────────────────────
    "calyrexshadow": [
        {
            "name": "Focus Sash Nasty Plot",
            "key_item": "focussash",
            "key_moves": {"nastyplot"},
            "moves": ["astralbarrage", "nastyplot", "drainingkiss", "grassknot"],
            "item": "focussash",
            "ability": "asonespectrier",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 4, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.355,
        },
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": {"trick"},
            "moves": ["astralbarrage", "trick", "psychic", "encore"],
            "item": "choicescarf",
            "ability": "asonespectrier",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.264,
        },
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": set(),
            "moves": ["astralbarrage", "psyshock", "grassknot", "terablast"],
            "item": "choicespecs",
            "ability": "asonespectrier",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Grass",
            "weight": 0.143,
        },
    ],
    # ── 4. Flutter Mane ───────────────────────────────────────
    "fluttermane": [
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": {"powergem"},
            "moves": ["moonblast", "shadowball", "powergem", "mysticalfire"],
            "item": "choicespecs",
            "ability": "protosynthesis",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.397,
        },
        {
            "name": "Focus Sash Support",
            "key_item": "focussash",
            "key_moves": {"taunt", "thunderwave"},
            "moves": ["moonblast", "thunderwave", "taunt", "shadowball"],
            "item": "focussash",
            "ability": "protosynthesis",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.329,
        },
        {
            "name": "Booster Energy",
            "key_item": "boosterenergy",
            "key_moves": {"perishsong", "hex"},
            "moves": ["moonblast", "hex", "perishsong", "disarmingvoice"],
            "item": "boosterenergy",
            "ability": "protosynthesis",
            "nature": "Timid",
            "evs": {"hp": 100, "atk": 0, "def": 108, "spa": 148, "spd": 4, "spe": 148},
            "tera_type": "Water",
            "weight": 0.212,
        },
    ],
    # ── 5. Miraidon ───────────────────────────────────────────
    "miraidon": [
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["electrodrift", "dracometeor", "voltswitch", "dazzlinggleam"],
            "item": "choicescarf",
            "ability": "hadronengine",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.311,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"uturn"},
            "moves": ["electrodrift", "dracometeor", "uturn", "terablast"],
            "item": "assaultvest",
            "ability": "hadronengine",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 4, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Ice",
            "weight": 0.242,
        },
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": set(),
            "moves": ["electrodrift", "dracometeor", "voltswitch", "overheat"],
            "item": "choicespecs",
            "ability": "hadronengine",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "tera_type": "Electric",
            "weight": 0.173,
        },
    ],
    # ── 6. Chien-Pao ─────────────────────────────────────────
    "chienpao": [
        {
            "name": "Focus Sash Swords Dance",
            "key_item": "focussash",
            "key_moves": {"swordsdance"},
            "moves": ["iciclecrash", "suckerpunch", "swordsdance", "sacredsword"],
            "item": "focussash",
            "ability": "swordofruin",
            "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Electric",
            "weight": 0.525,
        },
        {
            "name": "Life Orb",
            "key_item": "lifeorb",
            "key_moves": {"iceshard"},
            "moves": ["iciclecrash", "suckerpunch", "iceshard", "terablast"],
            "item": "lifeorb",
            "ability": "swordofruin",
            "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Dark",
            "weight": 0.218,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": {"crunch"},
            "moves": ["iciclecrash", "crunch", "iceshard", "suckerpunch"],
            "item": "choiceband",
            "ability": "swordofruin",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Fire",
            "weight": 0.072,
        },
    ],
    # ── 7. Ho-Oh ──────────────────────────────────────────────
    "hooh": [
        {
            "name": "Heavy-Duty Boots Wall",
            "key_item": "heavydutyboots",
            "key_moves": {"recover", "whirlwind"},
            "moves": ["sacredfire", "recover", "whirlwind", "bravebird"],
            "item": "heavydutyboots",
            "ability": "regenerator",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 236, "spa": 0, "spd": 0, "spe": 20},
            "tera_type": "Fairy",
            "weight": 0.683,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": {"earthquake"},
            "moves": ["sacredfire", "bravebird", "earthquake", "terablast"],
            "item": "choiceband",
            "ability": "regenerator",
            "nature": "Jolly",
            "evs": {"hp": 108, "atk": 180, "def": 4, "spa": 0, "spd": 4, "spe": 212},
            "tera_type": "Normal",
            "weight": 0.112,
        },
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["sacredfire", "bravebird", "earthquake", "uturn"],
            "item": "choicescarf",
            "ability": "regenerator",
            "nature": "Jolly",
            "evs": {"hp": 108, "atk": 180, "def": 4, "spa": 0, "spd": 4, "spe": 212},
            "tera_type": "Ground",
            "weight": 0.064,
        },
    ],
    # ── 8. Urshifu-Rapid-Strike ───────────────────────────────
    "urshifurapidstrike": [
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": set(),
            "moves": ["surgingstrikes", "closecombat", "aquajet", "uturn"],
            "item": "choiceband",
            "ability": "unseenfist",
            "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Water",
            "weight": 0.465,
        },
        {
            "name": "Focus Sash",
            "key_item": "focussash",
            "key_moves": {"swordsdance"},
            "moves": ["surgingstrikes", "closecombat", "aquajet", "swordsdance"],
            "item": "focussash",
            "ability": "unseenfist",
            "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Water",
            "weight": 0.199,
        },
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["surgingstrikes", "closecombat", "aquajet", "uturn"],
            "item": "choicescarf",
            "ability": "unseenfist",
            "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.125,
        },
    ],
    # ── 9. Glimmora ───────────────────────────────────────────
    "glimmora": [
        {
            "name": "Red Card Lead",
            "key_item": "redcard",
            "key_moves": {"stealthrock", "toxicspikes"},
            "moves": ["mortalspin", "stealthrock", "powergem", "mudshot"],
            "item": "redcard",
            "ability": "toxicdebris",
            "nature": "Timid",
            "evs": {"hp": 132, "atk": 0, "def": 124, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.314,
        },
        {
            "name": "Focus Sash Lead",
            "key_item": "focussash",
            "key_moves": {"endure"},
            "moves": ["mortalspin", "stealthrock", "powergem", "endure"],
            "item": "focussash",
            "ability": "toxicdebris",
            "nature": "Timid",
            "evs": {"hp": 244, "atk": 0, "def": 4, "spa": 4, "spd": 4, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.301,
        },
        {
            "name": "Air Balloon Lead",
            "key_item": "airballoon",
            "key_moves": set(),
            "moves": ["mortalspin", "stealthrock", "powergem", "mudshot"],
            "item": "airballoon",
            "ability": "toxicdebris",
            "nature": "Timid",
            "evs": {"hp": 132, "atk": 0, "def": 124, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Dark",
            "weight": 0.242,
        },
    ],
    # ── 10. Landorus-Therian ──────────────────────────────────
    "landorustherian": [
        {
            "name": "Rocky Helmet Pivot",
            "key_item": "rockyhelmet",
            "key_moves": {"stealthrock", "taunt"},
            "moves": ["earthquake", "uturn", "stealthrock", "taunt"],
            "item": "rockyhelmet",
            "ability": "intimidate",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 212, "spa": 0, "spd": 4, "spe": 40},
            "tera_type": "Water",
            "weight": 0.425,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"rocktomb", "crunch"},
            "moves": ["earthquake", "uturn", "rocktomb", "crunch"],
            "item": "assaultvest",
            "ability": "intimidate",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 4, "def": 252, "spa": 0, "spd": 0, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.182,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": {"stoneedge"},
            "moves": ["earthquake", "uturn", "stoneedge", "crunch"],
            "item": "choiceband",
            "ability": "intimidate",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.148,
        },
    ],
    # ── 11. Dondozo ───────────────────────────────────────────
    "dondozo": [
        {
            "name": "Leftovers RestTalk",
            "key_item": "leftovers",
            "key_moves": {"rest", "protect"},
            "moves": ["wavecrash", "fissure", "rest", "protect"],
            "item": "leftovers",
            "ability": "unaware",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Fairy",
            "weight": 0.668,
        },
        {
            "name": "Rocky Helmet",
            "key_item": "rockyhelmet",
            "key_moves": {"yawn"},
            "moves": ["wavecrash", "yawn", "protect", "fissure"],
            "item": "rockyhelmet",
            "ability": "unaware",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 4, "def": 252, "spa": 0, "spd": 0, "spe": 0},
            "tera_type": "Grass",
            "weight": 0.228,
        },
    ],
    # ── 12. Garganacl ─────────────────────────────────────────
    "garganacl": [
        {
            "name": "Leftovers Stall",
            "key_item": "leftovers",
            "key_moves": {"saltcure", "recover", "protect"},
            "moves": ["saltcure", "recover", "protect", "stealthrock"],
            "item": "leftovers",
            "ability": "purifyingsalt",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 4, "def": 76, "spa": 0, "spd": 148, "spe": 28},
            "tera_type": "Water",
            "weight": 0.70,
        },
        {
            "name": "Leftovers Curse",
            "key_item": "leftovers",
            "key_moves": {"curse", "substitute"},
            "moves": ["saltcure", "recover", "curse", "substitute"],
            "item": "leftovers",
            "ability": "purifyingsalt",
            "nature": "Careful",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spd": 236, "spe": 20},
            "tera_type": "Ghost",
            "weight": 0.25,
        },
    ],
    # ── 13. Zacian-Crowned ────────────────────────────────────
    "zaciancrowned": [
        {
            "name": "Bulky Swords Dance",
            "key_item": "rustedsword",
            "key_moves": {"swordsdance"},
            "moves": ["ironhead", "playrough", "swordsdance", "closecombat"],
            "item": "rustedsword",
            "ability": "intrepidsword",
            "nature": "Adamant",
            "evs": {"hp": 252, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.50,
        },
        {
            "name": "Fast Attacker",
            "key_item": "rustedsword",
            "key_moves": {"trailblaze", "wildcharge"},
            "moves": ["ironhead", "playrough", "trailblaze", "wildcharge"],
            "item": "rustedsword",
            "ability": "intrepidsword",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Ground",
            "weight": 0.50,
        },
    ],
    # ── 14. Gliscor ───────────────────────────────────────────
    "gliscor": [
        {
            "name": "Toxic Orb SubToxic",
            "key_item": "toxicorb",
            "key_moves": {"substitute", "toxic"},
            "moves": ["protect", "substitute", "toxic", "earthquake"],
            "item": "toxicorb",
            "ability": "poisonheal",
            "nature": "Jolly",
            "evs": {"hp": 228, "atk": 4, "def": 4, "spa": 0, "spd": 20, "spe": 252},
            "tera_type": "Steel",
            "weight": 0.50,
        },
        {
            "name": "Toxic Orb Taunt",
            "key_item": "toxicorb",
            "key_moves": {"taunt", "knockoff"},
            "moves": ["protect", "taunt", "knockoff", "earthquake"],
            "item": "toxicorb",
            "ability": "poisonheal",
            "nature": "Careful",
            "evs": {"hp": 228, "atk": 0, "def": 28, "spa": 0, "spd": 252, "spe": 0},
            "tera_type": "Water",
            "weight": 0.43,
        },
    ],
    # ── 15. Ursaluna-Bloodmoon ────────────────────────────────
    "ursalunabloodmoon": [
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"vacuumwave"},
            "moves": ["bloodmoon", "hypervoice", "earthpower", "vacuumwave"],
            "item": "assaultvest",
            "ability": "mindseye",
            "nature": "Modest",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 4},
            "tera_type": "Normal",
            "weight": 0.427,
        },
        {
            "name": "Lum Berry Calm Mind",
            "key_item": "lumberry",
            "key_moves": {"calmmind", "moonlight"},
            "moves": ["bloodmoon", "earthpower", "calmmind", "moonlight"],
            "item": "lumberry",
            "ability": "mindseye",
            "nature": "Modest",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.163,
        },
        {
            "name": "Silk Scarf",
            "key_item": "silkscarf",
            "key_moves": {"hypervoice"},
            "moves": ["bloodmoon", "hypervoice", "earthpower", "vacuumwave"],
            "item": "silkscarf",
            "ability": "mindseye",
            "nature": "Modest",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 4},
            "tera_type": "Normal",
            "weight": 0.136,
        },
    ],
    # ── 16. Dragonite ─────────────────────────────────────────
    "dragonite": [
        {
            "name": "Loaded Dice Scale Shot",
            "key_item": "loadeddice",
            "key_moves": {"scaleshot", "dragondance"},
            "moves": ["extremespeed", "scaleshot", "dragondance", "earthquake"],
            "item": "loadeddice",
            "ability": "multiscale",
            "nature": "Adamant",
            "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.249,
        },
        {
            "name": "Covert Cloak DD",
            "key_item": "covertcloak",
            "key_moves": {"dragondance", "roost"},
            "moves": ["extremespeed", "earthquake", "dragondance", "roost"],
            "item": "covertcloak",
            "ability": "multiscale",
            "nature": "Adamant",
            "evs": {"hp": 164, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 92},
            "tera_type": "Normal",
            "weight": 0.223,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": {"outrage"},
            "moves": ["extremespeed", "outrage", "earthquake", "ironhead"],
            "item": "choiceband",
            "ability": "multiscale",
            "nature": "Adamant",
            "evs": {"hp": 244, "atk": 252, "def": 12, "spa": 0, "spd": 0, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.202,
        },
    ],
    # ── 17. Clodsire ─────────────────────────────────────────
    "clodsire": [
        {
            "name": "Black Sludge Wall",
            "key_item": "blacksludge",
            "key_moves": {"recover", "toxic"},
            "moves": ["recover", "earthquake", "toxic", "protect"],
            "item": "blacksludge",
            "ability": "waterabsorb",
            "nature": "Careful",
            "evs": {"hp": 252, "atk": 0, "def": 100, "spa": 0, "spd": 156, "spe": 0},
            "tera_type": "Fairy",
            "weight": 0.493,
        },
        {
            "name": "Leftovers",
            "key_item": "leftovers",
            "key_moves": {"toxicspikes"},
            "moves": ["recover", "earthquake", "toxicspikes", "protect"],
            "item": "leftovers",
            "ability": "waterabsorb",
            "nature": "Calm",
            "evs": {"hp": 252, "atk": 0, "def": 108, "spa": 20, "spd": 124, "spe": 4},
            "tera_type": "Poison",
            "weight": 0.201,
        },
        {
            "name": "Sitrus Berry Unaware",
            "key_item": "sitrusberry",
            "key_moves": {"counter"},
            "moves": ["recover", "earthquake", "counter", "toxic"],
            "item": "sitrusberry",
            "ability": "unaware",
            "nature": "Careful",
            "evs": {"hp": 252, "atk": 0, "def": 4, "spa": 0, "spd": 252, "spe": 0},
            "tera_type": "Dark",
            "weight": 0.141,
        },
    ],
    # ── 18. Kyogre ────────────────────────────────────────────
    "kyogre": [
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": {"waterspout"},
            "moves": ["waterspout", "icebeam", "thunder", "surf"],
            "item": "choicespecs",
            "ability": "drizzle",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "tera_type": "Water",
            "weight": 0.239,
        },
        {
            "name": "Rocky Helmet",
            "key_item": "rockyhelmet",
            "key_moves": {"thunderwave"},
            "moves": ["surf", "icebeam", "thunderwave", "thunder"],
            "item": "rockyhelmet",
            "ability": "drizzle",
            "nature": "Modest",
            "evs": {"hp": 252, "atk": 0, "def": 164, "spa": 76, "spd": 4, "spe": 12},
            "tera_type": "Fairy",
            "weight": 0.230,
        },
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["waterspout", "icebeam", "thunder", "originpulse"],
            "item": "choicescarf",
            "ability": "drizzle",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Ground",
            "weight": 0.223,
        },
    ],
    # ── 19. Chi-Yu ────────────────────────────────────────────
    "chiyu": [
        {
            "name": "Choice Scarf",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["darkpulse", "overheat", "flamethrower", "psychic"],
            "item": "choicescarf",
            "ability": "beadsofruin",
            "nature": "Modest",
            "evs": {"hp": 100, "atk": 0, "def": 4, "spa": 164, "spd": 4, "spe": 236},
            "tera_type": "Fire",
            "weight": 0.478,
        },
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": {"fireblast"},
            "moves": ["darkpulse", "overheat", "fireblast", "terablast"],
            "item": "choicespecs",
            "ability": "beadsofruin",
            "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "tera_type": "Grass",
            "weight": 0.259,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"snarl"},
            "moves": ["darkpulse", "overheat", "flamethrower", "snarl"],
            "item": "assaultvest",
            "ability": "beadsofruin",
            "nature": "Modest",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 0},
            "tera_type": "Fire",
            "weight": 0.172,
        },
    ],
    # ── 20. Calyrex-Ice ───────────────────────────────────────
    "calyrexice": [
        {
            "name": "Leftovers Trick Room",
            "key_item": "leftovers",
            "key_moves": {"trickroom", "leechseed"},
            "moves": ["glaciallance", "trickroom", "leechseed", "protect"],
            "item": "leftovers",
            "ability": "asoneglastrier",
            "nature": "Brave",
            "evs": {"hp": 252, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Fire",
            "weight": 0.370,
        },
        {
            "name": "Loaded Dice",
            "key_item": "loadeddice",
            "key_moves": {"iciclespear", "bulletseed"},
            "moves": ["iciclespear", "highhorsepower", "bulletseed", "trickroom"],
            "item": "loadeddice",
            "ability": "asoneglastrier",
            "nature": "Brave",
            "evs": {"hp": 252, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 0},
            "tera_type": "Water",
            "weight": 0.360,
        },
    ],
    # ── 21. Iron Treads ───────────────────────────────────────
    "irontreads": [
        {
            "name": "Booster Energy",
            "key_item": "boosterenergy",
            "key_moves": {"rapidspin"},
            "moves": ["earthquake", "ironhead", "rapidspin", "knockoff"],
            "item": "boosterenergy",
            "ability": "quarkdrive",
            "nature": "Jolly",
            "evs": {"hp": 60, "atk": 196, "def": 20, "spa": 0, "spd": 92, "spe": 140},
            "tera_type": "Ground",
            "weight": 0.285,
        },
        {
            "name": "Choice Band",
            "key_item": "choiceband",
            "key_moves": set(),
            "moves": ["earthquake", "ironhead", "knockoff", "icespinner"],
            "item": "choiceband",
            "ability": "quarkdrive",
            "nature": "Jolly",
            "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Ground",
            "weight": 0.214,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"endeavor"},
            "moves": ["earthquake", "ironhead", "knockoff", "endeavor"],
            "item": "assaultvest",
            "ability": "quarkdrive",
            "nature": "Careful",
            "evs": {"hp": 252, "atk": 68, "def": 0, "spa": 0, "spd": 188, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.138,
        },
    ],
    # ── 22. Muk-Alola ─────────────────────────────────────────
    "mukalola": [
        {
            "name": "Leftovers Minimize",
            "key_item": "leftovers",
            "key_moves": {"minimize", "substitute"},
            "moves": ["knockoff", "minimize", "substitute", "drainpunch"],
            "item": "leftovers",
            "ability": "poisontouch",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 0, "spe": 4},
            "tera_type": "Grass",
            "weight": 0.421,
        },
        {
            "name": "Black Sludge",
            "key_item": "blacksludge",
            "key_moves": {"poisonjab"},
            "moves": ["knockoff", "poisonjab", "drainpunch", "shadowsneak"],
            "item": "blacksludge",
            "ability": "poisontouch",
            "nature": "Careful",
            "evs": {"hp": 252, "atk": 4, "def": 0, "spa": 0, "spd": 252, "spe": 0},
            "tera_type": "Poison",
            "weight": 0.371,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"clearsmog"},
            "moves": ["knockoff", "drainpunch", "poisonjab", "clearsmog"],
            "item": "assaultvest",
            "ability": "poisontouch",
            "nature": "Adamant",
            "evs": {"hp": 252, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Flying",
            "weight": 0.199,
        },
    ],
    # ── 23. Grimmsnarl ────────────────────────────────────────
    "grimmsnarl": [
        {
            "name": "Light Clay Screens",
            "key_item": "lightclay",
            "key_moves": {"reflect", "lightscreen"},
            "moves": ["spiritbreak", "thunderwave", "reflect", "lightscreen"],
            "item": "lightclay",
            "ability": "prankster",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 128, "spa": 0, "spd": 126, "spe": 4},
            "tera_type": "Steel",
            "weight": 0.608,
        },
        {
            "name": "Rocky Helmet Pivot",
            "key_item": "rockyhelmet",
            "key_moves": {"partingshot"},
            "moves": ["spiritbreak", "thunderwave", "partingshot", "taunt"],
            "item": "rockyhelmet",
            "ability": "prankster",
            "nature": "Impish",
            "evs": {"hp": 252, "atk": 0, "def": 128, "spa": 0, "spd": 126, "spe": 4},
            "tera_type": "Fire",
            "weight": 0.217,
        },
    ],
    # ── 24. Mimikyu ───────────────────────────────────────────
    "mimikyu": [
        {
            "name": "Rocky Helmet Curse",
            "key_item": "rockyhelmet",
            "key_moves": {"curse"},
            "moves": ["playrough", "curse", "shadowsneak", "trickroom"],
            "item": "rockyhelmet",
            "ability": "disguise",
            "nature": "Jolly",
            "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.302,
        },
        {
            "name": "Life Orb Attacker",
            "key_item": "lifeorb",
            "key_moves": {"swordsdance", "shadowclaw"},
            "moves": ["playrough", "shadowclaw", "shadowsneak", "swordsdance"],
            "item": "lifeorb",
            "ability": "disguise",
            "nature": "Adamant",
            "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Stellar",
            "weight": 0.236,
        },
        {
            "name": "Mental Herb Trick Room",
            "key_item": "mentalherb",
            "key_moves": {"trickroom"},
            "moves": ["playrough", "curse", "trickroom", "thunderwave"],
            "item": "mentalherb",
            "ability": "disguise",
            "nature": "Jolly",
            "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.176,
        },
    ],
    # ── 25. Archaludon ────────────────────────────────────────
    "archaludon": [
        {
            "name": "Sitrus Berry Wall",
            "key_item": "sitrusberry",
            "key_moves": {"bodypress", "stealthrock"},
            "moves": ["bodypress", "dracometeor", "stealthrock", "foulplay"],
            "item": "sitrusberry",
            "ability": "stamina",
            "nature": "Bold",
            "evs": {"hp": 252, "atk": 0, "def": 236, "spa": 0, "spd": 20, "spe": 0},
            "tera_type": "Fairy",
            "weight": 0.462,
        },
        {
            "name": "Rocky Helmet",
            "key_item": "rockyhelmet",
            "key_moves": {"thunderwave"},
            "moves": ["bodypress", "dracometeor", "thunderwave", "foulplay"],
            "item": "rockyhelmet",
            "ability": "stamina",
            "nature": "Bold",
            "evs": {"hp": 220, "atk": 0, "def": 236, "spa": 0, "spd": 0, "spe": 52},
            "tera_type": "Normal",
            "weight": 0.130,
        },
        {
            "name": "Power Herb Electroshot",
            "key_item": "powerherb",
            "key_moves": {"electroshot"},
            "moves": ["electroshot", "dracometeor", "flashcannon", "bodypress"],
            "item": "powerherb",
            "ability": "sturdy",
            "nature": "Modest",
            "evs": {"hp": 0, "atk": 0, "def": 4, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Water",
            "weight": 0.100,
        },
    ],
    # ── 26. Lunala ────────────────────────────────────────────
    "lunala": [
        {
            "name": "Rocky Helmet Wall",
            "key_item": "rockyhelmet",
            "key_moves": {"moonlight"},
            "moves": ["moongeistbeam", "moonblast", "moonlight", "calmmind"],
            "item": "rockyhelmet",
            "ability": "shadowshield",
            "nature": "Bold",
            "evs": {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Fairy",
            "weight": 0.381,
        },
        {
            "name": "Choice Specs",
            "key_item": "choicespecs",
            "key_moves": {"psyshock"},
            "moves": ["moongeistbeam", "moonblast", "psyshock", "icebeam"],
            "item": "choicespecs",
            "ability": "shadowshield",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.163,
        },
        {
            "name": "Power Herb",
            "key_item": "powerherb",
            "key_moves": {"meteorbeam"},
            "moves": ["moongeistbeam", "moonblast", "meteorbeam", "calmmind"],
            "item": "powerherb",
            "ability": "shadowshield",
            "nature": "Timid",
            "evs": {"hp": 4, "atk": 0, "def": 0, "spa": 252, "spd": 0, "spe": 252},
            "tera_type": "Fairy",
            "weight": 0.156,
        },
    ],
    # ── 27. Ditto ─────────────────────────────────────────────
    "ditto": [
        {
            "name": "Choice Scarf Imposter",
            "key_item": "choicescarf",
            "key_moves": set(),
            "moves": ["transform"],
            "item": "choicescarf",
            "ability": "imposter",
            "nature": "Naive",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.419,
        },
        {
            "name": "Quick Claw",
            "key_item": "quickclaw",
            "key_moves": set(),
            "moves": ["transform"],
            "item": "quickclaw",
            "ability": "imposter",
            "nature": "Naive",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.416,
        },
        {
            "name": "Focus Sash",
            "key_item": "focussash",
            "key_moves": set(),
            "moves": ["transform"],
            "item": "focussash",
            "ability": "imposter",
            "nature": "Naive",
            "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spd": 4, "spe": 252},
            "tera_type": "Normal",
            "weight": 0.131,
        },
    ],
    # ── 28. Lugia ─────────────────────────────────────────────
    "lugia": [
        {
            "name": "Leftovers SubWhirl",
            "key_item": "leftovers",
            "key_moves": {"substitute", "whirlwind"},
            "moves": ["recover", "psychicnoise", "whirlwind", "substitute"],
            "item": "leftovers",
            "ability": "multiscale",
            "nature": "Bold",
            "evs": {"hp": 236, "atk": 0, "def": 236, "spa": 4, "spd": 4, "spe": 28},
            "tera_type": "Normal",
            "weight": 0.844,
        },
        {
            "name": "Rocky Helmet",
            "key_item": "rockyhelmet",
            "key_moves": set(),
            "moves": ["recover", "psychicnoise", "whirlwind", "substitute"],
            "item": "rockyhelmet",
            "ability": "multiscale",
            "nature": "Bold",
            "evs": {"hp": 204, "atk": 0, "def": 252, "spa": 0, "spd": 0, "spe": 52},
            "tera_type": "Poison",
            "weight": 0.104,
        },
    ],
    # ── 29. Alomomola ─────────────────────────────────────────
    "alomomola": [
        {
            "name": "Rocky Helmet Wish",
            "key_item": "rockyhelmet",
            "key_moves": {"wish", "protect"},
            "moves": ["flipturn", "wish", "protect", "aquajet"],
            "item": "rockyhelmet",
            "ability": "regenerator",
            "nature": "Relaxed",
            "evs": {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0},
            "tera_type": "Fairy",
            "weight": 0.508,
        },
        {
            "name": "Covert Cloak",
            "key_item": "covertcloak",
            "key_moves": {"tickle"},
            "moves": ["flipturn", "wish", "protect", "tickle"],
            "item": "covertcloak",
            "ability": "regenerator",
            "nature": "Relaxed",
            "evs": {"hp": 252, "atk": 0, "def": 156, "spa": 0, "spd": 100, "spe": 0},
            "tera_type": "Normal",
            "weight": 0.244,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": {"mirrorcoat", "icywind"},
            "moves": ["flipturn", "aquajet", "mirrorcoat", "icywind"],
            "item": "assaultvest",
            "ability": "regenerator",
            "nature": "Relaxed",
            "evs": {"hp": 4, "atk": 0, "def": 252, "spa": 0, "spd": 252, "spe": 0},
            "tera_type": "Poison",
            "weight": 0.194,
        },
    ],
    # ── 30. Eternatus ─────────────────────────────────────────
    "eternatus": [
        {
            "name": "Leftovers Utility",
            "key_item": "leftovers",
            "key_moves": {"recover", "toxicspikes"},
            "moves": ["recover", "sludgebomb", "dynamaxcannon", "toxicspikes"],
            "item": "leftovers",
            "ability": "pressure",
            "nature": "Timid",
            "evs": {"hp": 76, "atk": 0, "def": 28, "spa": 148, "spd": 28, "spe": 228},
            "tera_type": "Fairy",
            "weight": 0.402,
        },
        {
            "name": "Covert Cloak",
            "key_item": "covertcloak",
            "key_moves": {"protect", "substitute"},
            "moves": ["sludgebomb", "flamethrower", "protect", "substitute"],
            "item": "covertcloak",
            "ability": "pressure",
            "nature": "Calm",
            "evs": {"hp": 252, "atk": 0, "def": 156, "spa": 4, "spd": 92, "spe": 4},
            "tera_type": "Fairy",
            "weight": 0.134,
        },
        {
            "name": "Assault Vest",
            "key_item": "assaultvest",
            "key_moves": set(),
            "moves": ["sludgebomb", "dynamaxcannon", "flamethrower", "shadowball"],
            "item": "assaultvest",
            "ability": "pressure",
            "nature": "Modest",
            "evs": {"hp": 156, "atk": 0, "def": 0, "spa": 0, "spd": 244, "spe": 108},
            "tera_type": "Fire",
            "weight": 0.101,
        },
    ],
}


# ═══════════════════════════════════════════════════════════════
#  SetInferencer: 베이지안 세트 추론 엔진
# ═══════════════════════════════════════════════════════════════

class SetInferencer:
    """관측 정보로 상대 포켓몬의 세트를 추론.

    추론 로직:
      1. 해당 포켓몬의 세트 목록 가져오기
      2. 각 세트에 대해 관측과 일치하는지 점수 매기기
      3. weight(사전확률) x 일치점수 → 사후확률
      4. 최고 확률 세트 반환 (confidence 포함)
      5. DB에 없는 포켓몬: None 반환 (기존 fallback 사용)
    """

    # 점수 가중치
    ITEM_MATCH_SCORE = 10.0     # 아이템 일치 시 큰 보너스
    MOVE_MATCH_SCORE = 3.0      # 기술이 세트에 있으면 +
    MOVE_MISMATCH_PENALTY = 0.0 # 관측된 기술이 세트에 없으면 세트 제외 (score=0)
    ABILITY_MATCH_SCORE = 2.0   # 특성 일치 보너스
    TERA_MATCH_SCORE = 2.0      # 테라 일치 보너스
    KEY_MOVE_SCORE = 5.0        # 핵심 식별 기술 보너스

    def __init__(self, game_data: GameData):
        self.gd = game_data
        self.db = SET_DB
        # 상대 팀에서 확정된 아이템 추적 (Item Clause)
        self.confirmed_items: dict[str, str] = {}  # species_id → item_id

    def reset(self):
        """새 배틀 시작 시 초기화."""
        self.confirmed_items.clear()

    def update_confirmed_item(self, species_id: str, item_id: str):
        """상대 포켓몬의 아이템이 확정되면 기록."""
        if item_id:
            self.confirmed_items[species_id] = item_id

    def infer_distribution(self, species_id: str,
                           observed_item: str = "",
                           observed_moves: list[str] | None = None,
                           observed_ability: str = "",
                           observed_tera: str = "",
                           max_sets: int = 3,
                           min_weight: float = 0.05) -> list[dict] | None:
        """모든 후보 세트 + 정규화된 사후확률 반환.

        Returns:
            [{"name", "moves", "item", "ability", "nature", "evs",
              "tera_type", "weight"}, ...] weight 내림차순, 합계=1.0
            DB에 없으면 None.
        """
        if observed_moves is None:
            observed_moves = []

        sets = self.db.get(species_id)
        if not sets:
            return None

        # Item Clause: 다른 팀원이 이미 확정한 아이템은 제외
        excluded_items: set[str] = set()
        for sp, itm in self.confirmed_items.items():
            if sp != species_id:
                excluded_items.add(itm)

        # 확정된 아이템 기록
        if observed_item:
            self.update_confirmed_item(species_id, observed_item)

        scores: list[float] = []
        for s in sets:
            score = self._score_set(
                s, observed_item, observed_moves,
                observed_ability, observed_tera, excluded_items,
                species_id,
            )
            scores.append(score)

        total = sum(scores)
        if total <= 0:
            return None

        # 사후확률 정규화
        probs = [sc / total for sc in scores]

        # (세트, 확률) 쌍 → 확률 내림차순 정렬
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])

        # min_weight 필터링 + 상위 max_sets개 선택
        candidates = [(i, p) for i, p in indexed if p >= min_weight]
        candidates = candidates[:max_sets]

        if not candidates:
            # 모두 필터링되면 최고 확률 1개만 반환
            best_idx = indexed[0][0]
            best_set = sets[best_idx]
            return [{
                "name": best_set["name"],
                "moves": list(best_set["moves"]),
                "item": best_set["item"],
                "ability": best_set.get("ability", ""),
                "nature": best_set.get("nature", "Hardy"),
                "evs": dict(best_set.get("evs", {k: 0 for k in STAT_KEYS})),
                "tera_type": best_set.get("tera_type", ""),
                "weight": 1.0,
            }]

        # 재정규화 (합=1.0)
        w_sum = sum(p for _, p in candidates)
        result = []
        for idx, prob in candidates:
            s = sets[idx]
            result.append({
                "name": s["name"],
                "moves": list(s["moves"]),
                "item": s["item"],
                "ability": s.get("ability", ""),
                "nature": s.get("nature", "Hardy"),
                "evs": dict(s.get("evs", {k: 0 for k in STAT_KEYS})),
                "tera_type": s.get("tera_type", ""),
                "weight": prob / w_sum,
            })

        return result

    def infer(self, species_id: str,
              observed_item: str = "",
              observed_moves: list[str] | None = None,
              observed_ability: str = "",
              observed_tera: str = "") -> dict | None:
        """관측 정보로 세트 확률 계산 → 최고 확률 세트 반환.

        Returns:
            dict with keys: name, moves, item, ability, nature, evs,
            tera_type, confidence.  DB에 없으면 None.
        """
        if observed_moves is None:
            observed_moves = []

        sets = self.db.get(species_id)
        if not sets:
            return None

        # Item Clause: 다른 팀원이 이미 확정한 아이템은 제외
        excluded_items: set[str] = set()
        for sp, itm in self.confirmed_items.items():
            if sp != species_id:
                excluded_items.add(itm)

        # 확정된 아이템 기록
        if observed_item:
            self.update_confirmed_item(species_id, observed_item)

        scores: list[float] = []
        for s in sets:
            score = self._score_set(
                s, observed_item, observed_moves,
                observed_ability, observed_tera, excluded_items,
                species_id,
            )
            scores.append(score)

        total = sum(scores)
        if total <= 0:
            return None

        # 사후확률 정규화
        probs = [sc / total for sc in scores]
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_set = sets[best_idx]

        return {
            "name": best_set["name"],
            "moves": list(best_set["moves"]),
            "item": best_set["item"],
            "ability": best_set.get("ability", ""),
            "nature": best_set.get("nature", "Hardy"),
            "evs": dict(best_set.get("evs", {k: 0 for k in STAT_KEYS})),
            "tera_type": best_set.get("tera_type", ""),
            "confidence": probs[best_idx],
        }

    def _score_set(self, s: dict,
                   observed_item: str,
                   observed_moves: list[str],
                   observed_ability: str,
                   observed_tera: str,
                   excluded_items: set[str],
                   species_id: str) -> float:
        """개별 세트의 관측 일치 점수 계산."""
        # 사전확률 (Smogon 사용률)
        prior = s.get("weight", 0.1)
        likelihood = 1.0

        # Item Clause: 이 세트의 아이템이 다른 팀원에 의해 사용 중이면 제외
        if s["item"] in excluded_items and species_id not in self.confirmed_items:
            return 0.0

        # 아이템 일치 검사
        if observed_item:
            if s["item"] == observed_item:
                likelihood *= self.ITEM_MATCH_SCORE
            else:
                # 아이템이 다르면 이 세트가 아님 → 제외
                return 0.0

        # 기술 일치 검사
        set_moves = set(s["moves"])
        for mv in observed_moves:
            if mv in set_moves:
                likelihood *= self.MOVE_MATCH_SCORE
                # 핵심 식별 기술 보너스
                key_moves = s.get("key_moves", set())
                if mv in key_moves:
                    likelihood *= self.KEY_MOVE_SCORE
            else:
                # 관측된 기술이 세트에 없으면 세트 제외
                return 0.0

        # 특성 일치
        if observed_ability and s.get("ability"):
            if _to_id(observed_ability) == _to_id(s["ability"]):
                likelihood *= self.ABILITY_MATCH_SCORE

        # 테라 타입 일치
        if observed_tera and s.get("tera_type"):
            if observed_tera.lower() == s["tera_type"].lower():
                likelihood *= self.TERA_MATCH_SCORE

        return prior * likelihood

    # ─── 간접 추론: 데미지 역추산 ──────────────────────────────

    def infer_item_from_damage(self, attacker_species: str, move_id: str,
                               defender_species: str, actual_hp_pct_lost: float,
                               is_physical: bool) -> str | None:
        """데미지 비율로 아이템 추론.

        실제 데미지 / 예상 데미지(무아이템) 비율로 판단:
          ~1.5 → Choice Band/Specs
          ~1.3 → Life Orb
          ~1.0 → 비공격 아이템
        """
        expected_dmg = self._calc_expected_damage_pct(
            attacker_species, move_id, defender_species)
        if expected_dmg <= 0.01:
            return None

        ratio = actual_hp_pct_lost / expected_dmg

        if 1.4 < ratio < 1.65:
            return "choiceband" if is_physical else "choicespecs"
        if 1.2 < ratio < 1.4:
            return "lifeorb"
        return None

    def _calc_expected_damage_pct(self, attacker_species: str, move_id: str,
                                  defender_species: str) -> float:
        """무아이템 기준 예상 데미지 비율 (방어측 HP의 %).

        간이 데미지 공식 사용 (정밀도보다 속도 우선).
        """
        atk_dex = self.gd.get_pokemon(attacker_species)
        def_dex = self.gd.get_pokemon(defender_species)
        move = self.gd.get_move(move_id)
        if not atk_dex or not def_dex or not move:
            return 0.0

        bp = move.get("basePower", 0)
        if bp <= 0:
            return 0.0

        level = 50
        atk_base = atk_dex["baseStats"]
        def_base = def_dex["baseStats"]

        if move.get("is_physical"):
            a_stat = atk_base.get("atk", 100)
            d_stat = def_base.get("def", 100)
        else:
            a_stat = atk_base.get("spa", 100)
            d_stat = def_base.get("spd", 100)

        # 간이 스탯 계산 (31 IV, 적당한 EV, 보정 무시)
        a_val = ((2 * a_stat + 31 + 32) * level // 100 + 5)
        d_val = ((2 * d_stat + 31 + 32) * level // 100 + 5)
        hp_val = ((2 * def_base.get("hp", 100) + 31 + 32) * level // 100
                  + level + 10)

        # STAB
        atk_types = atk_dex.get("types", [])
        move_type = move.get("type", "Normal")
        stab = 1.5 if move_type in atk_types else 1.0

        # 타입 상성
        def_types = def_dex.get("types", [])
        eff = self.gd.effectiveness(move_type, def_types)

        damage = ((2 * level / 5 + 2) * bp * a_val / d_val / 50 + 2)
        damage *= stab * eff * 0.925  # 평균 난수
        return damage / hp_val

    # ─── 간접 추론: 스피드 역추산 ──────────────────────────────

    def infer_scarf_from_speed(self, opp_species: str, my_speed: int,
                               opp_went_first: bool, move_used: str) -> bool:
        """상대가 스카프인지 추론.

        조건: 상대가 선공 & 상대 최속이어도 느린데 선공 → 스카프.
        선제기(priority > 0)는 제외.
        """
        if not opp_went_first:
            return False

        move = self.gd.get_move(move_used)
        if move and move.get("priority", 0) > 0:
            return False

        opp_max_speed = self._calc_max_speed(opp_species)
        if opp_max_speed < my_speed:
            return True
        return False

    def _calc_max_speed(self, species: str) -> int:
        """해당 포켓몬의 최속 스탯 (252 Spe, +Spe 성격, 31 IV)."""
        dex = self.gd.get_pokemon(species)
        if not dex:
            return 200
        base_spe = dex["baseStats"].get("spe", 100)
        level = 50
        # (2*Base + 31 + 252/4) * Lv/100 + 5 → *1.1 (최속 성격)
        raw = (2 * base_spe + 31 + 63) * level // 100 + 5
        return int(raw * 1.1)


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """세트 추론 기본 검증."""
    print("=== Set Inferencer 검증 ===\n")

    gd = GameData(device="cpu")
    si = SetInferencer(gd)

    # 1. Flutter Mane + Choice Specs → Specs 세트
    r = si.infer("fluttermane", observed_item="choicespecs")
    if r:
        print(f"Flutter Mane + Specs: {r['name']} (conf={r['confidence']:.0%})")
        print(f"  predicted tera: {r['tera_type']}")
        print(f"  predicted moves: {r['moves']}")
    else:
        print("Flutter Mane: DB에 없음")

    print()

    # 2. Koraidon + flamecharge → Loaded Dice 세트
    r = si.infer("koraidon", observed_moves=["flamecharge"])
    if r:
        print(f"Koraidon + Flame Charge: {r['name']} (conf={r['confidence']:.0%})")
        print(f"  predicted item: {r['item']}")
        print(f"  predicted moves: {r['moves']}")
    else:
        print("Koraidon: DB에 없음")

    print()

    # 3. Calyrex-Shadow + trick → Scarf 세트
    r = si.infer("calyrexshadow", observed_moves=["trick"])
    if r:
        print(f"Calyrex-Shadow + Trick: {r['name']} (conf={r['confidence']:.0%})")
        print(f"  predicted item: {r['item']}")
    else:
        print("Calyrex-Shadow: DB에 없음")

    print()

    # 4. Item Clause: Koraidon이 스카프 확정 → Miraidon 스카프 제외
    si.update_confirmed_item("koraidon", "choicescarf")
    r = si.infer("miraidon")
    if r:
        print(f"Miraidon (Koraidon이 스카프 확정): {r['name']} (conf={r['confidence']:.0%})")
        print(f"  predicted item: {r['item']}")
    else:
        print("Miraidon: DB에 없음")

    print()

    # 5. DB에 없는 포켓몬
    r = si.infer("pikachu")
    print(f"Pikachu (DB에 없음): {r}")

    print()

    # 6. 스카프 스피드 추론
    is_scarf = si.infer_scarf_from_speed("tinglu", my_speed=200,
                                          opp_went_first=True,
                                          move_used="earthquake")
    print(f"Ting-Lu 선공 (내 speed=200): scarf={is_scarf}")

    # 7. 세트 DB 커버리지
    print(f"\n세트 DB: {len(SET_DB)} 포켓몬, "
          f"총 {sum(len(v) for v in SET_DB.values())} 세트")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
