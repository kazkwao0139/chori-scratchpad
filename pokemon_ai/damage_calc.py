"""Gen 9 데미지 계산기 — PyTorch GPU 벡터화.

핵심 공식:
  damage = ((2*level/5 + 2) * power * A/D) / 50 + 2
         × STAB × type_eff × random(0.85~1.0) × ...modifiers
"""

from __future__ import annotations

import torch
from typing import Optional

from data_loader import (
    GameData, TYPE_TO_IDX, TYPES, NUM_TYPES,
    ITEM_EFFECTS, ABILITY_EFFECTS, _TYPE_BOOST_ITEMS,
)


class DamageCalculator:
    """배치 데미지 계산기.  한 번에 N개의 데미지를 GPU에서 계산."""

    def __init__(self, game_data: GameData):
        self.gd = game_data
        self.device = game_data.device
        self.type_chart = game_data.type_chart  # (18, 18)

    # ─── 배치 데미지 계산 ──────────────────────────────────────
    def calc_damage_batch(
        self,
        level: torch.Tensor,          # (N,)
        power: torch.Tensor,          # (N,)
        atk_stat: torch.Tensor,       # (N,) — 물/특 이미 선택된 값
        def_stat: torch.Tensor,       # (N,)
        stab: torch.Tensor,           # (N,) — 1.0 or 1.5 or 2.0
        type_eff: torch.Tensor,       # (N,) — 0, 0.25, 0.5, 1, 2, 4
        is_crit: torch.Tensor,        # (N,) bool
        is_burned: torch.Tensor,      # (N,) bool — physical + burn
        weather_mod: torch.Tensor,    # (N,) — 1.0 or 1.5 or 0.5
        item_mod: torch.Tensor,       # (N,) — Life Orb 1.3 등
        ability_mod: torch.Tensor,    # (N,) — 추가 배율
        other_mod: torch.Tensor,      # (N,) — 기타 배율 (terrain 등)
        random_roll: Optional[torch.Tensor] = None,  # (N,) 0.85~1.0
    ) -> torch.Tensor:
        """(N,) 텐서로 배치 데미지 반환."""
        # 기본 데미지
        base = ((2.0 * level / 5.0 + 2.0) * power * atk_stat / def_stat) / 50.0 + 2.0

        # 크리티컬: 1.5배 (Gen 6+)
        crit_mod = torch.where(is_crit, torch.tensor(1.5, device=self.device),
                               torch.tensor(1.0, device=self.device))

        # 화상: 물리 공격 시 0.5배
        burn_mod = torch.where(is_burned, torch.tensor(0.5, device=self.device),
                               torch.tensor(1.0, device=self.device))

        # 랜덤 롤 (0.85 ~ 1.0)
        if random_roll is None:
            random_roll = torch.empty_like(base).uniform_(0.85, 1.0)

        damage = base * crit_mod * random_roll * stab * type_eff * burn_mod
        damage = damage * weather_mod * item_mod * ability_mod * other_mod

        return damage.floor().clamp(min=1)

    # ─── 편의 함수: 단일 계산 ──────────────────────────────────
    def calc_damage(
        self,
        attacker: dict,    # {types, stats, ability, item, status, boosts}
        defender: dict,     # {types, stats, ability, item, status, boosts}
        move: dict,         # from data_loader moves
        weather: str = "",  # sun, rain, sand, snow
        terrain: str = "",  # electric, grassy, misty, psychic
        is_crit: bool = False,
        tera_type: Optional[str] = None,  # 테라스탈 타입
    ) -> tuple[int, int, int]:
        """단일 데미지 계산 → (min, max, expected)."""
        if move["is_status"]:
            return (0, 0, 0)

        level = 50  # BSS/VGC는 레벨 50

        # 공격/방어 스탯 결정
        if move["is_physical"]:
            atk_key, def_key = "atk", "def"
        else:
            atk_key, def_key = "spa", "spd"

        atk_val = float(attacker["stats"][atk_key])
        def_val = float(defender["stats"][def_key])

        # 스탯 부스트 적용
        atk_boost = attacker.get("boosts", {}).get(atk_key, 0)
        def_boost = defender.get("boosts", {}).get(def_key, 0)

        if is_crit:
            atk_boost = max(atk_boost, 0)  # 크리 시 공격 하락 무시
            def_boost = min(def_boost, 0)  # 크리 시 방어 상승 무시

        atk_val *= self._boost_multiplier(atk_boost)
        def_val *= self._boost_multiplier(def_boost)

        # STAB 계산
        # 기술 타입은 항상 move["type"] 사용 (테라블래스트는 battle_sim에서 미리 처리)
        move_type = move["type"]
        original_types = attacker.get("original_types", attacker.get("types", []))

        if tera_type:
            # 테라 상태: 원래 타입 + 테라 타입 모두 STAB 부여
            if move_type == tera_type and move_type in original_types:
                stab = 2.0  # 테라 + 원래 타입 일치 → 2배 STAB
            elif move_type == tera_type or move_type in original_types:
                stab = 1.5  # 테라 또는 원래 타입 일치
            else:
                stab = 1.0  # 불일치
        else:
            atk_types = attacker.get("types", [])
            stab = 1.5 if move_type in atk_types else 1.0

        # 특성 STAB 보정 (적응력 등)
        atk_ability = attacker.get("ability", "")
        ability_fx = ABILITY_EFFECTS.get(atk_ability, {})
        all_stab_types = set(original_types)
        if tera_type:
            all_stab_types.add(tera_type)
        if ability_fx.get("stab_mod") and move_type in all_stab_types:
            stab = ability_fx["stab_mod"]

        # 타입 상성
        def_types = defender.get("types", [])
        type_eff = self.gd.effectiveness(move_type, def_types)

        if type_eff == 0:
            return (0, 0, 0)

        # 날씨 보정
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

        # 필드 보정
        terrain_mod = 1.0
        if terrain == "electric" and move_type == "Electric":
            terrain_mod = 1.3
        elif terrain == "grassy" and move_type == "Grass":
            terrain_mod = 1.3
        elif terrain == "psychic" and move_type == "Psychic":
            terrain_mod = 1.3
        elif terrain == "misty" and move_type == "Dragon":
            terrain_mod = 0.5

        # 아이템 보정
        item_mod = 1.0
        atk_item = attacker.get("item", "")
        item_fx = ITEM_EFFECTS.get(atk_item, {})
        if "damage_mod" in item_fx:
            item_mod *= item_fx["damage_mod"]
        if "super_eff_mod" in item_fx and type_eff > 1.0:
            item_mod *= item_fx["super_eff_mod"]

        # 타입 강화 아이템
        if atk_item in _TYPE_BOOST_ITEMS:
            boost_type, boost_val = _TYPE_BOOST_ITEMS[atk_item]
            if move_type == boost_type:
                item_mod *= boost_val

        # 방어측 아이템
        def_item = defender.get("item", "")
        def_item_fx = ITEM_EFFECTS.get(def_item, {})
        # Assault Vest: 특방 1.5배는 이미 스탯에 반영해야 하지만, 여기서 보정
        if "spd_mod" in def_item_fx and move["is_special"]:
            def_val *= def_item_fx["spd_mod"]
        if "def_mod" in def_item_fx and "spd_mod" in def_item_fx:
            # Eviolite
            if move["is_physical"]:
                def_val *= def_item_fx["def_mod"]
            else:
                def_val *= def_item_fx["spd_mod"]

        # 특성 보정
        ability_mod = 1.0
        # 공격측 특성
        if ability_fx.get("atk_mod") and move["is_physical"]:
            ability_mod *= ability_fx["atk_mod"]
        if ability_fx.get("low_bp_mod") and move["basePower"] <= ability_fx.get("bp_threshold", 60):
            ability_mod *= ability_fx["low_bp_mod"]
        if ability_fx.get("contact_mod") and move.get("flags", {}).get("contact"):
            ability_mod *= ability_fx["contact_mod"]
        if ability_fx.get("punch_mod") and move.get("flags", {}).get("punch"):
            ability_mod *= ability_fx["punch_mod"]
        if ability_fx.get("bite_mod") and move.get("flags", {}).get("bite"):
            ability_mod *= ability_fx["bite_mod"]
        if ability_fx.get("secondary_mod") and move.get("secondary"):
            ability_mod *= ability_fx["secondary_mod"]

        # 방어측 특성
        def_ability = defender.get("ability", "")
        def_ability_fx = ABILITY_EFFECTS.get(def_ability, {})
        if def_ability_fx.get("super_eff_mod") and type_eff > 1.0:
            ability_mod *= def_ability_fx["super_eff_mod"]
        if def_ability_fx.get("full_hp_damage_mod"):
            cur_hp = defender.get("cur_hp", defender["stats"]["hp"])
            max_hp = defender["stats"]["hp"]
            if cur_hp >= max_hp:
                ability_mod *= def_ability_fx["full_hp_damage_mod"]
        if def_ability_fx.get("special_damage_mod") and move["is_special"]:
            ability_mod *= def_ability_fx["special_damage_mod"]
        if def_ability_fx.get("fire_resist") and move_type == "Fire":
            ability_mod *= def_ability_fx["fire_resist"]
        if def_ability_fx.get("ice_resist") and move_type == "Ice":
            ability_mod *= def_ability_fx["ice_resist"]

        # Ruin abilities
        if def_ability_fx.get("foe_atk_mod") and move["is_physical"]:
            atk_val *= def_ability_fx["foe_atk_mod"]
        if def_ability_fx.get("foe_spa_mod") and move["is_special"]:
            atk_val *= def_ability_fx["foe_spa_mod"]
        if ability_fx.get("foe_def_mod") and move["is_physical"]:
            def_val *= ability_fx["foe_def_mod"]
        if ability_fx.get("foe_spd_mod") and move["is_special"]:
            def_val *= ability_fx["foe_spd_mod"]

        # 화상
        is_burned = (attacker.get("status") == "brn"
                     and move["is_physical"]
                     and not ability_fx.get("ignore_burn"))

        power = float(move["basePower"])
        if power <= 0:
            return (0, 0, 0)

        # 배치 텐서로 변환 → 16개 랜덤 롤
        N = 16
        dev = self.device
        t_level = torch.full((N,), level, dtype=torch.float32, device=dev)
        t_power = torch.full((N,), power, dtype=torch.float32, device=dev)
        t_atk = torch.full((N,), atk_val, dtype=torch.float32, device=dev)
        t_def = torch.full((N,), def_val, dtype=torch.float32, device=dev)
        t_stab = torch.full((N,), stab, dtype=torch.float32, device=dev)
        t_eff = torch.full((N,), type_eff, dtype=torch.float32, device=dev)
        t_crit = torch.full((N,), is_crit, dtype=torch.bool, device=dev)
        t_burn = torch.full((N,), is_burned, dtype=torch.bool, device=dev)
        t_weather = torch.full((N,), weather_mod, dtype=torch.float32, device=dev)
        t_item = torch.full((N,), item_mod, dtype=torch.float32, device=dev)
        t_ability = torch.full((N,), ability_mod, dtype=torch.float32, device=dev)
        t_other = torch.full((N,), terrain_mod, dtype=torch.float32, device=dev)

        # 0.85 ~ 1.0 (16단계)
        rolls = torch.linspace(0.85, 1.0, N, device=dev)

        damages = self.calc_damage_batch(
            t_level, t_power, t_atk, t_def, t_stab, t_eff,
            t_crit, t_burn, t_weather, t_item, t_ability, t_other,
            random_roll=rolls,
        )

        min_dmg = int(damages.min().item())
        max_dmg = int(damages.max().item())
        avg_dmg = int(damages.mean().item())
        return (min_dmg, max_dmg, avg_dmg)

    # ─── 배치: 여러 매치업 동시 계산 ──────────────────────────
    def calc_matchup_batch(
        self,
        attackers: list[dict],
        defenders: list[dict],
        moves: list[dict],
        weather: str = "",
        terrain: str = "",
    ) -> torch.Tensor:
        """여러 (attacker, defender, move) 조합을 한 번에 계산.
        Returns: (N, 3) — min, max, expected per matchup."""
        results = []
        for atk, dfn, mv in zip(attackers, defenders, moves):
            mn, mx, avg = self.calc_damage(atk, dfn, mv, weather, terrain)
            results.append([mn, mx, avg])
        return torch.tensor(results, dtype=torch.float32, device=self.device)

    @staticmethod
    def _boost_multiplier(stage: int) -> float:
        """스탯 변화 단계 → 배율."""
        stage = max(-6, min(6, stage))
        if stage >= 0:
            return (2 + stage) / 2.0
        else:
            return 2.0 / (2 - stage)

    # ─── 유틸: 실전 스탯 계산 ─────────────────────────────────
    @staticmethod
    def calc_stat(base: int, iv: int, ev: int, level: int,
                  nature_mod: float, is_hp: bool = False) -> int:
        """개체값/노력치/레벨/성격으로 실전 스탯 계산."""
        if is_hp:
            if base == 1:  # 누오가 아닌 귀신류 (1HP)
                return 1
            return int((2 * base + iv + ev // 4) * level / 100) + level + 10
        else:
            raw = int((2 * base + iv + ev // 4) * level / 100) + 5
            return int(raw * nature_mod)

    @staticmethod
    def calc_stats_from_spread(
        base_stats: dict[str, int],
        nature: str,
        evs: dict[str, int],
        level: int = 50,
        ivs: Optional[dict[str, int]] = None,
    ) -> dict[str, int]:
        """종족값 + 성격 + EV → 실전 스탯 dict."""
        from data_loader import NATURES, STAT_KEYS

        if ivs is None:
            ivs = {k: 31 for k in STAT_KEYS}

        plus, minus = NATURES.get(nature, (None, None))
        stats = {}
        for key in STAT_KEYS:
            nat_mod = 1.0
            if plus == key:
                nat_mod = 1.1
            elif minus == key:
                nat_mod = 0.9
            stats[key] = DamageCalculator.calc_stat(
                base_stats[key], ivs[key], evs.get(key, 0),
                level, nat_mod, is_hp=(key == "hp"),
            )
        return stats


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """데미지 계산 검증."""
    print("=== Damage Calculator 검증 ===\n")

    gd = GameData(device="cpu")
    dc = DamageCalculator(gd)

    # 코라이돈 (Adamant 252Atk) vs 딩루 (Impish 244HP/116Def)
    koraidon_base = gd.get_pokemon("koraidon")["baseStats"]
    tinglu_base = gd.get_pokemon("tinglu")["baseStats"]

    k_stats = dc.calc_stats_from_spread(
        koraidon_base, "Adamant", {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252}
    )
    t_stats = dc.calc_stats_from_spread(
        tinglu_base, "Impish", {"hp": 244, "atk": 4, "def": 116, "spa": 0, "spd": 132, "spe": 12}
    )

    attacker = {
        "types": ["Fighting", "Dragon"],
        "stats": k_stats,
        "ability": "orichalcumpulse",
        "item": "lifeorb",
        "status": None,
        "boosts": {},
    }
    defender = {
        "types": ["Dark", "Ground"],
        "stats": t_stats,
        "ability": "vesselofruin",
        "item": "leftovers",
        "status": None,
        "boosts": {},
    }

    # 코라이돈 드레인펀치 → 딩루
    drain_punch = gd.get_move("drainpunch")
    mn, mx, avg = dc.calc_damage(attacker, defender, drain_punch)
    t_hp = t_stats["hp"]
    print(f"코라이돈 드레인펀치 → 딩루: {mn}-{mx} ({mn*100//t_hp}%-{mx*100//t_hp}%)")

    # 코라이돈 충격파 (Close Combat) → 딩루
    cc = gd.get_move("closecombat")
    mn, mx, avg = dc.calc_damage(attacker, defender, cc)
    print(f"코라이돈 인파이트 → 딩루: {mn}-{mx} ({mn*100//t_hp}%-{mx*100//t_hp}%)")

    # 코라이돈 지진 → 딩루
    eq = gd.get_move("earthquake")
    mn, mx, avg = dc.calc_damage(attacker, defender, eq)
    print(f"코라이돈 지진 → 딩루: {mn}-{mx} ({mn*100//t_hp}%-{mx*100//t_hp}%)")

    # 스탯 계산 확인
    print(f"\n코라이돈 실전 스탯 (Adamant 252/252): {k_stats}")
    print(f"딩루 실전 스탯 (Impish 244/116/132): {t_stats}")

    # 배치 성능 테스트
    import time
    N = 10000
    dev = gd.device
    t = torch.full((N,), 50.0, device=dev)
    start = time.time()
    _ = dc.calc_damage_batch(
        t, torch.full((N,), 100.0, device=dev),
        torch.full((N,), 200.0, device=dev),
        torch.full((N,), 150.0, device=dev),
        torch.full((N,), 1.5, device=dev),
        torch.full((N,), 2.0, device=dev),
        torch.zeros(N, dtype=torch.bool, device=dev),
        torch.zeros(N, dtype=torch.bool, device=dev),
        torch.ones(N, device=dev),
        torch.ones(N, device=dev),
        torch.ones(N, device=dev),
        torch.ones(N, device=dev),
    )
    elapsed = time.time() - start
    print(f"\n배치 {N}개 계산: {elapsed*1000:.1f}ms")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
