"""팀 최적화 — 커버리지, 시너지, 사용률 기반.

- 입력: 원하는 포켓몬 (일부 또는 전체)
- 커버리지 분석: 18타입 공격/방어 매트릭스
- 시너지 스코어: Smogon teammates 데이터
- 세트 추천: 사용률 기반
- 빈 슬롯 채우기: 약점 보완 + 시너지 최적화
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from data_loader import (
    GameData, TYPE_TO_IDX, TYPES, NUM_TYPES, STAT_KEYS, _to_id,
)
from damage_calc import DamageCalculator
from battle_sim import Pokemon, make_pokemon_from_stats


# ═══════════════════════════════════════════════════════════════
#  커버리지 분석
# ═══════════════════════════════════════════════════════════════

@dataclass
class CoverageReport:
    """팀 커버리지 분석 결과."""
    offensive_coverage: dict[str, float]   # 타입별 최대 공격 효과
    defensive_weaknesses: dict[str, float] # 타입별 최대 약점 배율
    uncovered_types: list[str]             # 효과적으로 공격 못하는 타입
    critical_weaknesses: list[str]         # 팀 전체가 약한 타입
    score: float                           # 종합 점수 (높을수록 좋음)


class TeamBuilder:
    """팀 구성 최적화."""

    def __init__(self, game_data: GameData, format_name: str = "bss"):
        self.gd = game_data
        self.dc = DamageCalculator(game_data)
        self.format_name = format_name
        self.stats = (game_data.bss_stats if format_name == "bss"
                      else game_data.vgc_stats)
        self.team_size = 6  # 풀 팀
        self.pick_size = 3 if format_name == "bss" else 4  # 선출 수

    # ─── 커버리지 분석 ───────────────────────────────────────
    def analyze_coverage(self, team: list[str | Pokemon]) -> CoverageReport:
        """팀의 공격/방어 커버리지 분석."""
        chart = self.gd.type_chart

        # 공격 커버리지: 팀 기술로 각 타입에 줄 수 있는 최대 효과
        offensive = {t: 0.0 for t in TYPES}
        for member in team:
            if isinstance(member, str):
                member_id = _to_id(member)
                dex = self.gd.get_pokemon(member)
                if not dex:
                    continue
                # 해당 포켓몬의 인기 기술 타입 확인
                usage = self.stats.get(member_id, {})
                move_types = set()
                for move_id in usage.get("moves", {}).keys():
                    move = self.gd.get_move(move_id)
                    if move and not move["is_status"]:
                        move_types.add(move["type"])
                types = dex["types"]
            else:
                types = member.types
                move_types = set()
                for mid in member.moves:
                    move = self.gd.get_move(mid)
                    if move and not move["is_status"]:
                        move_types.add(move["type"])

            # STAB 타입도 포함
            for t in types:
                if t not in move_types:
                    move_types.add(t)

            for atk_type in move_types:
                atk_idx = TYPE_TO_IDX.get(atk_type, 0)
                for def_type in TYPES:
                    def_idx = TYPE_TO_IDX.get(def_type, 0)
                    eff = chart[atk_idx, def_idx].item()
                    offensive[def_type] = max(offensive[def_type], eff)

        # 방어 약점: 각 타입의 공격에 대한 팀의 최소 내성
        defensive = {t: 0.0 for t in TYPES}
        for atk_type in TYPES:
            atk_idx = TYPE_TO_IDX.get(atk_type, 0)
            min_eff = float("inf")
            for member in team:
                if isinstance(member, str):
                    dex = self.gd.get_pokemon(member)
                    if not dex:
                        continue
                    types = dex["types"]
                else:
                    types = member.types

                eff = 1.0
                for dt in types:
                    dt_idx = TYPE_TO_IDX.get(dt, 0)
                    eff *= chart[atk_idx, dt_idx].item()
                min_eff = min(min_eff, eff)  # 팀 내 가장 잘 받는 포켓몬

            defensive[atk_type] = min_eff if min_eff != float("inf") else 1.0

        # 분석
        uncovered = [t for t, v in offensive.items() if v <= 1.0]
        critical_weak = [t for t, v in defensive.items() if v >= 2.0]

        # 스코어 계산
        off_score = sum(min(v, 2.0) for v in offensive.values()) / NUM_TYPES
        def_score = sum(1.0 / max(v, 0.25) for v in defensive.values()) / NUM_TYPES
        score = off_score * 0.4 + def_score * 0.6

        return CoverageReport(
            offensive_coverage=offensive,
            defensive_weaknesses=defensive,
            uncovered_types=uncovered,
            critical_weaknesses=critical_weak,
            score=score,
        )

    # ─── 시너지 스코어 ───────────────────────────────────────
    def synergy_score(self, team: list[str]) -> float:
        """Smogon teammates 데이터 기반 시너지 점수."""
        team_ids = [_to_id(name) for name in team]
        total_synergy = 0.0
        pairs = 0

        for i, pid1 in enumerate(team_ids):
            stats1 = self.stats.get(pid1, {})
            teammates = stats1.get("teammates", {})
            for j, pid2 in enumerate(team_ids):
                if i >= j:
                    continue
                # 쌍방향 시너지
                syn = teammates.get(pid2, 0.0)
                stats2 = self.stats.get(pid2, {})
                syn += stats2.get("teammates", {}).get(pid1, 0.0)
                total_synergy += syn
                pairs += 1

        return total_synergy / max(pairs, 1)

    # ─── 세트 추천 ───────────────────────────────────────────
    def recommend_set(self, species: str) -> dict:
        """Smogon 사용률 기반 최적 세트 추천."""
        pid = _to_id(species)
        usage = self.stats.get(pid)
        if not usage:
            return {"error": f"No stats for {species}"}

        dex = self.gd.get_pokemon(species)
        if not dex:
            return {"error": f"Unknown pokemon: {species}"}

        # 아이템
        items = sorted(usage["items"].items(), key=lambda x: -x[1])
        # 기술
        moves = sorted(usage["moves"].items(), key=lambda x: -x[1])
        # 특성
        abilities = sorted(usage["abilities"].items(), key=lambda x: -x[1])
        # 스프레드
        spreads = sorted(usage.get("spreads", []),
                         key=lambda x: -x["weight"])

        return {
            "name": dex["name"],
            "types": dex["types"],
            "ability": abilities[0][0] if abilities else "",
            "item": items[0][0] if items else "",
            "moves": [m[0] for m in moves[:4]],
            "nature": spreads[0]["nature"] if spreads else "Adamant",
            "evs": spreads[0]["evs"] if spreads else {},
            "usage_pct": usage["usage_pct"],
            "all_items": items[:5],
            "all_moves": moves[:8],
            "all_spreads": spreads[:3],
        }

    # ─── 빈 슬롯 채우기 ─────────────────────────────────────
    def fill_team(self, core: list[str], n_total: int = 6,
                  top_n_candidates: int = 50) -> list[str]:
        """핵심 포켓몬을 기반으로 나머지 슬롯을 채움.

        Args:
            core: 이미 정해진 포켓몬 리스트
            n_total: 목표 팀 크기
            top_n_candidates: 후보군 크기

        Returns:
            완성된 팀 리스트
        """
        team = list(core)
        team_ids = set(_to_id(name) for name in team)

        # 후보: 사용률 TOP N (이미 팀에 있는 포켓몬 제외)
        candidates = sorted(
            self.stats.items(),
            key=lambda x: -x[1]["usage_pct"],
        )
        candidates = [(pid, info) for pid, info in candidates
                      if pid not in team_ids][:top_n_candidates]

        while len(team) < n_total and candidates:
            best_score = -float("inf")
            best_candidate = None

            coverage = self.analyze_coverage(team)

            for pid, info in candidates:
                if pid in team_ids:
                    continue

                # 시너지 점수
                test_team = team + [info["name"]]
                syn = self.synergy_score(test_team)

                # 커버리지 개선도
                test_cov = self.analyze_coverage(test_team)
                cov_improvement = test_cov.score - coverage.score

                # 약점 보완 보너스
                weakness_fix = 0.0
                for wt in coverage.critical_weaknesses:
                    dex = self.gd.get_pokemon(pid)
                    if dex:
                        eff = self.gd.effectiveness(wt, dex["types"])
                        if eff <= 0.5:
                            weakness_fix += 1.0

                # 사용률 가중
                usage_bonus = info["usage_pct"] * 0.5

                score = (syn * 3.0 + cov_improvement * 5.0 +
                         weakness_fix * 2.0 + usage_bonus)

                if score > best_score:
                    best_score = score
                    best_candidate = (pid, info)

            if best_candidate:
                pid, info = best_candidate
                team.append(info["name"])
                team_ids.add(pid)
                candidates = [(p, i) for p, i in candidates if p != pid]
            else:
                break

        return team

    # ─── 팀 프리뷰: 최적 선출 선택 ──────────────────────────
    def choose_leads(self, team: list[Pokemon],
                     opponent_team: list[str] | None = None) -> list[int]:
        """6마리 중 선출할 3~4마리 인덱스 선택.

        Args:
            team: 아군 6마리
            opponent_team: 상대 팀 (알려진 경우)

        Returns:
            선출할 포켓몬 인덱스 리스트
        """
        pick_size = self.pick_size
        n = len(team)

        if n <= pick_size:
            return list(range(n))

        # 모든 조합 평가
        from itertools import combinations

        best_score = -float("inf")
        best_combo = list(range(pick_size))

        for combo in combinations(range(n), pick_size):
            selected = [team[i] for i in combo]
            score = 0.0

            # 커버리지
            cov = self.analyze_coverage(selected)
            score += cov.score * 10.0

            # 상대 팀 대항력
            if opponent_team:
                for opp_name in opponent_team:
                    opp_dex = self.gd.get_pokemon(opp_name)
                    if not opp_dex:
                        continue
                    opp_types = opp_dex["types"]
                    # 우리 팀 중 상대를 효과적으로 칠 수 있는 포켓몬이 있는지
                    for poke in selected:
                        for mid in poke.moves:
                            move = self.gd.get_move(mid)
                            if move and not move["is_status"]:
                                eff = self.gd.effectiveness(move["type"], opp_types)
                                if eff >= 2.0:
                                    score += 0.5
                                    break

            # 사용률 가중
            for poke in selected:
                usage = self.stats.get(poke.species_id, {})
                score += usage.get("usage_pct", 0) * 2.0

            if score > best_score:
                best_score = score
                best_combo = list(combo)

        return best_combo

    # ─── 팀 리포트 출력 ──────────────────────────────────────
    def print_team_report(self, team: list[str]):
        """팀 분석 리포트 출력."""
        print(f"\n{'='*60}")
        print(f"  팀 분석 리포트 ({self.format_name.upper()})")
        print(f"{'='*60}")

        # 팀원 목록
        print(f"\n팀 구성:")
        for i, name in enumerate(team, 1):
            rec = self.recommend_set(name)
            if "error" not in rec:
                print(f"  {i}. {rec['name']} ({'/'.join(rec['types'])})")
                print(f"     특성: {rec['ability']} | 아이템: {rec['item']}")
                print(f"     기술: {', '.join(rec['moves'][:4])}")
                print(f"     성격: {rec['nature']} | 사용률: {rec['usage_pct']:.1%}")
            else:
                dex = self.gd.get_pokemon(name)
                if dex:
                    print(f"  {i}. {dex['name']} ({'/'.join(dex['types'])})")

        # 커버리지
        cov = self.analyze_coverage(team)
        print(f"\n공격 커버리지:")
        for t in TYPES:
            eff = cov.offensive_coverage[t]
            bar = "█" * int(eff * 5)
            marker = " ⚠" if eff <= 1.0 else ""
            print(f"  {t:10s} {eff:.1f}x {bar}{marker}")

        if cov.uncovered_types:
            print(f"\n⚠ 효과적 공격 불가 타입: {', '.join(cov.uncovered_types)}")
        if cov.critical_weaknesses:
            print(f"⚠ 팀 공통 약점: {', '.join(cov.critical_weaknesses)}")

        # 시너지
        syn = self.synergy_score(team)
        print(f"\n시너지 스코어: {syn:.4f}")
        print(f"커버리지 스코어: {cov.score:.3f}")

        print(f"\n{'='*60}")

    # ─── 메타 분석 ───────────────────────────────────────────
    def top_pokemon(self, n: int = 20) -> list[tuple[str, float]]:
        """사용률 상위 포켓몬."""
        return sorted(
            [(info["name"], info["usage_pct"])
             for info in self.stats.values()],
            key=lambda x: -x[1],
        )[:n]


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """팀 빌더 검증."""
    print("=== Team Builder 검증 ===\n")

    gd = GameData(device="cpu")

    # BSS
    print("--- BSS (63 싱글) ---")
    tb_bss = TeamBuilder(gd, "bss")

    print("사용률 TOP 10:")
    for name, pct in tb_bss.top_pokemon(10):
        print(f"  {name:20s} {pct:.1%}")

    # 코어 3마리로 팀 완성
    core = ["Koraidon", "Ting-Lu", "Flutter Mane"]
    full_team = tb_bss.fill_team(core, 6)
    print(f"\n코어: {core}")
    print(f"완성 팀: {full_team}")

    tb_bss.print_team_report(full_team)

    # 선출 추천
    print("\n--- 선출 추천 ---")
    team_pokemon = [make_pokemon_from_stats(gd, name, "bss") for name in full_team]
    leads = tb_bss.choose_leads(team_pokemon, ["Miraidon", "Gholdengo", "Great Tusk"])
    print(f"추천 선출: {[team_pokemon[i].name for i in leads]}")

    # VGC
    print("\n\n--- VGC (64 더블) ---")
    tb_vgc = TeamBuilder(gd, "vgc")

    print("사용률 TOP 10:")
    for name, pct in tb_vgc.top_pokemon(10):
        print(f"  {name:20s} {pct:.1%}")

    core_vgc = ["Flutter Mane", "Incineroar"]
    full_vgc = tb_vgc.fill_team(core_vgc, 6)
    print(f"\n코어: {core_vgc}")
    print(f"완성 팀: {full_vgc}")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
