"""매크로 탐색기 — C(6,3) 전수 탐색으로 유효 매크로 발견.

룰로 역할을 분류하는 대신, 20개 조합을 전부 평가해서
실행 가능한 매크로를 점수화.

3가지 패턴:
  setup_sweep — 세팅 후 스윕 가능한 에이스가 있는가
  break_clean — 브레이커+클리너 구조가 성립하는가
  cycle       — 앵커 2마리로 소모전이 가능한가
"""
import sys, os
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import GameData, ABILITY_EFFECTS
from battle_sim import load_sample_teams, Pokemon

# ═══════════════════════════════════════════════════════════════
#  기술 분류
# ═══════════════════════════════════════════════════════════════

SETUP_ATK = {
    "swordsdance", "dragondance", "nastyplot", "calmmind",
    "quiverdance", "bellydrum", "shellsmash", "bulkup",
    "coil", "shiftgear", "tidyup", "victorydance", "workup",
}
SETUP_DEF = {"irondefense", "amnesia", "curse", "cosmicpower"}
SETUP_SPE = {
    "dragondance", "quiverdance", "shellsmash", "shiftgear",
    "agility", "autotomize", "rockpolish", "flamecharge", "trailblaze",
}
RECOVERY = {
    "recover", "roost", "softboiled", "moonlight", "morningsun",
    "synthesis", "milkdrink", "slackoff", "shoreup", "rest",
    "strengthsap",
}
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport"}
PRIORITY = {
    "extremespeed", "aquajet", "iceshard", "machpunch",
    "bulletpunch", "shadowsneak", "suckerpunch", "quickattack",
    "accelerock", "jetpunch", "vacuumwave", "thunderclap",
}
HP_SCALE = {"waterspout", "eruption"}
SCREENS = {"reflect", "lightscreen", "auroraveil"}
CHOICE = {"choiceband", "choicespecs", "choicescarf"}


# ═══════════════════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════════════════

def _off(mon):
    return max(mon.stats.get("atk", 0), mon.stats.get("spa", 0))

def _spe(mon):
    return mon.stats.get("spe", 0)

def _bulk(mon):
    hp = mon.stats.get("hp", 0)
    d = mon.stats.get("def", 0)
    sd = mon.stats.get("spd", 0)
    return (hp * d + hp * sd) / 2000.0

def _ms(mon):
    return set(mon.moves)


# ═══════════════════════════════════════════════════════════════
#  패턴 1: 세팅→스윕
# ═══════════════════════════════════════════════════════════════

def score_setup_sweep(combo):
    """세팅 에이스 + 서포트 구조 점수."""
    best = 0.0
    best_info = None

    for i, ace in enumerate(combo):
        ms = _ms(ace)
        # 구애 → 세팅 불가
        if ace.item in CHOICE:
            continue

        has_atk_setup = bool(ms & SETUP_ATK) and _off(ace) >= 100
        has_def_setup = bool(ms & SETUP_DEF) and "bodypress" in ms
        if not (has_atk_setup or has_def_setup):
            continue

        # +2 화력
        if has_atk_setup:
            boosted = _off(ace) * 2
        else:
            boosted = ace.stats.get("def", 0) * 2  # bodypress

        # 속도
        spe = _spe(ace)
        has_spe_boost = bool(ms & SETUP_SPE)
        eff_spe = spe * 1.5 if has_spe_boost else spe
        has_prio = bool(ms & PRIORITY)

        # 에이스 점수
        ace_s = boosted / 300.0
        if eff_spe >= 250:
            ace_s *= 1.5
        elif eff_spe >= 180:
            ace_s *= 1.2
        elif has_prio:
            ace_s *= 1.1
        elif eff_spe < 100:
            ace_s *= 0.5

        # 기합의띠 보험
        if ace.item == "focussash":
            ace_s *= 1.2

        # 서포트 점수
        others = [combo[j] for j in range(3) if j != i]
        sup_s = 0.0
        for o in others:
            oms = _ms(o)
            ob = _bulk(o)
            oab = ABILITY_EFFECTS.get(o.ability, {})

            if oms & HAZARDS:     sup_s += 0.3
            if oms & SCREENS:     sup_s += 0.25
            if oms & PIVOTS:      sup_s += 0.15
            if o.item == "focussash": sup_s += 0.2
            if ob >= 30:          sup_s += 0.2
            elif ob >= 20:        sup_s += 0.1
            if oab.get("switch_heal"): sup_s += 0.15
            # 클리너 백업
            if o.item == "choicescarf" and _off(o) >= 130:
                sup_s += 0.2
            if bool(oms & PRIORITY) and _off(o) >= 130:
                sup_s += 0.15

        total = ace_s * (1 + sup_s)
        if total > best:
            best = total
            best_info = {
                "ace": ace.name,
                "boosted": int(boosted),
                "eff_spe": int(eff_spe),
                "prio": has_prio,
                "support": [o.name for o in others],
            }

    return best, best_info


# ═══════════════════════════════════════════════════════════════
#  패턴 2: 브레이크→클린
# ═══════════════════════════════════════════════════════════════

def score_break_clean(combo):
    """브레이커(즉시 고화력) + 클리너(빠른 마무리) 구조."""
    best = 0.0
    best_info = None

    for i, brk in enumerate(combo):
        # 브레이커: 높은 즉시 화력
        bp = _off(brk)
        if brk.item == "choiceband" and brk.stats.get("atk", 0) >= brk.stats.get("spa", 0):
            bp *= 1.5
        elif brk.item == "choicespecs" and brk.stats.get("spa", 0) >= brk.stats.get("atk", 0):
            bp *= 1.5
        elif brk.item == "lifeorb":
            bp *= 1.3
        if bp < 200:
            continue

        # 피벗 보너스
        if _ms(brk) & PIVOTS:
            bp *= 1.1

        for j, cln in enumerate(combo):
            if j == i:
                continue

            # 클리너: 속도 + 화력
            co = _off(cln)
            cs = _spe(cln)
            cms = _ms(cln)
            cln_s = 0.0

            if cln.item == "choicescarf" and co >= 120:
                cln_s = co * (cs * 1.5) / 200.0
            elif cs >= 180 and co >= 120:
                cln_s = co * cs / 200.0
            elif bool(cms & PRIORITY) and co >= 130:
                cln_s = co * 1.3
            # HP기 리드폭격 (클리너는 아니지만 break_clean에선 유효)
            elif cln.item == "choicescarf" and bool(cms & HP_SCALE):
                cln_s = co * cs / 200.0

            if cln_s < 80:
                continue

            # 3번째
            third = [combo[k] for k in range(3) if k != i and k != j]
            tb = 0.0
            if third:
                t = third[0]
                tms = _ms(t)
                if tms & HAZARDS: tb += 0.25
                if _bulk(t) >= 25: tb += 0.15
                if t.item == "focussash": tb += 0.15
                if tms & PIVOTS: tb += 0.1

            total = (bp / 300.0) * (cln_s / 200.0) * (1 + tb)
            if total > best:
                best = total
                best_info = {
                    "breaker": brk.name,
                    "brk_pwr": int(bp),
                    "cleaner": cln.name,
                    "cln_scr": int(cln_s),
                    "third": third[0].name if third else None,
                }

    return best, best_info


# ═══════════════════════════════════════════════════════════════
#  패턴 3: 사이클
# ═══════════════════════════════════════════════════════════════

def score_cycle(combo):
    """앵커 2마리 + 공격 1마리 사이클."""
    anchors = []
    for mon in combo:
        ms = _ms(mon)
        ab = ABILITY_EFFECTS.get(mon.ability, {})
        b = _bulk(mon)

        sus = 0.0
        if ms & RECOVERY: sus += 0.6
        if ab.get("switch_heal"): sus += 0.5
        if mon.ability == "poisonheal": sus += 0.5
        if ab.get("ignore_boosts"): sus += 0.2

        if sus >= 0.3 and b >= 18:
            anchors.append((mon, b, sus))

    if len(anchors) < 2:
        return 0.0, None

    a1, b1, s1 = anchors[0]
    a2, b2, s2 = anchors[1]

    # 물리벽+특수벽 시너지
    syn = 0.0
    d1, sd1 = a1.stats.get("def", 0), a1.stats.get("spd", 0)
    d2, sd2 = a2.stats.get("def", 0), a2.stats.get("spd", 0)
    if (d1 > sd1 + 30 and sd2 > d2 + 30) or (sd1 > d1 + 30 and d2 > sd2 + 30):
        syn = 0.3

    # 헤저드
    hz = 0.0
    for mon in combo:
        if _ms(mon) & HAZARDS:
            hz += 0.2

    # 공격 압력 (3번째)
    atk_s = 0.0
    for mon in combo:
        if mon.name != a1.name and mon.name != a2.name:
            atk_s = _off(mon) / 200.0

    score = (b1 + b2) / 60.0 * (s1 + s2) / 2.0 * (1 + syn + hz + atk_s * 0.3)
    return score, {
        "anchor1": a1.name,
        "anchor2": a2.name,
        "bulk": int(b1 + b2),
        "third": [m.name for m in combo
                   if m.name != a1.name and m.name != a2.name][:1],
    }


# ═══════════════════════════════════════════════════════════════
#  탐색 메인
# ═══════════════════════════════════════════════════════════════

def search_macros(team, top_per_pattern=3):
    """6마리에서 C(6,3)=20 전수 탐색 → 패턴별 상위 반환."""
    by_pattern = {"setup_sweep": [], "break_clean": [], "cycle": []}

    for combo in combinations(team, 3):
        combo = list(combo)
        names = tuple(sorted(p.name for p in combo))

        ss, ss_d = score_setup_sweep(combo)
        if ss > 0.3:
            by_pattern["setup_sweep"].append((names, ss, ss_d))

        bc, bc_d = score_break_clean(combo)
        if bc > 0.2:
            by_pattern["break_clean"].append((names, bc, bc_d))

        cy, cy_d = score_cycle(combo)
        if cy > 0.3:
            by_pattern["cycle"].append((names, cy, cy_d))

    # 패턴별 정렬 + 상위 N개
    result = {}
    for pat, entries in by_pattern.items():
        entries.sort(key=lambda x: -x[1])
        if entries:
            result[pat] = entries[:top_per_pattern]

    return result


# ═══════════════════════════════════════════════════════════════
#  출력
# ═══════════════════════════════════════════════════════════════

PAT = {"setup_sweep": "Setup>Sweep", "break_clean": "Break>Clean", "cycle": "Cycle"}

def _fmt_detail(pat, d):
    if pat == "setup_sweep":
        p = "+prio" if d["prio"] else ""
        return f"ace={d['ace']} (+2>{d['boosted']}off spe={d['eff_spe']}{p})"
    elif pat == "break_clean":
        return f"brk={d['breaker']}({d['brk_pwr']}) > cln={d['cleaner']}({d['cln_scr']})"
    else:
        t = d["third"][0] if d.get("third") else "-"
        return f"core={d['anchor1']}+{d['anchor2']} bulk={d['bulk']} 3rd={t}"

def print_team(idx, team, gd):
    names = " / ".join(p.name for p in team)
    print(f"\n{'='*65}")
    print(f"  TEAM {idx+1}: {names}")
    print(f"{'='*65}")

    macros = search_macros(team)

    if not macros:
        print("  (no viable macros found)")
        return

    for pat in ["setup_sweep", "break_clean", "cycle"]:
        entries = macros.get(pat, [])
        if not entries:
            continue
        print(f"\n  --- {PAT[pat]} ---")
        for i, (combo, score, detail) in enumerate(entries):
            combo_str = " + ".join(combo)
            info = _fmt_detail(pat, detail)
            print(f"  [{i+1}] {score:.2f}  {combo_str}")
            print(f"       {info}")


def main():
    gd = GameData(device="cpu")
    path = os.path.join(os.path.dirname(__file__), "data", "sample_teams.txt")
    teams = load_sample_teams(gd, path)

    print(f"Loaded {len(teams)} teams")
    print(f"C(6,3)=20 combos x 3 patterns = 60 checks per team\n")

    for i, team in enumerate(teams):
        print_team(i, team, gd)

    print(f"\n{'='*65}\n  DONE\n{'='*65}")


if __name__ == "__main__":
    main()
