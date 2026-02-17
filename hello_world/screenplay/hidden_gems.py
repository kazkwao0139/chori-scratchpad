from pathlib import Path
"""
Hidden Gems: 평작 중에서 각본이 명작급인 영화 찾기.

4개 핵심 지표 — 방향은 데이터에서 자동 도출:
  명작(≥8.0) 평균 vs 망작(<4.0) 평균 비교 → 명작 방향이 "좋은 각본"

각 평작(5.0-7.0)을 명작 분포 기준 z-score로 점수화.
최소 15씬 이상인 영화만 포함 (안정적인 메트릭 확보).
"""

import json
import sys
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CHECKPOINT = f'{BASE}/screenplay/plot_entropy_checkpoint.json'

KEY_METRICS = ['repeat_ratio', 'arc_shift', 'setup_front', 'bi_entropy']
MIN_SCENES = 15  # 안정적인 메트릭을 위한 최소 씬 수


def main():
    cp = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
    valid = {k: v for k, v in cp.items()
             if isinstance(v, dict) and v.get('labels') and v.get('rating')
             and v.get('n_scenes', 0) >= MIN_SCENES}

    all_movies = {k: v for k, v in cp.items()
                  if isinstance(v, dict) and v.get('labels') and v.get('rating')}

    good = {k: v for k, v in valid.items() if v['rating'] >= 8.0}
    bad  = {k: v for k, v in valid.items() if v['rating'] < 4.0}
    mid  = {k: v for k, v in valid.items() if 5.0 <= v['rating'] <= 7.0}

    # 씬 필터링 전후 비교
    all_mid = {k: v for k, v in all_movies.items() if 5.0 <= v['rating'] <= 7.0}

    sep = "=" * 70
    print(sep)
    print("  HIDDEN GEMS — 평작 중 명작급 각본 찾기")
    print(sep)
    print(f"\n  최소 씬 수: {MIN_SCENES} (노이즈 제거)")
    print(f"  명작 (>=8.0): {len(good)}편  (기준선)")
    print(f"  망작 (<4.0):  {len(bad)}편  (대조군)")
    print(f"  평작 (5-7):   {len(mid)}편  (탐색 대상, 필터 전 {len(all_mid)}편)")

    # ── Step 1: 방향 자동 도출 ──
    print(f"\n{'─' * 70}")
    print("  Step 1: 명작 vs 망작 — 방향 자동 도출")
    print(f"{'─' * 70}")

    stats = {}
    for m in KEY_METRICS:
        gv = [v[m] for v in good.values()]
        bv = [v[m] for v in bad.values()]
        g_mean = sum(gv) / len(gv)
        b_mean = sum(bv) / len(bv)
        g_std = (sum((x - g_mean)**2 for x in gv) / max(len(gv)-1, 1)) ** 0.5
        # 방향: 명작이 더 높으면 +1, 더 낮으면 -1
        direction = +1 if g_mean > b_mean else -1
        stats[m] = {
            'g_mean': g_mean, 'g_std': g_std, 'b_mean': b_mean,
            'direction': direction,
        }

    print(f"\n  {'Metric':>14s} {'명작':>8s} {'망작':>8s} {'차이':>8s} {'방향':>14s}")
    print(f"  {'─'*56}")
    for m in KEY_METRICS:
        s = stats[m]
        diff = s['g_mean'] - s['b_mean']
        arrow = "높을수록 명작" if s['direction'] > 0 else "낮을수록 명작"
        print(f"  {m:>14s} {s['g_mean']:8.4f} {s['b_mean']:8.4f} {diff:+8.4f}   {arrow}")

    print(f"\n  해석:")
    print(f"    repeat_ratio ↑명작: 좋은 각본은 구조적 반복 패턴이 있다 (Scene-Sequel 사이클)")
    print(f"    arc_shift    ↓명작: 좋은 각본은 전후반 밸런스가 일정하다")
    print(f"    setup_front  ↓명작: 좋은 각본은 세팅을 전체에 걸쳐 배치한다")
    print(f"    bi_entropy   ↑명작: 좋은 각본은 이벤트 전환이 다양하다")

    # ── Step 2: 평작 점수화 ──
    scores = {}
    for title, v in mid.items():
        total_z = 0
        metric_details = {}
        for m in KEY_METRICS:
            s = stats[m]
            if s['g_std'] == 0:
                z = 0
            else:
                z = (v[m] - s['g_mean']) / s['g_std']
            adjusted_z = s['direction'] * z
            total_z += adjusted_z
            metric_details[m] = {'value': v[m], 'z': z, 'adj_z': adjusted_z}

        avg_z = total_z / len(KEY_METRICS)
        scores[title] = {
            'rating': v['rating'],
            'score': avg_z,
            'details': metric_details,
            'n_scenes': v.get('n_scenes', 0),
        }

    # ── Step 3: 랭킹 ──
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

    print(f"\n{sep}")
    print("  TOP 20: 평작인데 각본이 명작급인 영화")
    print(sep)
    print(f"\n  점수: 0 = 명작 평균과 동일, 양수 = 명작보다 더 좋음, 음수 = 명작보다 못함")
    print(f"\n  {'#':>3s} {'Score':>6s} {'IMDB':>5s} {'Scenes':>6s} {'Title':<32s} {'rep':>6s} {'arc':>6s} {'set':>6s} {'bi_e':>6s}")
    print(f"  {'─'*80}")

    for i, (title, info) in enumerate(ranked[:20]):
        d = info['details']
        print(f"  {i+1:3d} {info['score']:+6.2f} {info['rating']:5.1f} {info['n_scenes']:6d} {title[:32]:<32s} "
              f"{d['repeat_ratio']['value']:6.3f} {d['arc_shift']['value']:6.3f} "
              f"{d['setup_front']['value']:6.3f} {d['bi_entropy']['value']:6.3f}")

    # ── 명작 기준 비교 ──
    print(f"\n  [참고] 명작 평균:  ", end="")
    for m in KEY_METRICS:
        print(f"{m}={stats[m]['g_mean']:.3f}  ", end="")
    print()

    # ── Step 4: 각본이 최악인 평작 ──
    print(f"\n{sep}")
    print("  BOTTOM 10: 평작인데 각본이 망작급인 영화")
    print(sep)
    print(f"\n  {'#':>3s} {'Score':>6s} {'IMDB':>5s} {'Scenes':>6s} {'Title':<32s} {'rep':>6s} {'arc':>6s} {'set':>6s} {'bi_e':>6s}")
    print(f"  {'─'*80}")

    for i, (title, info) in enumerate(ranked[-10:][::-1]):
        d = info['details']
        print(f"  {i+1:3d} {info['score']:+6.2f} {info['rating']:5.1f} {info['n_scenes']:6d} {title[:32]:<32s} "
              f"{d['repeat_ratio']['value']:6.3f} {d['arc_shift']['value']:6.3f} "
              f"{d['setup_front']['value']:6.3f} {d['bi_entropy']['value']:6.3f}")

    # ── Step 5: GAP — 각본 대비 평점이 낮은 영화 ──
    print(f"\n{sep}")
    print("  GAP ANALYSIS: '묻힌 영화' — 각본은 좋은데 평점이 낮은")
    print(sep)

    mid_ratings = [v['rating'] for v in scores.values()]
    r_mean = sum(mid_ratings) / len(mid_ratings)
    r_std = (sum((r - r_mean)**2 for r in mid_ratings) / max(len(mid_ratings)-1, 1)) ** 0.5

    gaps = []
    for title, info in scores.items():
        r_z = (info['rating'] - r_mean) / r_std if r_std > 0 else 0
        gap = info['score'] - r_z
        gaps.append((title, info, gap, r_z))

    gaps.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Gap = Script_z - Rating_z (양수 = 각본이 평점보다 좋다)")
    print(f"  평작 평균 평점: {r_mean:.1f} (sigma={r_std:.2f})")
    print(f"\n  {'#':>3s} {'Gap':>6s} {'Script':>7s} {'R_z':>6s} {'IMDB':>5s} {'Title':<40s}")
    print(f"  {'─'*70}")

    for i, (title, info, gap, r_z) in enumerate(gaps[:15]):
        print(f"  {i+1:3d} {gap:+6.2f} {info['score']:+7.2f} {r_z:+6.2f} {info['rating']:5.1f} {title[:40]:<40s}")

    # ── Step 6: 반대 — 각본이 과대평가된 평작 ──
    print(f"\n{sep}")
    print("  OVERRATED: 평점은 높은데 각본이 별로인 평작")
    print(sep)
    print(f"\n  {'#':>3s} {'Gap':>6s} {'Script':>7s} {'R_z':>6s} {'IMDB':>5s} {'Title':<40s}")
    print(f"  {'─'*70}")

    for i, (title, info, gap, r_z) in enumerate(gaps[-10:][::-1]):
        print(f"  {i+1:3d} {gap:+6.2f} {info['score']:+7.2f} {r_z:+6.2f} {info['rating']:5.1f} {title[:40]:<40s}")

    # ── Sanity check: 명작도 찍어보기 ──
    print(f"\n{sep}")
    print("  SANITY CHECK: 명작 중 각본 점수 TOP/BOTTOM 5")
    print(sep)

    good_scores = {}
    for title, v in good.items():
        total_z = 0
        for m in KEY_METRICS:
            s = stats[m]
            z = (v[m] - s['g_mean']) / s['g_std'] if s['g_std'] > 0 else 0
            total_z += s['direction'] * z
        good_scores[title] = {
            'rating': v['rating'],
            'score': total_z / len(KEY_METRICS),
            'n_scenes': v.get('n_scenes', 0),
        }

    g_ranked = sorted(good_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    print(f"\n  Best screenplays among masterpieces:")
    for i, (t, info) in enumerate(g_ranked[:5]):
        print(f"    {i+1}. {info['score']:+6.2f} ({info['rating']}) {t}")
    print(f"\n  Worst screenplays among masterpieces:")
    for i, (t, info) in enumerate(g_ranked[-5:][::-1]):
        print(f"    {i+1}. {info['score']:+6.2f} ({info['rating']}) {t}")

    print(f"\n  Done. Total scored: {len(scores)} mediocre movies.")


if __name__ == "__main__":
    main()
