"""
4-Quadrant Analysis: IMDB Rating vs Screenplay Structure Score

X축: Screenplay structure score (z-score relative to masterpiece mean)
Y축: IMDB rating

Q1 (top-right): Great script + High rating = True masterpiece
Q2 (top-left):  Weak script + High rating = Carried by direction/acting
Q3 (bottom-left): Weak script + Low rating = True garbage
Q4 (bottom-right): Great script + Low rating = Buried good script
"""

import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
CHECKPOINT = f'{BASE}/screenplay/plot_entropy_checkpoint.json'

KEY_METRICS = ['repeat_ratio', 'arc_shift', 'setup_front', 'bi_entropy']
MIN_SCENES = 15


def main():
    cp = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
    valid = {k: v for k, v in cp.items()
             if isinstance(v, dict) and v.get('labels') and v.get('rating')
             and v.get('n_scenes', 0) >= MIN_SCENES}

    good = {k: v for k, v in valid.items() if v['rating'] >= 8.0}
    bad  = {k: v for k, v in valid.items() if v['rating'] < 4.0}

    # Compute stats from good/bad
    stats = {}
    for m in KEY_METRICS:
        gv = [v[m] for v in good.values()]
        bv = [v[m] for v in bad.values()]
        g_mean = sum(gv) / len(gv)
        g_std = (sum((x - g_mean)**2 for x in gv) / max(len(gv)-1, 1)) ** 0.5
        direction = +1 if g_mean > sum(bv)/len(bv) else -1
        stats[m] = {'g_mean': g_mean, 'g_std': g_std, 'direction': direction}

    # Score ALL movies
    all_scores = []
    for title, v in valid.items():
        total_z = 0
        for m in KEY_METRICS:
            s = stats[m]
            z = (v[m] - s['g_mean']) / s['g_std'] if s['g_std'] > 0 else 0
            total_z += s['direction'] * z
        score = total_z / len(KEY_METRICS)
        all_scores.append({
            'title': title,
            'rating': v['rating'],
            'score': score,
        })

    # Quadrant boundaries
    r_mid = 6.5  # rating midpoint
    s_mid = 0.0  # score midpoint (= masterpiece average)

    # Standard coordinate: X=script score, Y=rating
    # Q1(+,+) Q2(-,+) Q3(-,-) Q4(+,-)
    q1 = [m for m in all_scores if m['score'] >= s_mid and m['rating'] >= r_mid]  # TRUE MASTERPIECE
    q2 = [m for m in all_scores if m['score'] < s_mid and m['rating'] >= r_mid]   # CARRIED
    q3 = [m for m in all_scores if m['score'] < s_mid and m['rating'] < r_mid]    # TRUE GARBAGE
    q4 = [m for m in all_scores if m['score'] >= s_mid and m['rating'] < r_mid]   # BURIED GEM

    sep = "=" * 75

    print(sep)
    print("  4-QUADRANT ANALYSIS: Screenplay Structure vs IMDB Rating")
    print(sep)
    print(f"\n  Total: {len(all_scores)} movies (min {MIN_SCENES} scenes)")
    print(f"  Boundaries: rating={r_mid}, script_score={s_mid}")
    print(f"\n  Q1 (명작, 각본도 좋음):  {len(q1)}편")
    print(f"  Q2 (명작, 각본은 약함):  {len(q2)}편")
    print(f"  Q3 (저평가, 각본도 약함): {len(q3)}편")
    print(f"  Q4 (저평가, 각본은 좋음): {len(q4)}편")

    # ── Q1: True Masterpieces ──
    print(f"\n{sep}")
    print("  Q1: TRUE MASTERPIECE — 각본도 좋고 평점도 높은 진짜 명작")
    print(f"  (script >= 0, rating >= {r_mid})")
    print(sep)
    q1.sort(key=lambda x: x['score'] + (x['rating'] - r_mid) * 0.3, reverse=True)
    print(f"\n  {'#':>3s} {'Script':>7s} {'IMDB':>5s} {'Title':<45s}")
    print(f"  {'─'*62}")
    for i, m in enumerate(q1[:15]):
        print(f"  {i+1:3d} {m['score']:+7.2f} {m['rating']:5.1f} {m['title'][:45]}")
    if len(q1) > 15:
        print(f"  ... +{len(q1)-15}편")

    # ── Q2: Carried by Direction ──
    print(f"\n{sep}")
    print("  Q2: CARRIED — 각본은 약한데 연출/연기/음악으로 성공")
    print(f"  (script < 0, rating >= {r_mid})")
    print(sep)
    q2.sort(key=lambda x: x['score'])  # most negative first = weakest script
    print(f"\n  {'#':>3s} {'Script':>7s} {'IMDB':>5s} {'Title':<45s}")
    print(f"  {'─'*62}")
    for i, m in enumerate(q2[:15]):
        print(f"  {i+1:3d} {m['score']:+7.2f} {m['rating']:5.1f} {m['title'][:45]}")
    if len(q2) > 15:
        print(f"  ... +{len(q2)-15}편")

    # ── Q3: True Garbage ──
    print(f"\n{sep}")
    print("  Q3: TRUE GARBAGE — 각본도 별로고 영화도 별로")
    print(f"  (script < 0, rating < {r_mid})")
    print(sep)
    q3.sort(key=lambda x: x['score'] + (x['rating'] - r_mid) * 0.3)  # worst first
    print(f"\n  {'#':>3s} {'Script':>7s} {'IMDB':>5s} {'Title':<45s}")
    print(f"  {'─'*62}")
    for i, m in enumerate(q3[:15]):
        print(f"  {i+1:3d} {m['score']:+7.2f} {m['rating']:5.1f} {m['title'][:45]}")
    if len(q3) > 15:
        print(f"  ... +{len(q3)-15}편")

    # ── Q4: Buried Gems ──
    print(f"\n{sep}")
    print("  Q4: BURIED GEM — 각본은 좋은데 다른 요소가 죽인 영화")
    print(f"  (script >= 0, rating < {r_mid})")
    print(sep)
    q4.sort(key=lambda x: x['score'], reverse=True)  # best script first
    print(f"\n  {'#':>3s} {'Script':>7s} {'IMDB':>5s} {'Title':<45s}")
    print(f"  {'─'*62}")
    for i, m in enumerate(q4[:15]):
        print(f"  {i+1:3d} {m['score']:+7.2f} {m['rating']:5.1f} {m['title'][:45]}")
    if len(q4) > 15:
        print(f"  ... +{len(q4)-15}편")

    # ── Distribution stats ──
    print(f"\n{sep}")
    print("  QUADRANT STATISTICS")
    print(sep)

    for label, q in [("Q1 True Masterpiece", q1), ("Q2 Carried", q2),
                      ("Q3 True Garbage", q3), ("Q4 Buried Gem", q4)]:
        if not q:
            print(f"\n  {label}: 0편")
            continue
        ratings = [m['rating'] for m in q]
        scripts = [m['score'] for m in q]
        r_avg = sum(ratings) / len(ratings)
        s_avg = sum(scripts) / len(scripts)
        print(f"\n  {label}: {len(q)}편")
        print(f"    Rating: avg={r_avg:.1f}, range={min(ratings):.1f}–{max(ratings):.1f}")
        print(f"    Script: avg={s_avg:+.2f}, range={min(scripts):+.2f}–{max(scripts):+.2f}")

    # ── Extreme corners ──
    print(f"\n{sep}")
    print("  EXTREME CORNERS — 각 분면의 극단")
    print(sep)

    all_scores.sort(key=lambda x: x['score'] + x['rating'] * 0.5, reverse=True)
    print(f"\n  Most Q1 (best script + best rating):")
    for m in all_scores[:3]:
        print(f"    {m['score']:+.2f} / {m['rating']} — {m['title']}")

    # Biggest gap: high rating, low script
    all_scores.sort(key=lambda x: x['rating'] * 0.5 - x['score'], reverse=True)
    print(f"\n  Most Q2 (highest rating, weakest script):")
    for m in all_scores[:3]:
        print(f"    {m['score']:+.2f} / {m['rating']} — {m['title']}")

    all_scores.sort(key=lambda x: x['score'] + x['rating'] * 0.5)
    print(f"\n  Most Q3 (worst script + worst rating):")
    for m in all_scores[:3]:
        print(f"    {m['score']:+.2f} / {m['rating']} — {m['title']}")

    # Biggest gap: high script, low rating
    all_scores.sort(key=lambda x: x['score'] - x['rating'] * 0.5, reverse=True)
    print(f"\n  Most Q4 (best script, lowest rating):")
    for m in all_scores[:3]:
        print(f"    {m['score']:+.2f} / {m['rating']} — {m['title']}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
