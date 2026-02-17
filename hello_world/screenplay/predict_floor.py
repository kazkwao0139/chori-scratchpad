from pathlib import Path
"""
Asymmetric hypothesis: bad screenplay → guaranteed bad movie.
The three axes don't predict success, they predict the FLOOR.
Test: movies far from optimal on all three axes — do they consistently rate low?
"""

import json
import sys
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


def main():
    print("=" * 70)
    print("  FLOOR HYPOTHESIS: bad screenplay → guaranteed bad movie?")
    print("=" * 70)

    with open(f'{BASE}/screenplay/mass_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    rated = data['rated']

    # Filter: need all three axes
    complete = []
    for d in rated:
        if (d.get('char_var') is not None
                and d.get('narr_std') is not None
                and d.get('dir_ratio') is not None
                and d.get('rating') is not None
                and d.get('votes', 0) >= 1000):
            complete.append(d)

    print(f"\n  Complete data: {len(complete)} movies")

    # Compute deviations for each axis
    # First find the "optimal" for char_var and narr_std too
    # For dir_ratio we already know: ~0.575
    # For char_var and narr_std, let's find the optimal (peak of top-rated movies)

    top = [d for d in complete if d['rating'] >= 7.5]
    mid = [d for d in complete if 6.0 <= d['rating'] < 7.5]
    bot = [d for d in complete if d['rating'] < 6.0]

    opt_cv = sum(d['char_var'] for d in top) / len(top)
    opt_ns = sum(d['narr_std'] for d in top) / len(top)
    opt_dr = 0.575

    print(f"\n  Optimal points (avg of rating >= 7.5):")
    print(f"    char_var:  {opt_cv:.4f}")
    print(f"    narr_std:  {opt_ns:.4f}")
    print(f"    dir_ratio: {opt_dr:.3f}")

    # Compute normalized deviation for each movie on each axis
    cv_std = (sum((d['char_var'] - opt_cv)**2 for d in complete) / len(complete)) ** 0.5
    ns_std = (sum((d['narr_std'] - opt_ns)**2 for d in complete) / len(complete)) ** 0.5
    dr_std = (sum((d['dir_ratio'] - opt_dr)**2 for d in complete) / len(complete)) ** 0.5

    for d in complete:
        d['dev_cv'] = abs(d['char_var'] - opt_cv) / cv_std
        d['dev_ns'] = abs(d['narr_std'] - opt_ns) / ns_std
        d['dev_dr'] = abs(d['dir_ratio'] - opt_dr) / dr_std
        d['total_dev'] = (d['dev_cv']**2 + d['dev_ns']**2 + d['dev_dr']**2) ** 0.5

    # Sort by total deviation
    complete.sort(key=lambda d: d['total_dev'])

    # ─── Test 1: Quintile analysis ───
    print(f"\n{'=' * 70}")
    print("  QUINTILE ANALYSIS (by total 3D deviation)")
    print(f"{'=' * 70}")

    n = len(complete)
    qs = 5
    qsize = n // qs

    print(f"\n  {'Quintile':>10s} {'n':>4s} {'avg_r':>7s} {'min_r':>7s} {'max_r':>7s} {'%<5.5':>7s} {'%<6.0':>7s} {'%>=7.5':>7s} {'avg_dev':>8s}")
    print(f"  {'-'*70}")

    for q in range(qs):
        start = q * qsize
        end = start + qsize if q < qs - 1 else n
        items = complete[start:end]
        ar = sum(d['rating'] for d in items) / len(items)
        mn = min(d['rating'] for d in items)
        mx = max(d['rating'] for d in items)
        pct_bad = sum(1 for d in items if d['rating'] < 5.5) / len(items)
        pct_bad2 = sum(1 for d in items if d['rating'] < 6.0) / len(items)
        pct_good = sum(1 for d in items if d['rating'] >= 7.5) / len(items)
        ad = sum(d['total_dev'] for d in items) / len(items)
        label = f"Q{q+1} ({'closest' if q == 0 else 'farthest' if q == qs-1 else ''})"
        print(f"  {label:>10s} {len(items):4d} {ar:7.2f} {mn:7.1f} {mx:7.1f} {pct_bad:7.1%} {pct_bad2:7.1%} {pct_good:7.1%} {ad:8.2f}")

    # ─── Test 2: Extreme outliers ───
    print(f"\n{'=' * 70}")
    print("  EXTREME OUTLIERS (total_dev > 3.0)")
    print(f"{'=' * 70}")

    extremes = [d for d in complete if d['total_dev'] > 3.0]
    print(f"\n  {len(extremes)} movies with total_dev > 3.0:")
    extremes.sort(key=lambda d: -d['total_dev'])
    for d in extremes:
        print(f"    dev={d['total_dev']:.2f}  r={d['rating']:.1f}  cv={d['char_var']:.4f}  ns={d['narr_std']:.4f}  dr={d['dir_ratio']:.1%}")

    print(f"\n  avg rating of extremes: {sum(d['rating'] for d in extremes)/len(extremes):.2f}" if extremes else "  None")

    # ─── Test 3: Conditional floor ───
    print(f"\n{'=' * 70}")
    print("  CONDITIONAL FLOOR TEST")
    print(f"  'If ALL three axes are within X std of optimal, what's the min rating?'")
    print(f"{'=' * 70}")

    for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        inside = [d for d in complete
                  if d['dev_cv'] <= threshold
                  and d['dev_ns'] <= threshold
                  and d['dev_dr'] <= threshold]
        if not inside:
            continue
        ratings = [d['rating'] for d in inside]
        ar = sum(ratings) / len(ratings)
        mn = min(ratings)
        pct_bad = sum(1 for r in ratings if r < 5.5) / len(ratings)
        pct_good = sum(1 for r in ratings if r >= 7.5) / len(ratings)
        print(f"  ≤{threshold:.1f}σ: n={len(inside):4d}  avg={ar:.2f}  min={mn:.1f}  %bad(<5.5)={pct_bad:5.1%}  %good(≥7.5)={pct_good:5.1%}")

    # ─── Test 4: OUTSIDE threshold ───
    print(f"\n  Flipped: 'If ANY axis is beyond X std, what happens?'")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        outside = [d for d in complete
                   if d['dev_cv'] > threshold
                   or d['dev_ns'] > threshold
                   or d['dev_dr'] > threshold]
        if not outside:
            continue
        ratings = [d['rating'] for d in outside]
        ar = sum(ratings) / len(ratings)
        mn = min(ratings)
        pct_bad = sum(1 for r in ratings if r < 5.5) / len(ratings)
        pct_good = sum(1 for r in ratings if r >= 7.5) / len(ratings)
        print(f"  >{threshold:.1f}σ: n={len(outside):4d}  avg={ar:.2f}  min={mn:.1f}  %bad(<5.5)={pct_bad:5.1%}  %good(≥7.5)={pct_good:5.1%}")

    # ─── Test 5: Floor envelope ───
    print(f"\n{'=' * 70}")
    print("  FLOOR ENVELOPE: min rating as function of deviation")
    print(f"{'=' * 70}")

    # Bin by total_dev and show min/5th percentile rating
    max_dev = max(d['total_dev'] for d in complete)
    n_bins = 10
    bw = max_dev / n_bins

    print(f"\n  {'dev_range':>15s} {'n':>5s} {'min_r':>7s} {'p5_r':>7s} {'p10_r':>7s} {'avg_r':>7s} {'%<6':>6s}")
    print(f"  {'-'*55}")

    for b in range(n_bins):
        lo = b * bw
        hi = lo + bw
        items = [d for d in complete if lo <= d['total_dev'] < hi]
        if len(items) < 3:
            continue
        ratings = sorted([d['rating'] for d in items])
        mn = ratings[0]
        p5 = ratings[max(0, int(len(ratings) * 0.05))]
        p10 = ratings[max(0, int(len(ratings) * 0.10))]
        ar = sum(ratings) / len(ratings)
        pct_bad = sum(1 for r in ratings if r < 6.0) / len(ratings)
        print(f"  {lo:6.2f}-{hi:5.2f} {len(items):5d} {mn:7.1f} {p5:7.1f} {p10:7.1f} {ar:7.2f} {pct_bad:6.1%}")

    # ─── Test 6: Success rate by zone ───
    print(f"\n{'=' * 70}")
    print("  SUCCESS RATE BY ZONE")
    print(f"{'=' * 70}")

    zones = [
        ("Sweet spot (all ≤1.0σ)", lambda d: d['dev_cv'] <= 1.0 and d['dev_ns'] <= 1.0 and d['dev_dr'] <= 1.0),
        ("Danger zone (any >2.0σ)", lambda d: d['dev_cv'] > 2.0 or d['dev_ns'] > 2.0 or d['dev_dr'] > 2.0),
        ("Disaster zone (any >3.0σ)", lambda d: d['dev_cv'] > 3.0 or d['dev_ns'] > 3.0 or d['dev_dr'] > 3.0),
        ("Total dev < 1.0", lambda d: d['total_dev'] < 1.0),
        ("Total dev 1.0-2.0", lambda d: 1.0 <= d['total_dev'] < 2.0),
        ("Total dev 2.0-3.0", lambda d: 2.0 <= d['total_dev'] < 3.0),
        ("Total dev > 3.0", lambda d: d['total_dev'] > 3.0),
    ]

    print(f"\n  {'Zone':>35s} {'n':>5s} {'avg':>6s} {'min':>5s} {'%<5.5':>6s} {'%<6':>5s} {'%≥7':>5s} {'%≥8':>5s}")
    print(f"  {'-'*75}")
    for name, fn in zones:
        items = [d for d in complete if fn(d)]
        if not items:
            continue
        ratings = [d['rating'] for d in items]
        ar = sum(ratings) / len(ratings)
        mn = min(ratings)
        p1 = sum(1 for r in ratings if r < 5.5) / len(ratings)
        p2 = sum(1 for r in ratings if r < 6.0) / len(ratings)
        p3 = sum(1 for r in ratings if r >= 7.0) / len(ratings)
        p4 = sum(1 for r in ratings if r >= 8.0) / len(ratings)
        print(f"  {name:>35s} {len(items):5d} {ar:6.2f} {mn:5.1f} {p1:6.1%} {p2:5.1%} {p3:5.1%} {p4:5.1%}")

    # Save
    output = {
        'optimal': {'char_var': round(opt_cv, 4), 'narr_std': round(opt_ns, 4), 'dir_ratio': opt_dr},
        'n': len(complete),
    }
    with open(f'{BASE}/screenplay/predict_floor_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: predict_floor_results.json")


if __name__ == "__main__":
    main()
