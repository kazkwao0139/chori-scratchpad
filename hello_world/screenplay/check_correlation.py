"""
Live correlation check: read mass_checkpoint and compute
direction ratio vs IMDB rating correlation as data accumulates.
"""

import re
import json
import sys
import math
import zlib
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"


def zlib_entropy(text):
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def split_dialogue_direction(full_text):
    lines = full_text.split('\n')
    dialogue_lines = []
    direction_lines = []
    current_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_char = None
            continue

        clean = re.sub(r'\(.*?\)', '', stripped).strip()

        if (clean.isupper() and 2 <= len(clean) <= 30
                and not clean.startswith('INT') and not clean.startswith('EXT')
                and not clean.startswith('CUT') and not clean.startswith('FADE')
                and not clean.startswith('CLOSE') and not clean.startswith('ANGLE')
                and not clean.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", clean)):
            current_char = clean
            continue

        if current_char and len(stripped) > 1 and not stripped.isupper():
            dialogue_lines.append(stripped)
        else:
            direction_lines.append(stripped)
            current_char = None

    return ' '.join(dialogue_lines), ' '.join(direction_lines)


def pearson_r(xs, ys):
    n = len(xs)
    if n < 3:
        return 0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = (sum((x - mx)**2 for x in xs) / n) ** 0.5
    sy = (sum((y - my)**2 for y in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return 0, 0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    r = cov / (sx * sy)
    # t-test for significance
    if abs(r) >= 1:
        return r, 0
    t = r * math.sqrt((n - 2) / (1 - r**2))
    return r, t


def main():
    with open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8') as f:
        cp = json.load(f)

    scripts = cp.get('scripts', {})
    ratings = cp.get('ratings', {})
    results_raw = cp.get('results', {})

    # Count progress
    total = len(cp.get('script_list', []))
    processed = len(results_raw)
    valid_scripts = {k for k, v in scripts.items() if v is not None}
    valid_results = {k for k, v in results_raw.items() if v is not None}

    print(f"Progress: {processed}/{total} processed")
    print(f"  Valid scripts: {len(valid_scripts)}")
    print(f"  Valid results: {len(valid_results)}")

    # Compute direction ratio for all scripts with ratings
    data_points = []
    for title, text in scripts.items():
        if text is None or len(text) < 5000:
            continue
        rating_info = ratings.get(title, {})
        rating = rating_info.get('rating')
        if rating is None:
            continue

        dialogue, direction = split_dialogue_direction(text)
        if len(dialogue) < 500 or len(direction) < 500:
            continue

        total_len = len(dialogue) + len(direction)
        dir_ratio = len(direction) / total_len

        dial_zlib = zlib_entropy(dialogue)
        dir_zlib = zlib_entropy(direction)
        zlib_gap = dir_zlib - dial_zlib

        data_points.append({
            'title': title,
            'rating': rating,
            'dir_ratio': dir_ratio,
            'zlib_gap': zlib_gap,
            'dial_zlib': dial_zlib,
            'dir_zlib': dir_zlib,
        })

    print(f"  Data points with rating + direction ratio: {len(data_points)}")

    if len(data_points) < 5:
        print("  Not enough data yet.")
        return

    # Correlation: IMDB rating vs direction ratio
    ratings_list = [d['rating'] for d in data_points]
    dir_ratios = [d['dir_ratio'] for d in data_points]
    gaps = [d['zlib_gap'] for d in data_points]

    r_dir, t_dir = pearson_r(ratings_list, dir_ratios)
    r_gap, t_gap = pearson_r(ratings_list, gaps)

    print(f"\n{'='*70}")
    print(f"  CORRELATION WITH IMDB RATING (n={len(data_points)})")
    print(f"{'='*70}")
    print(f"  Rating vs dir_ratio:  r = {r_dir:+.4f}  (t = {t_dir:.2f})")
    print(f"  Rating vs zlib_gap:   r = {r_gap:+.4f}  (t = {t_gap:.2f})")

    # Tier breakdown
    tiers = [
        ('< 5.0', [d for d in data_points if d['rating'] < 5.0]),
        ('5.0-6.0', [d for d in data_points if 5.0 <= d['rating'] < 6.0]),
        ('6.0-7.0', [d for d in data_points if 6.0 <= d['rating'] < 7.0]),
        ('7.0-8.0', [d for d in data_points if 7.0 <= d['rating'] < 8.0]),
        ('>= 8.0', [d for d in data_points if d['rating'] >= 8.0]),
    ]

    print(f"\n  {'Tier':>10s} {'n':>4s} {'avg_rating':>10s} {'dir_ratio':>10s} {'zlib_gap':>10s}")
    print(f"  {'-'*50}")
    for name, items in tiers:
        if not items:
            continue
        ar = sum(d['rating'] for d in items) / len(items)
        adr = sum(d['dir_ratio'] for d in items) / len(items)
        ag = sum(d['zlib_gap'] for d in items) / len(items)
        print(f"  {name:>10s} {len(items):4d} {ar:10.1f} {adr:10.1%} {ag:+10.4f}")

    # Show individual bad movies
    bad = [d for d in data_points if d['rating'] < 6.0]
    if bad:
        print(f"\n  Bad movies (rating < 6.0):")
        for d in sorted(bad, key=lambda x: x['rating']):
            print(f"    r={d['rating']:.1f} dir={d['dir_ratio']:.1%} gap={d['zlib_gap']:+.4f} {d['title']}")

    # Show top rated
    great = [d for d in data_points if d['rating'] >= 8.0]
    if great:
        print(f"\n  Great movies (rating >= 8.0):")
        for d in sorted(great, key=lambda x: -x['rating']):
            print(f"    r={d['rating']:.1f} dir={d['dir_ratio']:.1%} gap={d['zlib_gap']:+.4f} {d['title']}")


if __name__ == "__main__":
    main()
