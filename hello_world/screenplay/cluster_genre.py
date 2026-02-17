from pathlib import Path
"""
Genre-based local minima: split by genre, then check within-genre
center-deviation vs rating correlation.
"""

import json
import sys
import math
import gzip
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


def pearson(a, b):
    n = len(a)
    if n < 10:
        return 0, 0, n
    ma, mb = sum(a)/n, sum(b)/n
    sa = (sum((v-ma)**2 for v in a)/n)**0.5
    sb = (sum((v-mb)**2 for v in b)/n)**0.5
    if sa == 0 or sb == 0:
        return 0, 0, n
    r = sum((ai-ma)*(bi-mb) for ai, bi in zip(a, b)) / (n*sa*sb)
    t = r * math.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 999
    return r, t, n


def main():
    print("=" * 70)
    print("  GENRE-BASED LOCAL MINIMA")
    print("=" * 70)

    # Load LLM checkpoint
    cp = json.load(open(f'{BASE}/screenplay/llm_mass_checkpoint.json', 'r', encoding='utf-8'))

    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    complete = {}
    for title, info in cp.items():
        if all(info.get(f) is not None for f in feat_names) and info.get('rating') is not None:
            complete[title] = {**info, 'title': title}

    print(f"\n  Complete movies: {len(complete)}")

    # Load genre from IMDB basics
    # mass_v2.py already matched titles to tconst. Let's load mass_v2_results.json for title list,
    # then match genres from _imdb_basics.tsv.gz

    # First load mass_v2 results which has the title→tconst mapping indirectly
    # Actually, let's just match by title from _imdb_basics.tsv.gz directly
    print("  Loading IMDB basics for genre data...")

    # Build title → genres from IMDB basics (movies only)
    title_genres = {}
    with gzip.open(f'{BASE}/screenplay/_imdb_basics.tsv.gz', 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            ttype = parts[1]
            if ttype != 'movie':
                continue
            primary = parts[2]
            original = parts[3]
            genres = parts[8] if parts[8] != '\\N' else ''
            # Store both primary and original title
            if genres:
                title_genres[primary] = genres.split(',')
                if original != primary:
                    title_genres[original] = genres.split(',')

    print(f"  IMDB movies with genre: {len(title_genres)}")

    # Match genres to our movies
    matched = 0
    genre_counts = {}
    for title, info in complete.items():
        # Try exact match
        genres = title_genres.get(title)
        if not genres:
            # Try removing ", The" etc.
            if title.endswith(', The'):
                genres = title_genres.get('The ' + title[:-5])
            elif title.endswith(', A'):
                genres = title_genres.get('A ' + title[:-3])
        if genres:
            info['genres'] = genres
            matched += 1
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        else:
            info['genres'] = []

    print(f"  Genre matched: {matched}/{len(complete)}")

    # Show genre distribution
    print(f"\n  Genre distribution:")
    for g, c in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"    {g:>15s}: {c:4d}")

    # Global standardization
    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in complete.values()]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    # ── Per-genre analysis ──
    print(f"\n{'=' * 70}")
    print("  PER-GENRE: CENTER DEVIATION vs RATING")
    print(f"{'=' * 70}")

    # For each genre with enough movies
    min_n = 30
    genre_results = []

    for genre in sorted(genre_counts.keys(), key=lambda g: -genre_counts[g]):
        members = [m for m in complete.values() if genre in m.get('genres', [])]
        if len(members) < min_n:
            continue

        n = len(members)
        avg_r = sum(m['rating'] for m in members) / n

        # Genre center in raw feature space
        raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}

        # Distance to center for each movie (standardized)
        for m in members:
            m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
        members.sort(key=lambda m: m['_dist'])

        # Close half vs far half
        half = n // 2
        close = members[:half]
        far = members[half:]
        close_avg = sum(m['rating'] for m in close) / len(close)
        far_avg = sum(m['rating'] for m in far) / len(far)
        delta = close_avg - far_avg

        # Correlation
        devs = [m['_dist'] for m in members]
        rats = [m['rating'] for m in members]
        r_val, t_val, _ = pearson(devs, rats)
        sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""

        genre_results.append({
            'genre': genre, 'n': n, 'avg_r': avg_r,
            'close_avg': close_avg, 'far_avg': far_avg, 'delta': delta,
            'r': r_val, 't': t_val, 'sig': sig,
            'center': raw_center
        })

    # Print summary
    print(f"\n  {'Genre':>15s} {'n':>5s} {'avg_r':>6s} {'close':>6s} {'far':>6s} {'Δ':>7s} {'r(dev)':>8s} {'t':>7s} {'sig':>4s}")
    print(f"  {'-'*70}")

    for r in genre_results:
        print(f"  {r['genre']:>15s} {r['n']:5d} {r['avg_r']:6.2f} "
              f"{r['close_avg']:6.2f} {r['far_avg']:6.2f} {r['delta']:+7.3f} "
              f"{r['r']:+8.4f} {r['t']:+7.2f} {r['sig']:>4s}")

    # ── Genre center profiles ──
    print(f"\n{'=' * 70}")
    print("  GENRE CENTER PROFILES")
    print(f"{'=' * 70}")
    print(f"\n  {'Genre':>15s} {'n':>5s} {'narr':>6s} {'cvar':>6s} {'gap':>7s} {'dppl':>6s} {'dr':>6s}")
    print(f"  {'-'*55}")
    for r in genre_results:
        c = r['center']
        print(f"  {r['genre']:>15s} {r['n']:5d} "
              f"{c['llm_narr_std']:6.1f} {c['llm_char_var']:6.1f} "
              f"{c['llm_ppl_gap']:+7.1f} {c['llm_dial_ppl']:6.1f} {c['dir_ratio']:6.1%}")

    # ── Significant findings detail ──
    sig_genres = [r for r in genre_results if r['sig']]
    if sig_genres:
        print(f"\n{'=' * 70}")
        print("  SIGNIFICANT GENRES — DETAIL")
        print(f"{'=' * 70}")
        for r in sig_genres:
            genre = r['genre']
            members = sorted(
                [m for m in complete.values() if genre in m.get('genres', [])],
                key=lambda m: m['_dist']
            )
            n = len(members)
            half = n // 2

            print(f"\n  {genre} (n={n}, r={r['r']:+.4f}, t={r['t']:+.2f} {r['sig']})")
            print(f"  Close half avg: {r['close_avg']:.2f}, Far half avg: {r['far_avg']:.2f}, Δ={r['delta']:+.3f}")

            # Center movies
            print(f"  Center 5:")
            for m in members[:5]:
                print(f"    d={m['_dist']:.2f} r={m['rating']:.1f} {m['title']}")
            print(f"  Edge 5:")
            for m in members[-5:]:
                print(f"    d={m['_dist']:.2f} r={m['rating']:.1f} {m['title']}")

            # Percentages
            close = members[:half]
            far = members[half:]
            close_bad = sum(1 for m in close if m['rating'] < 6.0) / len(close)
            far_bad = sum(1 for m in far if m['rating'] < 6.0) / len(far)
            close_good = sum(1 for m in close if m['rating'] >= 8.0) / len(close)
            far_good = sum(1 for m in far if m['rating'] >= 8.0) / len(far)
            print(f"  Close: %<6.0={close_bad:.1%}  %>=8.0={close_good:.1%}")
            print(f"  Far:   %<6.0={far_bad:.1%}  %>=8.0={far_good:.1%}")
    else:
        print(f"\n  No genre reached significance (p<0.05).")

    # ── Multi-genre intersection ──
    # Some movies belong to multiple genres. Try top genre PAIRS
    print(f"\n{'=' * 70}")
    print("  TOP GENRE PAIRS (n>=30)")
    print(f"{'=' * 70}")

    pair_counts = {}
    for m in complete.values():
        gs = m.get('genres', [])
        for i in range(len(gs)):
            for j in range(i+1, len(gs)):
                pair = tuple(sorted([gs[i], gs[j]]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    print(f"\n  {'Pair':>25s} {'n':>5s} {'avg_r':>6s} {'Δ':>7s} {'r':>8s} {'t':>7s} {'sig':>4s}")
    print(f"  {'-'*65}")

    for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        if cnt < 30:
            continue
        members = [m for m in complete.values()
                   if pair[0] in m.get('genres', []) and pair[1] in m.get('genres', [])]
        n = len(members)
        avg_r = sum(m['rating'] for m in members) / n
        raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}
        for m in members:
            m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
        members.sort(key=lambda m: m['_dist'])
        half = n // 2
        close = members[:half]
        far = members[half:]
        close_avg = sum(m['rating'] for m in close) / len(close)
        far_avg = sum(m['rating'] for m in far) / len(far)
        delta = close_avg - far_avg
        devs = [m['_dist'] for m in members]
        rats = [m['rating'] for m in members]
        r_val, t_val, _ = pearson(devs, rats)
        sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""
        label = f"{pair[0]}/{pair[1]}"
        print(f"  {label:>25s} {n:5d} {avg_r:6.2f} {delta:+7.3f} {r_val:+8.4f} {t_val:+7.2f} {sig:>4s}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
