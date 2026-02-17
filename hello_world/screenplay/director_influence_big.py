"""
감독/배우 영향력 정량화 — BIG DATA 버전 (1,116편).

mass_v2 메트릭 (dir_ratio, dial_zlib, dir_zlib, char_var, narr_std) 사용.
순수 텍스트 기반이므로 LLM 불필요. 전체 IMSDB 각본에 적용 가능.
"""

import json
import sys
import gzip
import math
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
MASS_CHECKPOINT = f'{BASE}/screenplay/mass_v2_checkpoint.json'
IMDB_BASICS = f'{BASE}/_copyrighted/screenplay/_imdb_basics.tsv.gz'
IMDB_RATINGS = f'{BASE}/_copyrighted/screenplay/_imdb_ratings.tsv.gz'
IMDB_DIR = f'{BASE}/_copyrighted/screenplay'

# mass_v2 metrics
METRICS = ['dir_ratio', 'dial_zlib', 'dir_zlib', 'char_var', 'narr_std']


def normalize(t):
    t = t.lower().strip()
    for s in [', the', ', a', ', an']:
        if t.endswith(s):
            t = s.strip(', ') + ' ' + t[:-len(s)]
    return t


def load_movies():
    with open(MASS_CHECKPOINT, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    done = raw['done']

    movies = {}
    for title, v in done.items():
        if not isinstance(v, dict):
            continue
        if v.get('rating') is None:
            continue
        if not all(m in v for m in METRICS):
            continue
        movies[title] = v

    return movies


def match_titles(movies):
    """Match all movies to IMDB tconst."""
    print("  Loading IMDB ratings for vote counts...")
    ratings = {}
    with gzip.open(IMDB_RATINGS, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 3:
                try:
                    ratings[p[0]] = int(p[2])
                except ValueError:
                    pass

    title_lookup = {}
    for t in movies:
        title_lookup[normalize(t)] = t

    print("  Scanning IMDB basics for title matches...")
    candidates = defaultdict(list)
    with gzip.open(IMDB_BASICS, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 9 or p[1] != 'movie':
                continue
            for tf in [p[2], p[3]]:
                n = normalize(tf)
                if n in title_lookup:
                    orig = title_lookup[n]
                    votes = ratings.get(p[0], 0)
                    candidates[orig].append((p[0], votes))

    matched = 0
    for title, cands in candidates.items():
        if title in movies:
            best = max(cands, key=lambda x: x[1])
            movies[title]['imdb_id'] = best[0]
            matched += 1

    print(f"  Matched {matched} / {len(movies)} movies to IMDB IDs")
    return movies


def load_crew(tconst_set):
    path = f"{IMDB_DIR}/_imdb_crew.tsv.gz"
    crew = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 2:
                continue
            if p[0] in tconst_set:
                directors = p[1] if p[1] != '\\N' else ''
                crew[p[0]] = directors.split(',') if directors else []
    return crew


def load_principals(tconst_set):
    path = f"{IMDB_DIR}/_imdb_principals.tsv.gz"
    principals = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 4:
                continue
            if p[0] in tconst_set and p[3] in ('actor', 'actress'):
                try:
                    principals[p[0]].append((int(p[1]), p[2]))
                except ValueError:
                    pass
    for tc in principals:
        principals[tc].sort()
        principals[tc] = [nc for _, nc in principals[tc][:2]]
    return dict(principals)


def load_names(nconst_set):
    path = f"{IMDB_DIR}/_imdb_names.tsv.gz"
    names = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 2:
                continue
            if p[0] in nconst_set:
                names[p[0]] = p[1]
    return names


def compute_script_score(movies):
    """Multi-metric z-score: direction-detect from top vs bottom rated movies."""
    sorted_by_rating = sorted(movies.items(), key=lambda x: x[1]['rating'], reverse=True)
    n = len(sorted_by_rating)
    top = dict(sorted_by_rating[:int(n * 0.15)])  # top 15%
    bot = dict(sorted_by_rating[-int(n * 0.15):])  # bottom 15%

    stats = {}
    for m in METRICS:
        tv = [v[m] for v in top.values() if v[m] is not None]
        bv = [v[m] for v in bot.values() if v[m] is not None]
        if not tv or not bv:
            continue
        t_mean = sum(tv) / len(tv)
        b_mean = sum(bv) / len(bv)
        all_vals = [v[m] for v in movies.values() if v[m] is not None]
        a_mean = sum(all_vals) / len(all_vals)
        a_std = (sum((x - a_mean)**2 for x in all_vals) / max(len(all_vals)-1, 1)) ** 0.5
        direction = +1 if t_mean > b_mean else -1
        stats[m] = {
            'mean': a_mean, 'std': a_std,
            'direction': direction,
            'top_mean': t_mean, 'bot_mean': b_mean,
        }

    print(f"\n  Direction auto-detection (top 15% vs bottom 15%):")
    print(f"  {'Metric':>12s} {'Top15%':>8s} {'Bot15%':>8s} {'Direction':>14s}")
    for m in METRICS:
        if m in stats:
            s = stats[m]
            arrow = "↑=better" if s['direction'] > 0 else "↓=better"
            print(f"  {m:>12s} {s['top_mean']:8.4f} {s['bot_mean']:8.4f}   {arrow}")

    # Z-score each movie
    for title, v in movies.items():
        total_z = 0
        count = 0
        for m in METRICS:
            if m not in stats or v.get(m) is None:
                continue
            s = stats[m]
            if s['std'] == 0:
                continue
            z = (v[m] - s['mean']) / s['std']
            total_z += s['direction'] * z
            count += 1
        v['script_z'] = total_z / count if count > 0 else 0

    return stats


def compute_residuals(movies):
    rated = [(v['rating'], v['script_z']) for v in movies.values()]
    n = len(rated)
    r_mean = sum(r for r, _ in rated) / n
    z_mean = sum(z for _, z in rated) / n
    cov = sum((r - r_mean) * (z - z_mean) for r, z in rated) / n
    var_z = sum((z - z_mean)**2 for _, z in rated) / n
    slope = cov / var_z if var_z > 0 else 0

    for title, v in movies.items():
        expected = r_mean + slope * v['script_z']
        v['expected'] = expected
        v['residual'] = v['rating'] - expected

    ss_res = sum(v['residual']**2 for v in movies.values())
    ss_tot = sum((v['rating'] - r_mean)**2 for v in movies.values())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Linear model: E[rating] = {r_mean:.2f} + {slope:.3f} * script_z")
    print(f"  R² = {r2:.4f} (script explains {r2*100:.1f}% of variance)")
    return r_mean, slope, r2


def analyze_group(movies, group_map, names, group_label, min_films=3):
    """Analyze a group (directors or actors)."""
    group_stats = []
    for nc, film_list in group_map.items():
        if len(film_list) < min_films:
            continue
        name = names.get(nc, nc)
        residuals = [v['residual'] for _, v in film_list]
        ratings = [v['rating'] for _, v in film_list]
        scripts = [v['script_z'] for _, v in film_list]
        avg_res = sum(residuals) / len(residuals)
        avg_rating = sum(ratings) / len(ratings)
        avg_script = sum(scripts) / len(scripts)
        if len(residuals) > 1:
            res_std = (sum((r - avg_res)**2 for r in residuals) / (len(residuals)-1)) ** 0.5
        else:
            res_std = 0
        films = [(t, v['rating'], v['script_z'], v['residual']) for t, v in film_list]
        group_stats.append({
            'name': name, 'nconst': nc, 'n_films': len(film_list),
            'avg_residual': avg_res, 'avg_rating': avg_rating,
            'avg_script_z': avg_script, 'res_std': res_std,
            'films': films,
        })

    group_stats.sort(key=lambda x: x['avg_residual'], reverse=True)
    return group_stats


def print_ranking(stats, label, top_n=15, detail_n=5):
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {label} INFLUENCE RANKING (min 3 films)")
    print(f"  양수 = 각본 이상으로 끌어올림 | 음수 = 각본보다 깎아먹음")
    print(sep)
    print(f"\n  {'#':>3s} {'Resid':>7s} {'IMDB':>6s} {'Scrpt':>6s} {'N':>3s} {'Std':>5s}  {'Name':<30s}")
    print(f"  {'─'*68}")

    for i, d in enumerate(stats[:top_n]):
        print(f"  {i+1:3d} {d['avg_residual']:+7.3f} {d['avg_rating']:6.1f} {d['avg_script_z']:+6.2f} {d['n_films']:3d} {d['res_std']:5.2f}  {d['name']:<30s}")

    if len(stats) > top_n:
        print(f"  {'···':>3s}")
        for i, d in enumerate(stats[-top_n:]):
            rank = len(stats) - top_n + i + 1
            print(f"  {rank:3d} {d['avg_residual']:+7.3f} {d['avg_rating']:6.1f} {d['avg_script_z']:+6.2f} {d['n_films']:3d} {d['res_std']:5.2f}  {d['name']:<30s}")

    # Detail for top and bottom
    print(f"\n  TOP {detail_n}:")
    for d in stats[:detail_n]:
        print(f"\n  {d['name']} ({d['n_films']}편, avg residual={d['avg_residual']:+.3f})")
        for t, r, sz, res in sorted(d['films'], key=lambda x: -x[3]):
            print(f"    {t[:45]:<45s} IMDB={r:.1f} Script={sz:+.2f} Res={res:+.3f}")

    print(f"\n  BOTTOM {detail_n}:")
    for d in stats[-detail_n:]:
        print(f"\n  {d['name']} ({d['n_films']}편, avg residual={d['avg_residual']:+.3f})")
        for t, r, sz, res in sorted(d['films'], key=lambda x: -x[3]):
            print(f"    {t[:45]:<45s} IMDB={r:.1f} Script={sz:+.2f} Res={res:+.3f}")


def main():
    sep = "=" * 78

    print(sep)
    print("  DIRECTOR & ACTOR INFLUENCE — BIG DATA (1,116 movies)")
    print(sep)

    # Step 1: Load
    print(f"\n{'─'*78}\n  Step 1: Load movies\n{'─'*78}")
    movies = load_movies()
    print(f"  Loaded {len(movies)} movies with ratings + metrics")

    # Step 2: Match to IMDB
    print(f"\n{'─'*78}\n  Step 2: Match to IMDB IDs\n{'─'*78}")
    movies = match_titles(movies)
    have_id = {k: v for k, v in movies.items() if 'imdb_id' in v}

    # Step 3: Load crew/principals/names
    print(f"\n{'─'*78}\n  Step 3: Load IMDB crew & cast data\n{'─'*78}")
    tconst_set = {v['imdb_id'] for v in have_id.values()}
    print(f"  Looking up {len(tconst_set)} unique IMDB IDs...")

    crew = load_crew(tconst_set)
    print(f"  Directors found for {len(crew)} movies")

    principals = load_principals(tconst_set)
    print(f"  Lead actors found for {len(principals)} movies")

    all_nconsts = set()
    for dirs in crew.values():
        all_nconsts.update(dirs)
    for actors in principals.values():
        all_nconsts.update(actors)

    names = load_names(all_nconsts)
    print(f"  Resolved {len(names)} / {len(all_nconsts)} names")

    # Step 4: Script z-scores & residuals
    print(f"\n{'─'*78}\n  Step 4: Compute script z-scores & residuals\n{'─'*78}")
    compute_script_score(movies)
    r_mean, slope, r2 = compute_residuals(movies)

    # Step 5: Director grouping
    print(f"\n{'─'*78}\n  Step 5: Director analysis\n{'─'*78}")
    director_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if not tc or tc not in crew:
            continue
        for nc in crew[tc]:
            director_map[nc].append((title, v))

    d2 = {nc: ms for nc, ms in director_map.items() if len(ms) >= 2}
    d3 = {nc: ms for nc, ms in director_map.items() if len(ms) >= 3}
    print(f"  Directors with 2+ films: {len(d2)} (covering {sum(len(ms) for ms in d2.values())} movies)")
    print(f"  Directors with 3+ films: {len(d3)} (covering {sum(len(ms) for ms in d3.values())} movies)")

    director_stats = analyze_group(movies, director_map, names, "DIRECTOR", min_films=3)
    print_ranking(director_stats, "DIRECTOR", top_n=15, detail_n=5)

    # Step 6: Actor grouping
    print(f"\n{'─'*78}\n  Step 6: Lead actor analysis\n{'─'*78}")
    actor_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if not tc or tc not in principals:
            continue
        for nc in principals[tc]:
            actor_map[nc].append((title, v))

    a2 = {nc: ms for nc, ms in actor_map.items() if len(ms) >= 2}
    a3 = {nc: ms for nc, ms in actor_map.items() if len(ms) >= 3}
    print(f"  Lead actors with 2+ films: {len(a2)} (covering {sum(len(ms) for ms in a2.values())} movies)")
    print(f"  Lead actors with 3+ films: {len(a3)} (covering {sum(len(ms) for ms in a3.values())} movies)")

    actor_stats = analyze_group(movies, actor_map, names, "ACTOR", min_films=3)
    print_ranking(actor_stats, "LEAD ACTOR", top_n=15, detail_n=5)

    # Step 7: Summary
    print(f"\n{sep}")
    print("  SUMMARY STATISTICS")
    print(sep)
    print(f"\n  Dataset: {len(movies)} movies")
    print(f"  Script model R² = {r2:.4f}")

    if director_stats:
        d_res = [d['avg_residual'] for d in director_stats]
        d_mean = sum(d_res) / len(d_res)
        d_std = (sum((r - d_mean)**2 for r in d_res) / max(len(d_res)-1, 1)) ** 0.5
        print(f"\n  Directors (3+ films): {len(director_stats)}")
        print(f"    Residual range: [{min(d_res):+.3f}, {max(d_res):+.3f}]")
        print(f"    Residual std: {d_std:.3f}")
        print(f"    → 감독 영향력 ≈ {d_std:.2f} IMDB points (1σ)")

    if actor_stats:
        a_res = [a['avg_residual'] for a in actor_stats]
        a_mean = sum(a_res) / len(a_res)
        a_std = (sum((r - a_mean)**2 for r in a_res) / max(len(a_res)-1, 1)) ** 0.5
        print(f"\n  Lead Actors (3+ films): {len(actor_stats)}")
        print(f"    Residual range: [{min(a_res):+.3f}, {max(a_res):+.3f}]")
        print(f"    Residual std: {a_std:.3f}")
        print(f"    → 배우 영향력 ≈ {a_std:.2f} IMDB points (1σ)")

    # Tom Cruise special
    print(f"\n{sep}")
    print("  TOM CRUISE SPECIAL")
    print(sep)
    cruise_nc = None
    for nc, name in names.items():
        if name == 'Tom Cruise':
            cruise_nc = nc
            break
    if cruise_nc and cruise_nc in actor_map:
        films = actor_map[cruise_nc]
        print(f"  Tom Cruise: {len(films)} films")
        residuals = [v['residual'] for _, v in films]
        avg = sum(residuals) / len(residuals)
        print(f"  Avg residual: {avg:+.3f}")
        for t, v in sorted(films, key=lambda x: -x[1]['residual']):
            print(f"    {t[:45]:<45s} IMDB={v['rating']:.1f} Script={v['script_z']:+.2f} Res={v['residual']:+.3f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
