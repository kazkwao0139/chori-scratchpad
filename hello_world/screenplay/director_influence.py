"""
감독/배우 영향력 정량화: 각본 구조 점수 대비 실제 평점의 잔차(residual) 분석.

접근법:
  1. 244편의 각본 구조 z-score (4 KEY_METRICS) 계산
  2. IMDB 데이터셋에서 감독/배우 정보 매칭
  3. 같은 감독이 2+편 연출한 경우, 평균 잔차 = 감독의 "filling power"
  4. 같은 배우가 2+편 출연한 경우, 평균 잔차 = 배우의 "filling power"

잔차 = 실제 IMDB 평점 - 각본 구조에서 기대되는 평점
양수 잔차 = 각본보다 더 잘 나옴 (감독/배우 기여)
음수 잔차 = 각본보다 못 나옴 (감독/배우가 깎아먹음)
"""

import json
import sys
import gzip
import math
import urllib.request
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
CHECKPOINT = f'{BASE}/screenplay/plot_entropy_checkpoint.json'
FIXED_RATINGS = f'{BASE}/screenplay/_fixed_ratings.json'
IMDB_BASICS = f'{BASE}/_copyrighted/screenplay/_imdb_basics.tsv.gz'
IMDB_RATINGS = f'{BASE}/_copyrighted/screenplay/_imdb_ratings.tsv.gz'
IMDB_DIR = f'{BASE}/_copyrighted/screenplay'

KEY_METRICS = ['repeat_ratio', 'arc_shift', 'setup_front', 'bi_entropy']
MIN_SCENES = 15


def download_if_missing(filename, url):
    path = f"{IMDB_DIR}/{filename}"
    if Path(path).exists():
        size = Path(path).stat().st_size
        print(f"  [EXISTS] {filename} ({size/1024/1024:.1f} MB)")
        return path
    print(f"  [DOWNLOAD] {filename} from {url}")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        with open(path, 'wb') as f:
            f.write(data)
        print(f"  [OK] {filename} ({len(data)/1024/1024:.1f} MB)")
        return path
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return None


def load_movies():
    """Load 244 movies with 4 key metrics and ratings."""
    cp = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
    fr = json.load(open(FIXED_RATINGS, 'r', encoding='utf-8'))

    valid = {}
    for k, v in cp.items():
        if not isinstance(v, dict):
            continue
        if not v.get('rating') or not v.get('labels'):
            continue
        if not all(m in v for m in KEY_METRICS):
            continue
        if v.get('n_scenes', 0) < MIN_SCENES:
            continue
        valid[k] = v

    # Attach imdb_id where available
    for title in valid:
        if title in fr and 'imdb_id' in fr[title]:
            valid[title]['imdb_id'] = fr[title]['imdb_id']

    return valid


def match_titles_to_tconst(movies):
    """Match remaining movies without imdb_id using _imdb_basics.tsv.gz."""
    unmatched = {k for k in movies if 'imdb_id' not in movies[k]}
    if not unmatched:
        return movies

    print(f"\n  Matching {len(unmatched)} movies via IMDB basics...")

    # Normalize titles for matching
    def normalize(t):
        t = t.lower().strip()
        for suffix in [', the', ', a', ', an']:
            if t.endswith(suffix):
                t = suffix.strip(', ') + ' ' + t[:-len(suffix)]
        return t

    # Build lookup: normalized title -> original title
    title_lookup = {}
    for t in unmatched:
        title_lookup[normalize(t)] = t
        # Also try without suffixes like "_bad", " v2"
        clean = t.replace('_bad', '').replace(' v2', '').strip()
        if clean != t:
            title_lookup[normalize(clean)] = t

    # We also need ratings to pick the best match (most voted)
    ratings = {}
    print("  Loading IMDB ratings...")
    with gzip.open(IMDB_RATINGS, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    ratings[parts[0]] = int(parts[2])  # votes
                except ValueError:
                    pass

    # Scan basics for title matches (movie type only)
    candidates = defaultdict(list)  # original_title -> [(tconst, votes)]
    print("  Scanning IMDB basics for matches...")
    with gzip.open(IMDB_BASICS, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            tconst, ttype, primary, original = parts[0], parts[1], parts[2], parts[3]
            if ttype != 'movie':
                continue

            for title_field in [primary, original]:
                norm = normalize(title_field)
                if norm in title_lookup:
                    orig = title_lookup[norm]
                    votes = ratings.get(tconst, 0)
                    candidates[orig].append((tconst, votes, primary))

    # Pick best match (most votes) for each
    matched = 0
    for orig_title, cands in candidates.items():
        if orig_title in movies and 'imdb_id' not in movies[orig_title]:
            best = max(cands, key=lambda x: x[1])
            movies[orig_title]['imdb_id'] = best[0]
            matched += 1

    still_missing = [k for k in movies if 'imdb_id' not in movies[k]]
    print(f"  Matched {matched} more. Still missing: {len(still_missing)}")
    if still_missing:
        print(f"    Missing: {still_missing[:10]}...")

    return movies


def load_crew(tconst_set):
    """Load director info from title.crew.tsv.gz for our movies."""
    path = download_if_missing(
        '_imdb_crew.tsv.gz',
        'https://datasets.imdbws.com/title.crew.tsv.gz'
    )
    if not path:
        return {}

    crew = {}
    print("  Scanning crew data...")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            tc = parts[0]
            if tc in tconst_set:
                directors = parts[1] if parts[1] != '\\N' else ''
                crew[tc] = directors.split(',') if directors else []
    return crew


def load_principals(tconst_set):
    """Load principal cast from title.principals.tsv.gz."""
    path = download_if_missing(
        '_imdb_principals.tsv.gz',
        'https://datasets.imdbws.com/title.principals.tsv.gz'
    )
    if not path:
        return {}

    principals = defaultdict(list)
    print("  Scanning principals data...")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            tc, ordering, nconst, category = parts[0], parts[1], parts[2], parts[3]
            if tc in tconst_set:
                if category in ('actor', 'actress'):
                    try:
                        principals[tc].append((int(ordering), nconst))
                    except ValueError:
                        pass
    # Sort by ordering (billing) and keep top 2
    for tc in principals:
        principals[tc].sort()
        principals[tc] = [nc for _, nc in principals[tc][:2]]
    return dict(principals)


def load_names(nconst_set):
    """Load names from name.basics.tsv.gz for relevant people."""
    path = download_if_missing(
        '_imdb_names.tsv.gz',
        'https://datasets.imdbws.com/name.basics.tsv.gz'
    )
    if not path:
        return {}

    names = {}
    print("  Scanning name data...")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            nc = parts[0]
            if nc in nconst_set:
                names[nc] = parts[1]
    return names


def compute_script_scores(movies):
    """Compute script z-scores using hidden_gems methodology."""
    good = {k: v for k, v in movies.items() if v['rating'] >= 8.0}
    bad = {k: v for k, v in movies.items() if v['rating'] < 4.0}

    if len(good) < 3 or len(bad) < 3:
        print(f"  WARNING: good={len(good)}, bad={len(bad)} — too few for reliable stats")

    # Direction auto-detection
    stats = {}
    for m in KEY_METRICS:
        gv = [v[m] for v in good.values()]
        bv = [v[m] for v in bad.values()]
        g_mean = sum(gv) / len(gv)
        g_std = (sum((x - g_mean)**2 for x in gv) / max(len(gv)-1, 1)) ** 0.5
        b_mean = sum(bv) / len(bv)
        direction = +1 if g_mean > b_mean else -1
        stats[m] = {'g_mean': g_mean, 'g_std': g_std, 'direction': direction}

    # Compute z-score for each movie
    for title, v in movies.items():
        total_z = 0
        for m in KEY_METRICS:
            s = stats[m]
            z = (v[m] - s['g_mean']) / s['g_std'] if s['g_std'] > 0 else 0
            total_z += s['direction'] * z
        v['script_z'] = total_z / len(KEY_METRICS)

    return stats


def compute_residuals(movies):
    """Compute residual: actual rating - expected rating from script z-score.

    Expected rating = simple linear mapping from script_z.
    Using the whole dataset: E[rating | script_z] = mean_rating + slope * script_z
    """
    rated = [(v['rating'], v['script_z']) for v in movies.values()]
    n = len(rated)
    r_mean = sum(r for r, _ in rated) / n
    z_mean = sum(z for _, z in rated) / n

    # OLS slope: cov(r, z) / var(z)
    cov = sum((r - r_mean) * (z - z_mean) for r, z in rated) / n
    var_z = sum((z - z_mean)**2 for _, z in rated) / n
    slope = cov / var_z if var_z > 0 else 0

    print(f"\n  Linear model: E[rating] = {r_mean:.2f} + {slope:.3f} * script_z")
    print(f"  (slope > 0 means better script → higher rating)")

    # Residual for each movie
    for title, v in movies.items():
        expected = r_mean + slope * v['script_z']
        v['expected_rating'] = expected
        v['residual'] = v['rating'] - expected

    # R² for sanity
    ss_res = sum(v['residual']**2 for v in movies.values())
    ss_tot = sum((v['rating'] - r_mean)**2 for v in movies.values())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  R² = {r2:.4f} (script explains {r2*100:.1f}% of rating variance)")

    return r_mean, slope


def main():
    sep = "=" * 70

    print(sep)
    print("  DIRECTOR & ACTOR INFLUENCE — 각본 너머의 영향력 정량화")
    print(sep)

    # ── Step 1: Load movies ──
    print(f"\n{'─' * 70}")
    print("  Step 1: Load movie data")
    print(f"{'─' * 70}")
    movies = load_movies()
    print(f"  Loaded {len(movies)} movies with 4 key metrics")

    # ── Step 2: Match all to tconst ──
    print(f"\n{'─' * 70}")
    print("  Step 2: Match titles to IMDB IDs")
    print(f"{'─' * 70}")
    movies = match_titles_to_tconst(movies)

    have_id = {k: v for k, v in movies.items() if 'imdb_id' in v}
    print(f"  Total with IMDB ID: {len(have_id)} / {len(movies)}")

    # ── Step 3: Download & load IMDB crew/principals/names ──
    print(f"\n{'─' * 70}")
    print("  Step 3: Download IMDB datasets")
    print(f"{'─' * 70}")

    tconst_set = {v['imdb_id'] for v in have_id.values()}

    crew = load_crew(tconst_set)
    print(f"  Found crew data for {len(crew)} movies")

    principals = load_principals(tconst_set)
    print(f"  Found principal cast for {len(principals)} movies")

    # Collect all nconsts we need names for
    all_nconsts = set()
    for dirs in crew.values():
        all_nconsts.update(dirs)
    for actors in principals.values():
        all_nconsts.update(actors)

    print(f"  Need names for {len(all_nconsts)} people")
    names = load_names(all_nconsts)
    print(f"  Resolved {len(names)} names")

    # ── Step 4: Compute script z-scores ──
    print(f"\n{'─' * 70}")
    print("  Step 4: Compute script z-scores & residuals")
    print(f"{'─' * 70}")

    stats = compute_script_scores(movies)
    r_mean, slope = compute_residuals(movies)

    # ── Step 5: Build director → movies mapping ──
    print(f"\n{'─' * 70}")
    print("  Step 5: Group by director")
    print(f"{'─' * 70}")

    director_movies = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if not tc or tc not in crew:
            continue
        for nc in crew[tc]:
            director_movies[nc].append((title, v))

    # Filter: 2+ movies
    multi = {nc: ms for nc, ms in director_movies.items() if len(ms) >= 2}
    print(f"  Directors with 1 movie: {sum(1 for ms in director_movies.values() if len(ms) == 1)}")
    print(f"  Directors with 2+ movies: {len(multi)}")
    total_covered = sum(len(ms) for ms in multi.values())
    print(f"  Movies covered: {total_covered}")

    # Compute per-director stats
    director_stats = []
    for nc, ms in multi.items():
        name = names.get(nc, nc)
        residuals = [v['residual'] for _, v in ms]
        ratings = [v['rating'] for _, v in ms]
        scripts = [v['script_z'] for _, v in ms]
        avg_res = sum(residuals) / len(residuals)
        avg_rating = sum(ratings) / len(ratings)
        avg_script = sum(scripts) / len(scripts)
        # Std of residuals (consistency)
        if len(residuals) > 1:
            res_std = (sum((r - avg_res)**2 for r in residuals) / (len(residuals)-1)) ** 0.5
        else:
            res_std = 0
        film_list = [(t, v['rating'], v['script_z'], v['residual']) for t, v in ms]
        director_stats.append({
            'name': name,
            'nconst': nc,
            'n_films': len(ms),
            'avg_residual': avg_res,
            'avg_rating': avg_rating,
            'avg_script_z': avg_script,
            'res_std': res_std,
            'films': film_list,
        })

    # Sort by avg_residual (descending = most value added)
    director_stats.sort(key=lambda x: x['avg_residual'], reverse=True)

    print(f"\n{sep}")
    print("  DIRECTOR INFLUENCE RANKING")
    print(f"  (Residual = Actual IMDB - Expected from Script)")
    print(f"  양수 = 감독이 각본 이상으로 끌어올림 | 음수 = 감독이 깎아먹음")
    print(sep)
    print(f"\n  {'#':>3s} {'Residual':>9s} {'AvgIMDB':>8s} {'Script_z':>9s} {'Films':>5s} {'Std':>5s}  {'Director':<25s}")
    print(f"  {'─'*75}")

    for i, d in enumerate(director_stats):
        print(f"  {i+1:3d} {d['avg_residual']:+9.3f} {d['avg_rating']:8.1f} {d['avg_script_z']:+9.3f} {d['n_films']:5d} {d['res_std']:5.2f}  {d['name']:<25s}")

    # Detail for top 5 and bottom 5
    print(f"\n{sep}")
    print("  TOP 5 DIRECTORS — 각본 이상으로 끌어올리는 감독들")
    print(sep)
    for d in director_stats[:5]:
        print(f"\n  {d['name']} ({d['n_films']}편, avg residual={d['avg_residual']:+.3f})")
        for t, r, sz, res in d['films']:
            print(f"    {t[:40]:<40s}  IMDB={r:.1f}  Script_z={sz:+.2f}  Residual={res:+.3f}")

    print(f"\n{sep}")
    print("  BOTTOM 5 DIRECTORS — 각본 대비 깎아먹는 감독들")
    print(sep)
    for d in director_stats[-5:]:
        print(f"\n  {d['name']} ({d['n_films']}편, avg residual={d['avg_residual']:+.3f})")
        for t, r, sz, res in d['films']:
            print(f"    {t[:40]:<40s}  IMDB={r:.1f}  Script_z={sz:+.2f}  Residual={res:+.3f}")

    # ── Step 6: Actor analysis ──
    print(f"\n{'─' * 70}")
    print("  Step 6: Group by lead actor")
    print(f"{'─' * 70}")

    actor_movies = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if not tc or tc not in principals:
            continue
        for nc in principals[tc]:
            actor_movies[nc].append((title, v))

    multi_actors = {nc: ms for nc, ms in actor_movies.items() if len(ms) >= 2}
    print(f"  Lead actors with 2+ movies: {len(multi_actors)}")
    total_actor = sum(len(ms) for ms in multi_actors.values())
    print(f"  Movies covered: {total_actor}")

    actor_stats = []
    for nc, ms in multi_actors.items():
        name = names.get(nc, nc)
        residuals = [v['residual'] for _, v in ms]
        ratings = [v['rating'] for _, v in ms]
        scripts = [v['script_z'] for _, v in ms]
        avg_res = sum(residuals) / len(residuals)
        avg_rating = sum(ratings) / len(ratings)
        avg_script = sum(scripts) / len(scripts)
        if len(residuals) > 1:
            res_std = (sum((r - avg_res)**2 for r in residuals) / (len(residuals)-1)) ** 0.5
        else:
            res_std = 0
        film_list = [(t, v['rating'], v['script_z'], v['residual']) for t, v in ms]
        actor_stats.append({
            'name': name,
            'nconst': nc,
            'n_films': len(ms),
            'avg_residual': avg_res,
            'avg_rating': avg_rating,
            'avg_script_z': avg_script,
            'res_std': res_std,
            'films': film_list,
        })

    actor_stats.sort(key=lambda x: x['avg_residual'], reverse=True)

    print(f"\n{sep}")
    print("  LEAD ACTOR INFLUENCE RANKING")
    print(f"  (Top 2 billed actors per movie)")
    print(f"  양수 = 배우가 각본 이상으로 끌어올림 | 음수 = 배우가 깎아먹음")
    print(sep)
    print(f"\n  {'#':>3s} {'Residual':>9s} {'AvgIMDB':>8s} {'Script_z':>9s} {'Films':>5s} {'Std':>5s}  {'Actor':<25s}")
    print(f"  {'─'*75}")

    for i, a in enumerate(actor_stats):
        print(f"  {i+1:3d} {a['avg_residual']:+9.3f} {a['avg_rating']:8.1f} {a['avg_script_z']:+9.3f} {a['n_films']:5d} {a['res_std']:5.2f}  {a['name']:<25s}")

    # Detail for top 5 and bottom 5
    print(f"\n{sep}")
    print("  TOP 5 ACTORS — 각본 이상으로 끌어올리는 배우들")
    print(sep)
    for a in actor_stats[:5]:
        print(f"\n  {a['name']} ({a['n_films']}편, avg residual={a['avg_residual']:+.3f})")
        for t, r, sz, res in a['films']:
            print(f"    {t[:40]:<40s}  IMDB={r:.1f}  Script_z={sz:+.2f}  Residual={res:+.3f}")

    print(f"\n{sep}")
    print("  BOTTOM 5 ACTORS — 각본 대비 깎아먹는 배우들")
    print(sep)
    for a in actor_stats[-5:]:
        print(f"\n  {a['name']} ({a['n_films']}편, avg residual={a['avg_residual']:+.3f})")
        for t, r, sz, res in a['films']:
            print(f"    {t[:40]:<40s}  IMDB={r:.1f}  Script_z={sz:+.2f}  Residual={res:+.3f}")

    # ── Summary statistics ──
    print(f"\n{sep}")
    print("  SUMMARY STATISTICS")
    print(sep)

    if director_stats:
        d_residuals = [d['avg_residual'] for d in director_stats]
        d_mean = sum(d_residuals) / len(d_residuals)
        d_std = (sum((r - d_mean)**2 for r in d_residuals) / max(len(d_residuals)-1, 1)) ** 0.5
        print(f"\n  Directors (2+ films): {len(director_stats)}")
        print(f"    Residual range: [{min(d_residuals):+.3f}, {max(d_residuals):+.3f}]")
        print(f"    Residual mean: {d_mean:+.3f}, std: {d_std:.3f}")
        print(f"    Interpretation: 감독이 각본에 더하는 영향의 크기 ≈ {d_std:.2f} IMDB points")

    if actor_stats:
        a_residuals = [a['avg_residual'] for a in actor_stats]
        a_mean = sum(a_residuals) / len(a_residuals)
        a_std = (sum((r - a_mean)**2 for r in a_residuals) / max(len(a_residuals)-1, 1)) ** 0.5
        print(f"\n  Lead Actors (2+ films): {len(actor_stats)}")
        print(f"    Residual range: [{min(a_residuals):+.3f}, {max(a_residuals):+.3f}]")
        print(f"    Residual mean: {a_mean:+.3f}, std: {a_std:.3f}")
        print(f"    Interpretation: 주연배우가 각본에 더하는 영향의 크기 ≈ {a_std:.2f} IMDB points")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
