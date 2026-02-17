"""IMDB vs Metacritic vs Rotten Tomatoes — 잔차 비교 분석.

핵심 가설: Meryl Streep의 IMDB 잔차가 음수인 이유는
IMDB가 대중성(popularity)을 측정하기 때문이다.
비평가 점수(Metacritic/RT)로 바꾸면 양수로 반전될 수 있다.
"""

import json, sys, gzip, math
from collections import defaultdict
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
IMDB_DIR = f'{BASE}/_copyrighted/screenplay'
MC_CHECKPOINT = f'{BASE}/screenplay/metacritic_checkpoint.json'
IMSDB_CHECKPOINT = f'{BASE}/screenplay/mass_v2_checkpoint.json'
SAVANT_CHECKPOINT = f'{BASE}/screenplay/scrape_multi_checkpoint.json'

METRICS = ['dir_ratio', 'dial_zlib', 'dir_zlib', 'char_var', 'narr_std']


def normalize(t):
    t = t.lower().strip()
    for s in [', the', ', a', ', an']:
        if t.endswith(s):
            t = s.strip(', ') + ' ' + t[:-len(s)]
    return t


def t_pvalue(t_stat, df):
    abs_t = abs(t_stat)
    upper = abs_t + 80
    n_steps = 20000
    def pdf(x):
        return (1 + x * x / df) ** (-(df + 1) / 2)
    h = (upper - abs_t) / n_steps
    total = pdf(abs_t) + pdf(upper)
    for i in range(1, n_steps):
        xi = abs_t + i * h
        total += (4 if i % 2 else 2) * pdf(xi)
    tail = total * h / 3
    h2 = (2 * upper) / n_steps
    total2 = pdf(-upper) + pdf(upper)
    for i in range(1, n_steps):
        xi = -upper + i * h2
        total2 += (4 if i % 2 else 2) * pdf(xi)
    full = total2 * h2 / 3
    return min(2 * tail / full, 1.0) if full > 0 else 1.0


def load_movies():
    """Load merged IMSDB + Script Savant movies with metrics."""
    with open(IMSDB_CHECKPOINT, 'r', encoding='utf-8') as f:
        imsdb = json.load(f)['done']

    movies = {}
    for title, v in imsdb.items():
        if isinstance(v, dict) and v.get('rating') is not None and all(m in v for m in METRICS):
            movies[title] = v

    try:
        with open(SAVANT_CHECKPOINT, 'r', encoding='utf-8') as f:
            savant = json.load(f)
        seen_norm = {normalize(t) for t in movies}
        for title, v in savant.get('done', {}).items():
            if isinstance(v, dict) and all(m in v for m in METRICS):
                n = normalize(title)
                if n not in seen_norm:
                    movies[title] = v
                    seen_norm.add(n)
    except FileNotFoundError:
        pass

    print(f"  Movies with metrics: {len(movies)}")
    return movies


def match_to_imdb(movies):
    """Match titles to IMDB tconst, assign ratings if missing."""
    # Load ratings
    ratings_db = {}
    with gzip.open(f'{IMDB_DIR}/_imdb_ratings.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 3:
                try:
                    ratings_db[p[0]] = (float(p[1]), int(p[2]))
                except ValueError:
                    pass

    title_lookup = {normalize(t): t for t in movies}

    # Scan basics
    candidates = defaultdict(list)
    with gzip.open(f'{IMDB_DIR}/_imdb_basics.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 9 or p[1] != 'movie':
                continue
            for tf in [p[2], p[3]]:
                n = normalize(tf)
                if n in title_lookup:
                    orig = title_lookup[n]
                    votes = ratings_db.get(p[0], (0, 0))[1]
                    candidates[orig].append((p[0], votes))

    for title, cands in candidates.items():
        if title in movies:
            best_tc = max(cands, key=lambda x: x[1])[0]
            movies[title]['imdb_id'] = best_tc
            if movies[title].get('rating') is None and best_tc in ratings_db:
                movies[title]['rating'] = ratings_db[best_tc][0]

    movies = {k: v for k, v in movies.items() if v.get('rating') is not None and 'imdb_id' in v}
    print(f"  IMDB matched with ratings: {len(movies)}")
    return movies


def load_metacritic():
    """Load metacritic checkpoint — keyed by imdb_id."""
    with open(MC_CHECKPOINT, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['done']  # {imdb_id: {title, metascore, rt, imdb, genre, year}}


def load_crew_and_actors(tconst_set):
    """Load directors and lead actors from IMDB datasets."""
    NUL = '\\N'

    # Crew
    crew = {}
    with gzip.open(f'{IMDB_DIR}/_imdb_crew.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in tconst_set:
                d = p[1] if p[1] != NUL else ''
                crew[p[0]] = d.split(',') if d else []

    # Principals (top-2 actors)
    principals = defaultdict(list)
    with gzip.open(f'{IMDB_DIR}/_imdb_principals.tsv.gz', 'rt', encoding='utf-8') as f:
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

    # Names
    all_nc = set()
    for ds in crew.values():
        all_nc.update(ds)
    for acts in principals.values():
        all_nc.update(acts)

    names = {}
    with gzip.open(f'{IMDB_DIR}/_imdb_names.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in all_nc:
                names[p[0]] = p[1]

    print(f"  Crew: {len(crew)}, Principals: {len(principals)}, Names: {len(names)}")
    return crew, dict(principals), names


def compute_script_z(movies):
    """Compute script z-scores from metrics."""
    sr = sorted(movies.items(), key=lambda x: x[1]['rating'], reverse=True)
    n = len(sr)
    top = dict(sr[:int(n * 0.15)])
    bot = dict(sr[-int(n * 0.15):])

    mstats = {}
    for m in METRICS:
        tv = [v[m] for v in top.values() if v[m] is not None]
        bv = [v[m] for v in bot.values() if v[m] is not None]
        av = [v[m] for v in movies.values() if v[m] is not None]
        if not tv or not bv or not av:
            continue
        am = sum(av) / len(av)
        ast = (sum((x - am)**2 for x in av) / max(len(av)-1, 1)) ** 0.5
        direction = +1 if sum(tv)/len(tv) > sum(bv)/len(bv) else -1
        mstats[m] = {'mean': am, 'std': ast, 'dir': direction}

    for t, v in movies.items():
        tz, cnt = 0, 0
        for m in METRICS:
            if m in mstats and mstats[m]['std'] > 0 and v.get(m) is not None:
                tz += mstats[m]['dir'] * (v[m] - mstats[m]['mean']) / mstats[m]['std']
                cnt += 1
        v['script_z'] = tz / cnt if cnt > 0 else 0


def build_model(movies, y_key, y_label):
    """Build linear model: E[Y] = mean + slope * script_z, compute residuals."""
    vals = [(v[y_key], v['script_z']) for v in movies.values() if v.get(y_key) is not None]
    n = len(vals)
    if n < 10:
        print(f"  [{y_label}] Too few data points: {n}")
        return None

    ym = sum(y for y, _ in vals) / n
    zm = sum(z for _, z in vals) / n
    cov = sum((y - ym) * (z - zm) for y, z in vals) / n
    vz = sum((z - zm)**2 for _, z in vals) / n
    slope = cov / vz if vz > 0 else 0

    res_key = f'res_{y_label}'
    for v in movies.values():
        if v.get(y_key) is not None:
            v[res_key] = v[y_key] - (ym + slope * v['script_z'])
        else:
            v[res_key] = None

    ss_res = sum(v[res_key]**2 for v in movies.values() if v.get(res_key) is not None)
    ss_tot = sum((v[y_key] - ym)**2 for v in movies.values() if v.get(y_key) is not None)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"  [{y_label}] N={n}, E[Y] = {ym:.2f} + {slope:.3f} * script_z, R² = {r2:.4f}")
    return {'n': n, 'mean': ym, 'slope': slope, 'r2': r2, 'res_key': res_key}


def actor_residuals(movies, principals, names, res_key, min_films=3):
    """Compute per-actor average residuals for a given model."""
    act_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in principals and v.get(res_key) is not None:
            for nc in principals[tc]:
                act_map[nc].append((title, v))

    results = []
    for nc, films in act_map.items():
        if len(films) < min_films:
            continue
        name = names.get(nc, nc)
        res_vals = [v[res_key] for _, v in films]
        mr = sum(res_vals) / len(res_vals)
        sd = (sum((r - mr)**2 for r in res_vals) / (len(res_vals) - 1)) ** 0.5 if len(res_vals) > 1 else 0
        se = sd / math.sqrt(len(res_vals))
        ts = mr / se if se > 0 else 0
        p = t_pvalue(ts, len(res_vals) - 1)
        scr = [v['script_z'] for _, v in films]
        results.append({
            'name': name, 'nconst': nc, 'n': len(films),
            'avg_res': round(mr, 3), 'std': round(sd, 3),
            't': round(ts, 2), 'p': round(p, 6),
            'avg_script': round(sum(scr)/len(scr), 2),
        })

    results.sort(key=lambda x: x['t'], reverse=True)
    return results


def director_residuals(movies, crew, names, res_key, min_films=4):
    """Compute per-director average residuals."""
    dir_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in crew and v.get(res_key) is not None:
            for nc in crew[tc]:
                dir_map[nc].append((title, v))

    results = []
    for nc, films in dir_map.items():
        if len(films) < min_films:
            continue
        name = names.get(nc, nc)
        res_vals = [v[res_key] for _, v in films]
        mr = sum(res_vals) / len(res_vals)
        sd = (sum((r - mr)**2 for r in res_vals) / (len(res_vals) - 1)) ** 0.5 if len(res_vals) > 1 else 0
        se = sd / math.sqrt(len(res_vals))
        ts = mr / se if se > 0 else 0
        p = t_pvalue(ts, len(res_vals) - 1)
        results.append({
            'name': name, 'nconst': nc, 'n': len(films),
            'avg_res': round(mr, 3), 'std': round(sd, 3),
            't': round(ts, 2), 'p': round(p, 6),
        })

    results.sort(key=lambda x: x['t'], reverse=True)
    return results


def find_person(results_list, name_substr):
    """Find a person by partial name match."""
    for r in results_list:
        if name_substr.lower() in r['name'].lower():
            return r
    return None


def main():
    sep = "=" * 80
    print(sep)
    print("  IMDB vs METACRITIC vs RT — RESIDUAL COMPARISON")
    print(sep)

    # 1. Load movies
    print(f"\n  Step 1: Load movies")
    movies = load_movies()
    movies = match_to_imdb(movies)

    # 2. Load metacritic scores and merge
    print(f"\n  Step 2: Merge Metacritic/RT scores")
    mc_data = load_metacritic()

    mc_matched = 0
    for title, v in movies.items():
        imdb_id = v.get('imdb_id')
        if imdb_id and imdb_id in mc_data:
            entry = mc_data[imdb_id]
            v['metascore'] = entry.get('metascore')
            v['rt_score'] = entry.get('rt')
            mc_matched += 1

    has_mc = sum(1 for v in movies.values() if v.get('metascore') is not None)
    has_rt = sum(1 for v in movies.values() if v.get('rt_score') is not None)
    print(f"  MC data matched: {mc_matched}")
    print(f"  With Metacritic score: {has_mc}")
    print(f"  With RT score: {has_rt}")

    # 3. Compute script_z (based on IMDB top/bottom as before)
    print(f"\n  Step 3: Compute script z-scores")
    compute_script_z(movies)

    # 4. Build 3 models
    print(f"\n  Step 4: Build linear models")
    model_imdb = build_model(movies, 'rating', 'imdb')
    model_mc = build_model(movies, 'metascore', 'mc')
    model_rt = build_model(movies, 'rt_score', 'rt')

    # 5. Load crew & actors
    print(f"\n  Step 5: Load IMDB crew & cast")
    tc_set = {v['imdb_id'] for v in movies.values()}
    crew, principals, names = load_crew_and_actors(tc_set)

    # 6. Actor residuals for each model
    print(f"\n  Step 6: Actor residuals")
    actors_imdb = actor_residuals(movies, principals, names, model_imdb['res_key'])
    actors_mc = actor_residuals(movies, principals, names, model_mc['res_key']) if model_mc else []
    actors_rt = actor_residuals(movies, principals, names, model_rt['res_key']) if model_rt else []

    # 7. Director residuals
    print(f"\n  Step 7: Director residuals")
    dirs_imdb = director_residuals(movies, crew, names, model_imdb['res_key'])
    dirs_mc = director_residuals(movies, crew, names, model_mc['res_key']) if model_mc else []
    dirs_rt = director_residuals(movies, crew, names, model_rt['res_key']) if model_rt else []

    # 8. COMPARISON TABLE — key actors
    print(f"\n{sep}")
    print("  KEY ACTORS — IMDB vs MC vs RT RESIDUALS")
    print(sep)

    key_actors = [
        'Meryl Streep', 'Tom Cruise', 'Leonardo DiCaprio', 'Cary Grant',
        'Matt Damon', 'Brad Pitt', 'Robert De Niro', 'Al Pacino',
        'Tom Hanks', 'Tom Hardy', 'Kate Winslet', 'Kevin Spacey',
        'Tobey Maguire', 'Denzel Washington', 'Morgan Freeman',
        'Samuel L. Jackson', 'Natalie Portman', 'Scarlett Johansson',
        'Nicole Kidman', 'Cate Blanchett', 'Christian Bale',
        'Joaquin Phoenix', 'Ryan Gosling', 'Amy Adams',
    ]

    print(f"\n  {'Name':<22s} {'IMDB':>8s} {'MC':>8s} {'RT':>8s}  {'IMDB_t':>7s} {'MC_t':>7s} {'RT_t':>7s}  {'N_i':>3s} {'N_m':>3s} {'N_r':>3s}  Flip?")
    print(f"  {'─'*100}")

    flipped = []
    for name in key_actors:
        a_i = find_person(actors_imdb, name)
        a_m = find_person(actors_mc, name)
        a_r = find_person(actors_rt, name)

        res_i = f"{a_i['avg_res']:+.3f}" if a_i else "  N/A"
        res_m = f"{a_m['avg_res']:+.3f}" if a_m else "  N/A"
        res_r = f"{a_r['avg_res']:+.3f}" if a_r else "  N/A"
        t_i = f"{a_i['t']:+.2f}" if a_i else "  N/A"
        t_m = f"{a_m['t']:+.2f}" if a_m else "  N/A"
        t_r = f"{a_r['t']:+.2f}" if a_r else "  N/A"
        n_i = f"{a_i['n']:3d}" if a_i else "  -"
        n_m = f"{a_m['n']:3d}" if a_m else "  -"
        n_r = f"{a_r['n']:3d}" if a_r else "  -"

        # Check flip: IMDB negative → MC or RT positive
        flip = ""
        if a_i and a_i['avg_res'] < 0:
            if a_m and a_m['avg_res'] > 0:
                flip = "MC↑"
            if a_r and a_r['avg_res'] > 0:
                flip += " RT↑"
        elif a_i and a_i['avg_res'] > 0:
            if a_m and a_m['avg_res'] < 0:
                flip = "MC↓"
            if a_r and a_r['avg_res'] < 0:
                flip += " RT↓"

        if flip:
            flipped.append((name, flip))

        print(f"  {name:<22s} {res_i:>8s} {res_m:>8s} {res_r:>8s}  {t_i:>7s} {t_m:>7s} {t_r:>7s}  {n_i:>3s} {n_m:>3s} {n_r:>3s}  {flip}")

    # 9. Flipped actors
    if flipped:
        print(f"\n  FLIPPED ACTORS (IMDB → MC/RT 부호 반전):")
        for name, direction in flipped:
            print(f"    {name}: {direction}")

    # 10. KEY DIRECTORS comparison
    print(f"\n{sep}")
    print("  KEY DIRECTORS — IMDB vs MC vs RT RESIDUALS")
    print(sep)

    key_dirs = [
        'Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese',
        'David Fincher', 'Ridley Scott', 'James Cameron',
        'Quentin Tarantino', 'Tim Burton', 'Michael Mann',
        'Denis Villeneuve', 'Peter Jackson', 'Clint Eastwood',
        'Coen', 'Wachowski', 'Ron Howard', 'Woody Allen',
    ]

    print(f"\n  {'Name':<22s} {'IMDB':>8s} {'MC':>8s} {'RT':>8s}  {'IMDB_t':>7s} {'MC_t':>7s} {'RT_t':>7s}  {'N_i':>3s} {'N_m':>3s} {'N_r':>3s}")
    print(f"  {'─'*90}")

    for name in key_dirs:
        d_i = find_person(dirs_imdb, name)
        d_m = find_person(dirs_mc, name)
        d_r = find_person(dirs_rt, name)

        res_i = f"{d_i['avg_res']:+.3f}" if d_i else "  N/A"
        res_m = f"{d_m['avg_res']:+.3f}" if d_m else "  N/A"
        res_r = f"{d_r['avg_res']:+.3f}" if d_r else "  N/A"
        t_i = f"{d_i['t']:+.2f}" if d_i else "  N/A"
        t_m = f"{d_m['t']:+.2f}" if d_m else "  N/A"
        t_r = f"{d_r['t']:+.2f}" if d_r else "  N/A"
        n_i = f"{d_i['n']:3d}" if d_i else "  -"
        n_m = f"{d_m['n']:3d}" if d_m else "  -"
        n_r = f"{d_r['n']:3d}" if d_r else "  -"

        print(f"  {(d_i or d_m or d_r or {}).get('name', name):<22s} {res_i:>8s} {res_m:>8s} {res_r:>8s}  {t_i:>7s} {t_m:>7s} {t_r:>7s}  {n_i:>3s} {n_m:>3s} {n_r:>3s}")

    # 11. FULL actor rankings by each metric (top 30)
    print(f"\n{sep}")
    print("  TOP 30 ACTORS BY EACH METRIC")
    print(sep)

    for label, alist in [('IMDB', actors_imdb), ('Metacritic', actors_mc), ('RT', actors_rt)]:
        if not alist:
            continue
        print(f"\n  --- {label} residual ranking (top 30) ---")
        print(f"  {'#':>3s} {'t':>7s} {'Resid':>7s} {'N':>3s} {'Std':>5s}  Name")
        for i, a in enumerate(alist[:30]):
            print(f"  {i+1:3d} {a['t']:+7.2f} {a['avg_res']:+7.3f} {a['n']:3d} {a['std']:5.2f}  {a['name']}")

    # 12. Correlation between IMDB residual and MC residual at movie level
    print(f"\n{sep}")
    print("  MOVIE-LEVEL CORRELATION: IMDB residual vs MC residual")
    print(sep)

    if model_imdb and model_mc:
        pairs = []
        for v in movies.values():
            ri = v.get(model_imdb['res_key'])
            rm = v.get(model_mc['res_key'])
            if ri is not None and rm is not None:
                pairs.append((ri, rm))

        n = len(pairs)
        if n > 10:
            mi = sum(a for a, _ in pairs) / n
            mm = sum(b for _, b in pairs) / n
            cov = sum((a - mi) * (b - mm) for a, b in pairs) / n
            vi = sum((a - mi)**2 for a, _ in pairs) / n
            vm = sum((b - mm)**2 for _, b in pairs) / n
            corr = cov / (vi * vm) ** 0.5 if vi > 0 and vm > 0 else 0
            print(f"  N = {n} movies")
            print(f"  Pearson r(IMDB_res, MC_res) = {corr:.4f}")
            print(f"  → {'높은' if abs(corr) > 0.5 else '보통' if abs(corr) > 0.3 else '낮은'} 상관: "
                  f"{'대중과 비평이 비슷하게 본다' if corr > 0.5 else '대중과 비평의 시선이 다르다'}")

    # 13. Biggest divergences: movies where IMDB and MC residuals disagree most
    if model_imdb and model_mc:
        print(f"\n  BIGGEST DIVERGENCES (IMDB res - MC res):")
        divs = []
        for title, v in movies.items():
            ri = v.get(model_imdb['res_key'])
            rm = v.get(model_mc['res_key'])
            if ri is not None and rm is not None:
                # Normalize MC residual to same scale as IMDB (MC is 0-100, IMDB is 0-10)
                rm_scaled = rm / 10.0
                divs.append((title, ri, rm_scaled, ri - rm_scaled))

        divs.sort(key=lambda x: x[3], reverse=True)
        print(f"\n  대중 > 비평 (IMDB residual >> MC residual):")
        for title, ri, rm, diff in divs[:10]:
            print(f"    {title:<40s} IMDB_res={ri:+.2f}, MC_res(scaled)={rm:+.2f}, gap={diff:+.2f}")

        print(f"\n  비평 > 대중 (MC residual >> IMDB residual):")
        for title, ri, rm, diff in divs[-10:]:
            print(f"    {title:<40s} IMDB_res={ri:+.2f}, MC_res(scaled)={rm:+.2f}, gap={diff:+.2f}")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == "__main__":
    main()
