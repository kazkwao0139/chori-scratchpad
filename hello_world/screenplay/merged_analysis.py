from pathlib import Path
"""
통합 감독/배우 영향력 분석 — IMSDB + Script Savant 합산.

두 데이터 소스를 합쳐서:
1. 메트릭 z-score → script_z
2. 선형 모델 → residual
3. 감독/배우 그루핑 → influence ranking
4. t-test → 통계적 유의성
5. influence_data.json 갱신
6. DIRECTOR_INFLUENCE.md 갱신
"""

import json, sys, gzip, math
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
IMSDB_CHECKPOINT = f'{BASE}/screenplay/mass_v2_checkpoint.json'
SAVANT_CHECKPOINT = f'{BASE}/screenplay/scrape_multi_checkpoint.json'
IMDB_DIR = f'{BASE}/_copyrighted/screenplay'

METRICS = ['dir_ratio', 'dial_zlib', 'dir_zlib', 'char_var', 'narr_std']


def normalize(t):
    t = t.lower().strip()
    for s in [', the', ', a', ', an']:
        if t.endswith(s):
            t = s.strip(', ') + ' ' + t[:-len(s)]
    return t


def t_pvalue(t_stat, df):
    """Numerical integration of Student-t tail probability."""
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


def load_merged():
    """Load and merge IMSDB + Script Savant data.

    IMSDB entries already have ratings.
    Script Savant entries may not have ratings yet (assigned in scraper Step 3).
    We accept all entries with metrics; ratings are assigned in match_titles().
    """
    # IMSDB (already has ratings)
    with open(IMSDB_CHECKPOINT, 'r', encoding='utf-8') as f:
        imsdb = json.load(f)['done']

    imsdb_movies = {}
    for title, v in imsdb.items():
        if isinstance(v, dict) and v.get('rating') is not None and all(m in v for m in METRICS):
            imsdb_movies[title] = v
            imsdb_movies[title]['_source'] = 'imsdb'

    print(f"  IMSDB: {len(imsdb_movies)} movies with ratings + metrics")

    # Script Savant (may or may not have ratings yet)
    savant_movies = {}
    savant_no_rating = 0
    try:
        with open(SAVANT_CHECKPOINT, 'r', encoding='utf-8') as f:
            savant = json.load(f)
        done = savant.get('done', {})
        for title, v in done.items():
            if isinstance(v, dict) and all(m in v for m in METRICS):
                savant_movies[title] = v
                savant_movies[title]['_source'] = 'scriptsavant'
                if v.get('rating') is None:
                    savant_no_rating += 1
        print(f"  Script Savant: {len(savant_movies)} movies with metrics ({savant_no_rating} need rating)")
    except FileNotFoundError:
        print("  Script Savant checkpoint not found — using IMSDB only")

    # Merge: IMSDB takes priority for duplicates
    merged = {}
    seen_norm = {}
    for title, v in imsdb_movies.items():
        n = normalize(title)
        merged[title] = v
        seen_norm[n] = title

    added = 0
    for title, v in savant_movies.items():
        n = normalize(title)
        if n not in seen_norm:
            merged[title] = v
            seen_norm[n] = title
            added += 1

    print(f"  Merged total: {len(merged)} movies ({added} new from Script Savant)")
    return merged


def match_titles(movies):
    """Match movies to IMDB tconst and assign ratings if missing."""
    print("  Loading IMDB ratings...")
    ratings_db = {}  # tconst -> (rating, votes)
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

    print("  Scanning IMDB basics...")
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

    matched = 0
    rating_assigned = 0
    for title, cands in candidates.items():
        if title in movies:
            best_tc = max(cands, key=lambda x: x[1])[0]
            movies[title]['imdb_id'] = best_tc
            matched += 1
            # Assign rating if missing (Script Savant entries)
            if movies[title].get('rating') is None and best_tc in ratings_db:
                movies[title]['rating'] = ratings_db[best_tc][0]
                movies[title]['votes'] = ratings_db[best_tc][1]
                rating_assigned += 1

    # Drop movies without rating
    before = len(movies)
    movies = {k: v for k, v in movies.items() if v.get('rating') is not None}
    dropped = before - len(movies)

    print(f"  IMDB matched: {matched} / {before}")
    print(f"  Ratings assigned to Script Savant entries: {rating_assigned}")
    print(f"  Dropped (no rating): {dropped}")
    print(f"  Final dataset: {len(movies)} movies")
    return movies


def load_crew_principals_names(tconst_set):
    """Load directors, actors, names from IMDB datasets."""
    # Crew (directors)
    crew = {}
    with gzip.open(f'{IMDB_DIR}/_imdb_crew.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in tconst_set:
                d = p[1] if p[1] != '\\N' else ''
                crew[p[0]] = d.split(',') if d else []

    # Principals (actors)
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

    print(f"  Directors for {len(crew)} movies, actors for {len(principals)} movies, {len(names)} names")
    return crew, dict(principals), names


def compute_scores(movies):
    """Compute script z-scores and residuals."""
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
        mstats[m] = {
            'mean': am, 'std': ast, 'dir': direction,
            'top': round(sum(tv)/len(tv), 4),
            'bot': round(sum(bv)/len(bv), 4),
        }

    print(f"\n  Direction detection:")
    for m in METRICS:
        if m in mstats:
            s = mstats[m]
            arrow = "↑" if s['dir'] > 0 else "↓"
            print(f"    {m:>12s}: top={s['top']:.4f} bot={s['bot']:.4f} {arrow}")

    for t, v in movies.items():
        tz, cnt = 0, 0
        for m in METRICS:
            if m in mstats and mstats[m]['std'] > 0 and v.get(m) is not None:
                tz += mstats[m]['dir'] * (v[m] - mstats[m]['mean']) / mstats[m]['std']
                cnt += 1
        v['script_z'] = tz / cnt if cnt > 0 else 0

    rated = [(v['rating'], v['script_z']) for v in movies.values()]
    n_all = len(rated)
    rm = sum(r for r, _ in rated) / n_all
    zm = sum(z for _, z in rated) / n_all
    cov = sum((r - rm) * (z - zm) for r, z in rated) / n_all
    vz = sum((z - zm)**2 for _, z in rated) / n_all
    slope = cov / vz if vz > 0 else 0

    for t, v in movies.items():
        v['residual'] = v['rating'] - (rm + slope * v['script_z'])

    ss_res = sum(v['residual']**2 for v in movies.values())
    ss_tot = sum((v['rating'] - rm)**2 for v in movies.values())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Model: E[rating] = {rm:.2f} + {slope:.3f} * script_z")
    print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
    return mstats, rm, slope, r2


def analyze_with_ttest(group_map, names, min_films=3):
    """Group analysis with t-test."""
    results = []
    for nc, films in group_map.items():
        if len(films) < min_films:
            continue
        name = names.get(nc, nc)
        res = [v['residual'] for _, v in films]
        rat = [v['rating'] for _, v in films]
        scr = [v['script_z'] for _, v in films]
        mr = sum(res) / len(res)
        sd = (sum((r - mr)**2 for r in res) / (len(res) - 1)) ** 0.5 if len(res) > 1 else 0
        se = sd / math.sqrt(len(res))
        ts = mr / se if se > 0 else 0
        p = t_pvalue(ts, len(res) - 1)

        fl = []
        for t, v in sorted(films, key=lambda x: -x[1]['residual']):
            fl.append({
                'title': t, 'rating': v['rating'],
                'script_z': round(v['script_z'], 3),
                'residual': round(v['residual'], 3),
            })

        results.append({
            'name': name, 'nconst': nc, 'n': len(films),
            'avg_res': round(mr, 3), 'std': round(sd, 3),
            't': round(ts, 2), 'p': round(p, 6),
            'avg_rating': round(sum(rat)/len(rat), 1),
            'avg_script': round(sum(scr)/len(scr), 2),
            'films': fl,
        })

    results.sort(key=lambda x: x['t'], reverse=True)
    return results


def print_results(results, label, n_sig_check=None):
    """Print ranking with t-test results."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {label} — t-test RANKING (sorted by t-statistic)")
    print(sep)
    print(f"  {'#':>3s} {'t':>7s} {'p':>10s} {'Resid':>7s} {'IMDB':>6s} {'ScrZ':>6s} {'N':>3s} {'Std':>5s}  Name")
    print(f"  {'─'*78}")

    if n_sig_check is None:
        n_sig_check = len(results)
    bonf = 0.05 / n_sig_check if n_sig_check > 0 else 0.05

    sig_bonf = []
    sig_05 = []
    sig_neg = []

    for i, d in enumerate(results):
        marker = ''
        if d['p'] < bonf:
            marker = ' ***'
            sig_bonf.append(d)
        elif d['p'] < 0.05:
            marker = ' *'
            sig_05.append(d)
        if d['avg_res'] < 0 and d['p'] < 0.05:
            sig_neg.append(d)

        if i < 20 or i >= len(results) - 10 or d['p'] < 0.05:
            print(f"  {i+1:3d} {d['t']:+7.2f} {d['p']:10.6f} {d['avg_res']:+7.3f} {d['avg_rating']:6.1f} {d['avg_script']:+6.2f} {d['n']:3d} {d['std']:5.2f}  {d['name']}{marker}")
        elif i == 20:
            print(f"  {'···':>3s}")

    print(f"\n  Bonferroni α = 0.05/{n_sig_check} = {bonf:.6f}")
    print(f"  Bonferroni survivors (***): {len(sig_bonf)}")
    for d in sig_bonf:
        print(f"    {d['name']}: t={d['t']:+.2f}, p={d['p']:.6f}, n={d['n']}, avg_res={d['avg_res']:+.3f}")
    print(f"  Nominally significant (p<0.05, *): {len(sig_05)}")
    for d in sig_05:
        print(f"    {d['name']}: t={d['t']:+.2f}, p={d['p']:.6f}, n={d['n']}, avg_res={d['avg_res']:+.3f}")
    if sig_neg:
        print(f"  Negative significant (p<0.05): {len(sig_neg)}")
        for d in sig_neg:
            print(f"    {d['name']}: t={d['t']:+.2f}, p={d['p']:.6f}, n={d['n']}, avg_res={d['avg_res']:+.3f}")

    return sig_bonf, sig_05


def save_influence_json(meta, directors, actors):
    """Save influence_data.json."""
    output = {
        'meta': meta,
        'directors': [{k: v for k, v in d.items() if k != 'nconst'} for d in directors],
        'actors': [{k: v for k, v in a.items() if k != 'nconst'} for a in actors],
    }
    path = f'{BASE}/screenplay/influence_data.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {path}")
    print(f"  {len(directors)} directors, {len(actors)} actors")


def generate_markdown(meta, directors, actors, movies):
    """Generate updated DIRECTOR_INFLUENCE.md."""
    lines = []
    lines.append("# Director & Actor Influence Analysis")
    lines.append("")
    lines.append(f"> **Dataset**: {meta['n_movies']} movies (IMSDB + Script Savant)")
    lines.append(f"> **IMDB Matched**: {meta['n_matched']} movies")
    lines.append(f"> **Generated**: 2026-02-16")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("1. 각본 텍스트에서 5개 메트릭 추출 (dir_ratio, dial_zlib, dir_zlib, char_var, narr_std)")
    lines.append("2. 상위 15% vs 하위 15% 비교로 방향 자동 감지 → 복합 z-score (`script_z`)")
    lines.append(f"3. 선형 모델: `E[rating] = {meta['r_mean']:.2f} + {meta['slope']:.3f} × script_z` (R² = {meta['r2']:.4f})")
    lines.append("4. `residual = actual_IMDB - expected` → 각본 이상/이하 기여도")
    lines.append("5. 감독/배우별 그루핑 → one-sample t-test (H₀: mean residual = 0)")
    lines.append("")

    lines.append("## Metric Directions")
    lines.append("")
    lines.append("| Metric | Top 15% | Bot 15% | Direction |")
    lines.append("|--------|---------|---------|-----------|")
    for m in meta['metrics']:
        md = meta['metric_directions'][m]
        arrow = "↑ higher=better" if md['dir'] > 0 else "↓ lower=better"
        lines.append(f"| {m} | {md['top']:.4f} | {md['bot']:.4f} | {arrow} |")
    lines.append("")

    # Directors
    dir_bonf = 0.05 / len(directors) if directors else 1
    lines.append(f"## Directors ({len(directors)} with 4+ films)")
    lines.append("")
    lines.append(f"Bonferroni α = 0.05/{len(directors)} = {dir_bonf:.6f}")
    lines.append("")
    lines.append("| # | Name | N | Avg Res | t | p | IMDB | ScriptZ | Sig |")
    lines.append("|---|------|---|---------|---|---|------|---------|-----|")
    for i, d in enumerate(directors):
        sig = "***" if d['p'] < dir_bonf else ("*" if d['p'] < 0.05 else "")
        lines.append(f"| {i+1} | {d['name']} | {d['n']} | {d['avg_res']:+.3f} | {d['t']:+.2f} | {d['p']:.6f} | {d['avg_rating']:.1f} | {d['avg_script']:+.2f} | {sig} |")
    lines.append("")

    # Top directors detail
    bonf_dirs = [d for d in directors if d['p'] < dir_bonf]
    sig_dirs = [d for d in directors if d['p'] < 0.05]
    lines.append(f"### Bonferroni Survivors ({len(bonf_dirs)})")
    lines.append("")
    for d in bonf_dirs:
        lines.append(f"#### {d['name']} ({d['n']}편, avg residual = {d['avg_res']:+.3f}, p = {d['p']:.6f})")
        lines.append("")
        lines.append("| Title | IMDB | Script Z | Residual |")
        lines.append("|-------|------|----------|----------|")
        for f in d['films']:
            lines.append(f"| {f['title']} | {f['rating']:.1f} | {f['script_z']:+.3f} | {f['residual']:+.3f} |")
        lines.append("")

    non_bonf_sig = [d for d in sig_dirs if d['p'] >= dir_bonf]
    if non_bonf_sig:
        lines.append(f"### Nominally Significant p<0.05 ({len(non_bonf_sig)})")
        lines.append("")
        for d in non_bonf_sig:
            lines.append(f"#### {d['name']} ({d['n']}편, avg residual = {d['avg_res']:+.3f}, p = {d['p']:.6f})")
            lines.append("")
            lines.append("| Title | IMDB | Script Z | Residual |")
            lines.append("|-------|------|----------|----------|")
            for f in d['films']:
                lines.append(f"| {f['title']} | {f['rating']:.1f} | {f['script_z']:+.3f} | {f['residual']:+.3f} |")
            lines.append("")

    # Actors
    act_bonf = 0.05 / len(actors) if actors else 1
    lines.append(f"## Lead Actors ({len(actors)} with 3+ films)")
    lines.append("")
    lines.append(f"Bonferroni α = 0.05/{len(actors)} = {act_bonf:.6f}")
    lines.append("")
    lines.append("| # | Name | N | Avg Res | t | p | IMDB | ScriptZ | Sig |")
    lines.append("|---|------|---|---------|---|---|------|---------|-----|")
    for i, a in enumerate(actors):
        sig = "***" if a['p'] < act_bonf else ("*" if a['p'] < 0.05 else "")
        lines.append(f"| {i+1} | {a['name']} | {a['n']} | {a['avg_res']:+.3f} | {a['t']:+.2f} | {a['p']:.6f} | {a['avg_rating']:.1f} | {a['avg_script']:+.2f} | {sig} |")
    lines.append("")

    # Top actors detail
    bonf_acts = [a for a in actors if a['p'] < act_bonf]
    if bonf_acts:
        lines.append(f"### Bonferroni Survivors ({len(bonf_acts)})")
        lines.append("")
        for a in bonf_acts:
            lines.append(f"#### {a['name']} ({a['n']}편, avg residual = {a['avg_res']:+.3f}, p = {a['p']:.6f})")
            lines.append("")
            lines.append("| Title | IMDB | Script Z | Residual |")
            lines.append("|-------|------|----------|----------|")
            for f in a['films']:
                lines.append(f"| {f['title']} | {f['rating']:.1f} | {f['script_z']:+.3f} | {f['residual']:+.3f} |")
            lines.append("")

    # Tom Cruise
    cruise = None
    for a in actors:
        if a['name'] == 'Tom Cruise':
            cruise = a
            break
    if cruise:
        lines.append("## Tom Cruise Special")
        lines.append("")
        lines.append(f"- **{cruise['n']}편**, avg residual = {cruise['avg_res']:+.3f}, t = {cruise['t']:+.2f}, p = {cruise['p']:.6f}")
        all_pos = all(f['residual'] > 0 for f in cruise['films'])
        lines.append(f"- 전 작품 양수 residual: {'Yes' if all_pos else 'No'}")
        lines.append("")
        lines.append("| Title | IMDB | Script Z | Residual |")
        lines.append("|-------|------|----------|----------|")
        for f in cruise['films']:
            lines.append(f"| {f['title']} | {f['rating']:.1f} | {f['script_z']:+.3f} | {f['residual']:+.3f} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- 각본 텍스트의 구조적 메트릭만으로는 IMDB 평점 분산의 ~2%만 설명")
    lines.append("- 감독/배우 residual은 '각본 이상의 기여'를 정량화")
    lines.append("- Bonferroni 보정을 통과한 인물 = 통계적으로 검증된 '각본 초월자'")
    lines.append("- 데이터가 많을수록 통계적 검정력(power)이 증가하여 더 많은 인물이 유의미해질 수 있음")
    lines.append("")

    # Save auto-generated part to separate file (never overwrites manual analysis)
    auto_path = f'{BASE}/DIRECTOR_INFLUENCE_AUTO.md'
    with open(auto_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {auto_path} (auto-generated, safe to re-run)")
    print(f"  NOTE: Manual analysis is in {BASE}/DIRECTOR_INFLUENCE.md (not overwritten)")


def main():
    sep = "=" * 80
    print(sep)
    print("  MERGED DIRECTOR & ACTOR INFLUENCE ANALYSIS")
    print(sep)

    # 1. Load & merge
    print(f"\n{'─'*80}\n  Step 1: Load & merge data\n{'─'*80}")
    movies = load_merged()

    # 2. IMDB matching
    print(f"\n{'─'*80}\n  Step 2: Match to IMDB\n{'─'*80}")
    movies = match_titles(movies)

    # 3. Load crew/actors/names
    print(f"\n{'─'*80}\n  Step 3: Load IMDB crew & cast\n{'─'*80}")
    tc_set = {v['imdb_id'] for v in movies.values() if 'imdb_id' in v}
    print(f"  {len(tc_set)} unique IMDB IDs")
    crew, principals, names = load_crew_principals_names(tc_set)

    # 4. Scores
    print(f"\n{'─'*80}\n  Step 4: Script z-scores & residuals\n{'─'*80}")
    mstats, rm, slope, r2 = compute_scores(movies)

    # 5. Director analysis (4+ films for larger dataset)
    print(f"\n{'─'*80}\n  Step 5: Director analysis\n{'─'*80}")
    dir_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in crew:
            for nc in crew[tc]:
                dir_map[nc].append((title, v))

    min_dir = 4
    n_eligible = sum(1 for films in dir_map.values() if len(films) >= min_dir)
    print(f"  Directors with {min_dir}+ films: {n_eligible}")
    directors = analyze_with_ttest(dir_map, names, min_films=min_dir)
    print_results(directors, f"DIRECTORS ({min_dir}+ films)", n_sig_check=len(directors))

    # 6. Actor analysis (3+ films)
    print(f"\n{'─'*80}\n  Step 6: Actor analysis\n{'─'*80}")
    act_map = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in principals:
            for nc in principals[tc]:
                act_map[nc].append((title, v))

    min_act = 3
    n_eligible_a = sum(1 for films in act_map.values() if len(films) >= min_act)
    print(f"  Actors with {min_act}+ films: {n_eligible_a}")
    actors = analyze_with_ttest(act_map, names, min_films=min_act)
    print_results(actors, f"LEAD ACTORS ({min_act}+ films)", n_sig_check=len(actors))

    # 7. Summary
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    n_matched = sum(1 for v in movies.values() if 'imdb_id' in v)
    print(f"  Total movies: {len(movies)}")
    print(f"  IMDB matched: {n_matched}")
    print(f"  R² = {r2:.4f}")
    print(f"  Directors analyzed: {len(directors)}")
    print(f"  Actors analyzed: {len(actors)}")

    d_bonf = [d for d in directors if d['p'] < 0.05 / len(directors)] if directors else []
    a_bonf = [a for a in actors if a['p'] < 0.05 / len(actors)] if actors else []
    print(f"  Bonferroni directors: {len(d_bonf)}")
    print(f"  Bonferroni actors: {len(a_bonf)}")

    # 8. Save JSON + markdown
    print(f"\n{'─'*80}\n  Step 7: Save outputs\n{'─'*80}")
    meta = {
        'n_movies': len(movies),
        'n_matched': n_matched,
        'r_mean': round(rm, 4),
        'slope': round(slope, 4),
        'r2': round(r2, 4),
        'metrics': METRICS,
        'metric_directions': {m: {'dir': mstats[m]['dir'], 'top': mstats[m]['top'], 'bot': mstats[m]['bot']} for m in METRICS if m in mstats},
    }
    save_influence_json(meta, directors, actors)
    generate_markdown(meta, directors, actors, movies)

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
