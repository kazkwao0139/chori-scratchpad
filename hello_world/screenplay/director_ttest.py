from pathlib import Path
"""
감독 영향력 통계적 유의성 검증.
H0: 감독의 평균 잔차 = 0 (각본 이상의 기여 없음)
One-sample t-test, two-tailed.
"""

import json, sys, gzip, math
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
METRICS = ['dir_ratio', 'dial_zlib', 'dir_zlib', 'char_var', 'narr_std']


def normalize(t):
    t = t.lower().strip()
    for s in [', the', ', a', ', an']:
        if t.endswith(s):
            t = s.strip(', ') + ' ' + t[:-len(s)]
    return t


def t_pvalue(t_stat, df):
    """Numerical integration for two-tailed p-value of t-distribution."""
    abs_t = abs(t_stat)
    upper = abs_t + 80
    n_steps = 20000

    def pdf(x):
        return (1 + x * x / df) ** (-(df + 1) / 2)

    # Tail integral: abs_t to upper
    h = (upper - abs_t) / n_steps
    total = pdf(abs_t) + pdf(upper)
    for i in range(1, n_steps):
        xi = abs_t + i * h
        total += (4 if i % 2 else 2) * pdf(xi)
    tail = total * h / 3

    # Full integral: -upper to upper
    h2 = (2 * upper) / n_steps
    total2 = pdf(-upper) + pdf(upper)
    for i in range(1, n_steps):
        xi = -upper + i * h2
        total2 += (4 if i % 2 else 2) * pdf(xi)
    full = total2 * h2 / 3

    return min(2 * tail / full, 1.0) if full > 0 else 1.0


def main():
    # Load movies
    with open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
    done = raw['done']

    movies = {}
    for title, v in done.items():
        if not isinstance(v, dict) or v.get('rating') is None:
            continue
        if not all(m in v for m in METRICS):
            continue
        movies[title] = v

    # Match titles
    ratings_db = {}
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_ratings.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 3:
                try:
                    ratings_db[p[0]] = int(p[2])
                except ValueError:
                    pass

    title_lookup = {normalize(t): t for t in movies}
    candidates = defaultdict(list)
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_basics.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 9 or p[1] != 'movie':
                continue
            for tf in [p[2], p[3]]:
                n = normalize(tf)
                if n in title_lookup:
                    candidates[title_lookup[n]].append((p[0], ratings_db.get(p[0], 0)))

    for title, cands in candidates.items():
        if title in movies:
            movies[title]['imdb_id'] = max(cands, key=lambda x: x[1])[0]

    # Script z-scores
    sorted_r = sorted(movies.items(), key=lambda x: x[1]['rating'], reverse=True)
    nn = len(sorted_r)
    top = dict(sorted_r[:int(nn * 0.15)])
    bot = dict(sorted_r[-int(nn * 0.15):])

    stats = {}
    for m in METRICS:
        av = [v[m] for v in movies.values() if v[m] is not None]
        tv = [v[m] for v in top.values() if v[m] is not None]
        bv = [v[m] for v in bot.values() if v[m] is not None]
        am = sum(av) / len(av)
        ast = (sum((x - am) ** 2 for x in av) / max(len(av) - 1, 1)) ** 0.5
        stats[m] = {
            'mean': am, 'std': ast,
            'dir': +1 if sum(tv) / len(tv) > sum(bv) / len(bv) else -1
        }

    for t, v in movies.items():
        tz = 0
        cnt = 0
        for m in METRICS:
            if stats[m]['std'] > 0 and v.get(m) is not None:
                tz += stats[m]['dir'] * (v[m] - stats[m]['mean']) / stats[m]['std']
                cnt += 1
        v['script_z'] = tz / cnt if cnt > 0 else 0

    rated = [(v['rating'], v['script_z']) for v in movies.values()]
    n_all = len(rated)
    r_mean = sum(r for r, _ in rated) / n_all
    z_mean = sum(z for _, z in rated) / n_all
    cov = sum((r - r_mean) * (z - z_mean) for r, z in rated) / n_all
    vz = sum((z - z_mean) ** 2 for _, z in rated) / n_all
    slope = cov / vz if vz > 0 else 0

    for t, v in movies.items():
        v['residual'] = v['rating'] - (r_mean + slope * v['script_z'])

    # Load crew + names
    tc_set = {v['imdb_id'] for v in movies.values() if 'imdb_id' in v}
    crew = {}
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_crew.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in tc_set:
                d = p[1] if p[1] != '\\N' else ''
                crew[p[0]] = d.split(',') if d else []

    all_nc = set()
    for ds in crew.values():
        all_nc.update(ds)

    names = {}
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_names.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in all_nc:
                names[p[0]] = p[1]

    # Director grouping
    dir_movies = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in crew:
            for nc in crew[tc]:
                dir_movies[nc].append((title, v))

    # t-test for directors with 5+ films
    sep = "=" * 82
    print(sep)
    print("  DIRECTOR INFLUENCE — STATISTICAL SIGNIFICANCE TEST")
    print("  H0: 감독의 평균 잔차 = 0 (각본 이상의 기여 없음)")
    print("  One-sample t-test, two-tailed")
    print(sep)
    print()

    results = []
    for nc, films in dir_movies.items():
        if len(films) < 4:
            continue
        name = names.get(nc, nc)
        residuals = [v['residual'] for _, v in films]
        n_f = len(residuals)
        mean_r = sum(residuals) / n_f
        std_r = (sum((r - mean_r) ** 2 for r in residuals) / (n_f - 1)) ** 0.5 if n_f > 1 else 0
        se = std_r / math.sqrt(n_f) if n_f > 0 else 0
        t_stat = mean_r / se if se > 0 else 0
        df = n_f - 1
        p = t_pvalue(t_stat, df)
        results.append((name, n_f, mean_r, std_r, t_stat, p, df))

    # Sort by t-stat descending
    results.sort(key=lambda x: x[4], reverse=True)

    print(f"  {'Director':<28s} {'N':>3s} {'AvgRes':>7s} {'Std':>6s} {'t':>7s} {'df':>3s} {'p-value':>10s}  Sig")
    print(f"  {'─' * 78}")

    for name, n_f, mean_r, std_r, t_stat, p, df in results:
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '** '
        elif p < 0.05:
            sig = '*  '
        elif p < 0.1:
            sig = '†  '
        else:
            sig = '   '
        print(f"  {name:<28s} {n_f:3d} {mean_r:+7.3f} {std_r:6.3f} {t_stat:+7.2f} {df:3d} {p:10.6f}  {sig}")

    print()
    print(f"  *** p < 0.001  ** p < 0.01  * p < 0.05  † p < 0.1")
    print()

    sig_levels = {
        'p < 0.001 (positive)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.001 and m > 0),
        'p < 0.01  (positive)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.01 and m > 0),
        'p < 0.05  (positive)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.05 and m > 0),
        'p < 0.001 (negative)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.001 and m < 0),
        'p < 0.01  (negative)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.01 and m < 0),
        'p < 0.05  (negative)': sum(1 for _, _, m, _, _, p, _ in results if p < 0.05 and m < 0),
    }

    print(f"  SUMMARY:")
    for label, count in sig_levels.items():
        print(f"    {label}: {count}명")

    # Highlight: the "statistically verified masters"
    print(f"\n{sep}")
    print("  STATISTICALLY VERIFIED MASTERS (p < 0.05, positive, 5+ films)")
    print(sep)
    masters = [(name, n_f, mean_r, std_r, t_stat, p, df)
               for name, n_f, mean_r, std_r, t_stat, p, df in results
               if p < 0.05 and mean_r > 0 and n_f >= 5]

    if masters:
        print(f"\n  {'Director':<28s} {'N':>3s} {'AvgRes':>7s} {'t':>7s} {'p':>10s}")
        print(f"  {'─' * 60}")
        for name, n_f, mean_r, std_r, t_stat, p, df in masters:
            stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            print(f"  {name:<28s} {n_f:3d} {mean_r:+7.3f} {t_stat:+7.2f} {p:10.6f} {stars}")
    print()

    # Also check: Bonferroni correction
    n_tests = len(results)
    bonf = 0.05 / n_tests
    bonf_masters = [(name, n_f, mean_r, std_r, t_stat, p, df)
                    for name, n_f, mean_r, std_r, t_stat, p, df in results
                    if p < bonf and mean_r > 0 and n_f >= 5]

    print(f"  Bonferroni correction: α = 0.05 / {n_tests} = {bonf:.6f}")
    if bonf_masters:
        print(f"  Survives Bonferroni (5+ films):")
        for name, n_f, mean_r, _, t_stat, p, _ in bonf_masters:
            print(f"    {name} (N={n_f}, t={t_stat:+.2f}, p={p:.6f})")
    else:
        print(f"  None survive Bonferroni at 5+ films.")

    # Relax to 4+ films
    bonf_masters4 = [(name, n_f, mean_r, std_r, t_stat, p, df)
                     for name, n_f, mean_r, std_r, t_stat, p, df in results
                     if p < bonf and mean_r > 0]
    if bonf_masters4 and len(bonf_masters4) > len(bonf_masters):
        print(f"  Survives Bonferroni (4+ films):")
        for name, n_f, mean_r, _, t_stat, p, _ in bonf_masters4:
            print(f"    {name} (N={n_f}, t={t_stat:+.2f}, p={p:.6f})")


if __name__ == "__main__":
    main()
