"""Generate influence_data.json for markdown report."""

import json, sys, gzip, math
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
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


def main():
    with open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)
    done = raw['done']

    movies = {}
    for title, v in done.items():
        if isinstance(v, dict) and v.get('rating') is not None and all(m in v for m in METRICS):
            movies[title] = v

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

    # z-scores
    sr = sorted(movies.items(), key=lambda x: x[1]['rating'], reverse=True)
    nn = len(sr)
    top = dict(sr[:int(nn * 0.15)])
    bot = dict(sr[-int(nn * 0.15):])

    mstats = {}
    for m in METRICS:
        av = [v[m] for v in movies.values() if v[m] is not None]
        tv = [v[m] for v in top.values() if v[m] is not None]
        bv = [v[m] for v in bot.values() if v[m] is not None]
        am = sum(av) / len(av)
        ast = (sum((x - am) ** 2 for x in av) / max(len(av) - 1, 1)) ** 0.5
        mstats[m] = {
            'mean': am, 'std': ast,
            'dir': +1 if sum(tv) / len(tv) > sum(bv) / len(bv) else -1,
            'top_mean': round(sum(tv) / len(tv), 4),
            'bot_mean': round(sum(bv) / len(bv), 4),
        }

    for t, v in movies.items():
        tz = 0
        cnt = 0
        for m in METRICS:
            if mstats[m]['std'] > 0 and v.get(m) is not None:
                tz += mstats[m]['dir'] * (v[m] - mstats[m]['mean']) / mstats[m]['std']
                cnt += 1
        v['script_z'] = tz / cnt if cnt > 0 else 0

    rated = [(v['rating'], v['script_z']) for v in movies.values()]
    n_all = len(rated)
    rm = sum(r for r, _ in rated) / n_all
    zm = sum(z for _, z in rated) / n_all
    cov = sum((r - rm) * (z - zm) for r, z in rated) / n_all
    vz = sum((z - zm) ** 2 for _, z in rated) / n_all
    slope = cov / vz if vz > 0 else 0
    for t, v in movies.items():
        v['residual'] = v['rating'] - (rm + slope * v['script_z'])
    ss_res = sum(v['residual'] ** 2 for v in movies.values())
    ss_tot = sum((v['rating'] - rm) ** 2 for v in movies.values())
    r2 = 1 - ss_res / ss_tot

    # crew + principals + names
    tc_set = {v['imdb_id'] for v in movies.values() if 'imdb_id' in v}
    crew = {}
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_crew.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2 and p[0] in tc_set:
                d = p[1] if p[1] != '\\N' else ''
                crew[p[0]] = d.split(',') if d else []

    principals = defaultdict(list)
    with gzip.open(f'{BASE}/_copyrighted/screenplay/_imdb_principals.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 4:
                continue
            if p[0] in tc_set and p[3] in ('actor', 'actress'):
                try:
                    principals[p[0]].append((int(p[1]), p[2]))
                except ValueError:
                    pass
    for tc in principals:
        principals[tc].sort()
        principals[tc] = [nc for _, nc in principals[tc][:2]]

    all_nc = set()
    for ds in crew.values():
        all_nc.update(ds)
    for acts in principals.values():
        all_nc.update(acts)
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

    # Actor grouping
    actor_movies = defaultdict(list)
    for title, v in movies.items():
        tc = v.get('imdb_id')
        if tc and tc in principals:
            for nc in principals[tc]:
                actor_movies[nc].append((title, v))

    output = {
        'meta': {
            'n_movies': len(movies),
            'n_matched': sum(1 for v in movies.values() if 'imdb_id' in v),
            'r_mean': round(rm, 4),
            'slope': round(slope, 4),
            'r2': round(r2, 4),
            'metrics': METRICS,
            'metric_directions': {m: {'dir': mstats[m]['dir'], 'top': mstats[m]['top_mean'], 'bot': mstats[m]['bot_mean']} for m in METRICS},
        },
        'directors': [],
        'actors': [],
    }

    # Directors 4+
    for nc, films in dir_movies.items():
        if len(films) < 4:
            continue
        name = names.get(nc, nc)
        res = [v['residual'] for _, v in films]
        rat = [v['rating'] for _, v in films]
        scr = [v['script_z'] for _, v in films]
        mr = sum(res) / len(res)
        sd = (sum((r - mr) ** 2 for r in res) / (len(res) - 1)) ** 0.5 if len(res) > 1 else 0
        se = sd / math.sqrt(len(res))
        ts = mr / se if se > 0 else 0
        p = t_pvalue(ts, len(res) - 1)
        fl = []
        for t, v in sorted(films, key=lambda x: -x[1]['residual']):
            fl.append({
                'title': t, 'rating': v['rating'],
                'script_z': round(v['script_z'], 3),
                'residual': round(v['residual'], 3)
            })
        output['directors'].append({
            'name': name, 'n': len(films),
            'avg_res': round(mr, 3), 'std': round(sd, 3),
            't': round(ts, 2), 'p': round(p, 6),
            'avg_rating': round(sum(rat) / len(rat), 1),
            'avg_script': round(sum(scr) / len(scr), 2),
            'films': fl
        })
    output['directors'].sort(key=lambda x: x['t'], reverse=True)

    # Actors 3+
    for nc, films in actor_movies.items():
        if len(films) < 3:
            continue
        name = names.get(nc, nc)
        res = [v['residual'] for _, v in films]
        rat = [v['rating'] for _, v in films]
        scr = [v['script_z'] for _, v in films]
        mr = sum(res) / len(res)
        sd = (sum((r - mr) ** 2 for r in res) / (len(res) - 1)) ** 0.5 if len(res) > 1 else 0
        se = sd / math.sqrt(len(res))
        ts = mr / se if se > 0 else 0
        p = t_pvalue(ts, len(res) - 1)
        fl = []
        for t, v in sorted(films, key=lambda x: -x[1]['residual']):
            fl.append({
                'title': t, 'rating': v['rating'],
                'script_z': round(v['script_z'], 3),
                'residual': round(v['residual'], 3)
            })
        output['actors'].append({
            'name': name, 'n': len(films),
            'avg_res': round(mr, 3), 'std': round(sd, 3),
            't': round(ts, 2), 'p': round(p, 6),
            'avg_rating': round(sum(rat) / len(rat), 1),
            'avg_script': round(sum(scr) / len(scr), 2),
            'films': fl
        })
    output['actors'].sort(key=lambda x: x['t'], reverse=True)

    with open(f'{BASE}/screenplay/influence_data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved: {len(output['directors'])} directors, {len(output['actors'])} actors")
    print(f"Meta: {output['meta']}")


if __name__ == "__main__":
    main()
