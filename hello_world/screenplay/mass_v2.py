"""
Mass analysis v2: download ALL IMSDB scripts, match with local IMDB dataset.
No more web scraping for ratings — use title.ratings.tsv.gz directly.
Focus: direction ratio as third axis candidate.
"""

import re
import math
import sys
import time
import json
import zlib
import gzip
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CHECKPOINT = f"{BASE}/screenplay/mass_v2_checkpoint.json"


def fetch_url(url):
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def extract_full_text(html):
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return ""
    text = m.group(1)
    text = re.sub(r'<b>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def zlib_entropy(text):
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def split_dialogue_direction(full_text):
    lines = full_text.split('\n')
    dial, dirn = [], []
    cur = None
    for line in lines:
        s = line.strip()
        if not s:
            cur = None
            continue
        c = re.sub(r'\(.*?\)', '', s).strip()
        if (c.isupper() and 2 <= len(c) <= 30
                and not c.startswith('INT') and not c.startswith('EXT')
                and not c.startswith('CUT') and not c.startswith('FADE')
                and not c.startswith('CLOSE') and not c.startswith('ANGLE')
                and not c.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", c)):
            cur = c
            continue
        if cur and len(s) > 1 and not s.isupper():
            dial.append(s)
        else:
            dirn.append(s)
            cur = None
    return ' '.join(dial), ' '.join(dirn)


def bigram_entropy(text):
    text = text.lower()
    if len(text) < 2:
        return 0.0
    bigrams = Counter()
    unigrams = Counter()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        bigrams[(a, b)] += 1
        unigrams[a] += 1
    H = 0.0
    total = sum(bigrams.values())
    for (a, b), cnt in bigrams.items():
        p_ab = cnt / total
        p_ba = cnt / unigrams[a]
        H -= p_ab * math.log2(p_ba)
    return H


def parse_dialogue_chars(text):
    lines = text.split('\n')
    characters = defaultdict(list)
    cur = None
    for line in lines:
        s = line.strip()
        if not s:
            cur = None
            continue
        c = re.sub(r'\(.*?\)', '', s).strip()
        if (c.isupper() and 2 <= len(c) <= 30
                and not c.startswith('INT') and not c.startswith('EXT')
                and not c.startswith('CUT') and not c.startswith('FADE')
                and not c.startswith('CLOSE') and not c.startswith('ANGLE')
                and not c.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", c)):
            cur = c
            continue
        if cur and len(s) > 1 and not s.isupper():
            characters[cur].append(s)
    return {ch: ' '.join(ls) for ch, ls in characters.items()}


def char_variance(text):
    dialogue = parse_dialogue_chars(text)
    vals = []
    for ch, t in dialogue.items():
        if len(t) < 500:
            continue
        vals.append(bigram_entropy(t))
    if len(vals) < 3:
        return None
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def compute_narr_std(text, n_windows=20):
    text = text.strip()
    if len(text) < n_windows * 100:
        return None
    ws = len(text) // n_windows
    vals = []
    for i in range(n_windows):
        start = i * ws
        end = start + ws if i < n_windows - 1 else len(text)
        vals.append(zlib_entropy(text[start:end]))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return var ** 0.5


def load_checkpoint():
    try:
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'done': {}}


def save_checkpoint(cp):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False)


def main():
    print("=" * 70)
    print("  MASS ANALYSIS v2 — ALL IMSDB + LOCAL IMDB RATINGS")
    print("=" * 70)

    # Phase 1: Load IMDB ratings locally
    print("\n  Loading IMDB ratings dataset...")
    ratings_db = {}
    with gzip.open(f'{BASE}/screenplay/_imdb_ratings.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    ratings_db[parts[0]] = (float(parts[1]), int(parts[2]))
                except ValueError:
                    pass

    # Build title -> (tconst, votes) index (pick highest votes)
    print("  Building title index from IMDB basics...")
    title_index = defaultdict(list)
    with gzip.open(f'{BASE}/screenplay/_imdb_basics.tsv.gz', 'rt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            tc, ttype, primary, original = parts[0], parts[1], parts[2], parts[3]
            if ttype != 'movie':
                continue
            votes = ratings_db.get(tc, (0, 0))[1]
            title_index[primary.lower()].append((tc, votes))
            if original.lower() != primary.lower():
                title_index[original.lower()].append((tc, votes))

    def get_rating(title):
        t = title.lower().strip()
        variants = [t]
        if t.endswith(', the'):
            variants.append('the ' + t[:-5])
        elif t.startswith('the '):
            variants.append(t[4:] + ', the')
        t_clean = re.sub(r'\s*\(\d{4}\)\s*$', '', t)
        if t_clean != t:
            variants.append(t_clean)

        for v in variants:
            if v in title_index:
                cands = sorted(title_index[v], key=lambda x: -x[1])
                tc, _ = cands[0]
                if tc in ratings_db:
                    r, votes = ratings_db[tc]
                    return r, votes
        return None, None

    print(f"  Ratings: {len(ratings_db):,d}, Titles: {len(title_index):,d}")

    # Phase 2: Get IMSDB script list
    print("\n  Getting IMSDB script list...")
    # Reuse from old checkpoint if available
    try:
        old_cp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
        scripts_list = old_cp.get('script_list', [])
    except Exception:
        scripts_list = []

    if len(scripts_list) < 100:
        scripts_list = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0':
            url = f'https://imsdb.com/alphabetical/{letter}'
            html = fetch_url(url)
            if not html:
                continue
            links = re.findall(r'<a[^>]*href="(/Movie Scripts/[^"]+)"[^>]*>\s*([^<]+)', html)
            for href, title in links:
                m = re.match(r'/Movie Scripts/(.+) Script\.html', href)
                if m:
                    slug = m.group(1).replace(' ', '-')
                    script_url = f'https://imsdb.com/scripts/{slug}.html'
                    scripts_list.append((title.strip(), script_url))
            time.sleep(0.1)

    print(f"  Total scripts: {len(scripts_list)}")

    # Phase 3: Download and analyze
    cp = load_checkpoint()
    # Also load previously cached scripts
    old_scripts = {}
    try:
        old_cp2 = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
        old_scripts = {k: v for k, v in old_cp2.get('scripts', {}).items() if v}
    except Exception:
        pass
    # Also narrative flow cache
    try:
        flow_cache = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
        for k, v in flow_cache.items():
            if v and k not in old_scripts:
                old_scripts[k] = v
    except Exception:
        pass

    print(f"  Pre-cached scripts: {len(old_scripts)}")
    print(f"  Already done: {len(cp['done'])}")

    total = len(scripts_list)
    batch_count = 0

    for i, (title, url) in enumerate(scripts_list):
        if title in cp['done']:
            continue

        # Check cache first
        if title in old_scripts:
            full_text = old_scripts[title]
        else:
            html = fetch_url(url)
            if not html:
                cp['done'][title] = None
                continue
            full_text = extract_full_text(html)
            if len(full_text) < 5000:
                cp['done'][title] = None
                continue
            time.sleep(0.15)

        # Analyze
        dial, dirn = split_dialogue_direction(full_text)
        if len(dial) < 500 or len(dirn) < 500:
            cp['done'][title] = None
            continue

        total_len = len(dial) + len(dirn)
        dir_ratio = len(dirn) / total_len
        dial_zlib = zlib_entropy(dial)
        dir_zlib = zlib_entropy(dirn)
        gap = dir_zlib - dial_zlib

        rating, votes = get_rating(title)

        cvar = char_variance(full_text)
        nstd = compute_narr_std(full_text)

        cp['done'][title] = {
            'dir_ratio': round(dir_ratio, 4),
            'gap': round(gap, 4),
            'dial_zlib': round(dial_zlib, 4),
            'dir_zlib': round(dir_zlib, 4),
            'rating': rating,
            'votes': votes,
            'char_var': round(cvar, 6) if cvar else None,
            'narr_std': round(nstd, 6) if nstd else None,
            'text_len': len(full_text),
        }

        batch_count += 1
        done_total = len(cp['done'])
        r_str = f"r={rating:.1f}" if rating else "r=?"
        print(f"  [{done_total}/{total}] {title:40s} dir={dir_ratio:.0%} {r_str}", flush=True)

        if batch_count % 20 == 0:
            save_checkpoint(cp)

    save_checkpoint(cp)

    # Phase 4: Analysis
    print(f"\n{'=' * 70}")
    print("  FINAL ANALYSIS")
    print(f"{'=' * 70}")

    valid = [v for v in cp['done'].values() if v is not None]
    rated = [v for v in valid if v.get('rating') and v.get('votes', 0) >= 1000]

    print(f"  Total analyzed: {len(valid)}")
    print(f"  With reliable rating (votes>=1000): {len(rated)}")

    if len(rated) < 10:
        print("  Not enough rated movies!")
        save_checkpoint(cp)
        return

    # Pearson correlation
    rs = [d['rating'] for d in rated]
    dr = [d['dir_ratio'] for d in rated]

    n = len(rs)
    mx, my = sum(rs)/n, sum(dr)/n
    sx = (sum((x-mx)**2 for x in rs)/n)**0.5
    sy = (sum((y-my)**2 for y in dr)/n)**0.5
    if sx > 0 and sy > 0:
        r = sum((x-mx)*(y-my) for x, y in zip(rs, dr)) / (n*sx*sy)
        t = r * math.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 999
    else:
        r, t = 0, 0

    print(f"\n  Pearson r(rating, dir_ratio) = {r:+.4f}  t={t:.2f}  n={n}")

    # Non-linear: check if DEVIATION from 55-60% correlates with rating
    optimal = 0.575  # midpoint guess
    dev = [abs(d['dir_ratio'] - optimal) for d in rated]
    mx2, my2 = sum(rs)/n, sum(dev)/n
    sx2 = (sum((x-mx2)**2 for x in rs)/n)**0.5
    sy2 = (sum((y-my2)**2 for y in dev)/n)**0.5
    if sx2 > 0 and sy2 > 0:
        r2 = sum((x-mx2)*(y-my2) for x, y in zip(rs, dev)) / (n*sx2*sy2)
        t2 = r2 * math.sqrt((n-2)/(1-r2**2)) if abs(r2) < 1 else 999
    else:
        r2, t2 = 0, 0

    print(f"  Pearson r(rating, |dir_ratio - 57.5%|) = {r2:+.4f}  t={t2:.2f}")
    print(f"  (negative = high rating correlates with LESS deviation = closer to optimal)")

    # Tier breakdown
    print(f"\n  {'Tier':>10s} {'n':>4s} {'avg_r':>7s} {'dir%':>7s} {'|dev|':>7s} {'gap':>8s}")
    print(f"  {'-'*50}")
    for lo, hi, name in [(0,4,'< 4'), (4,5.5,'4-5.5'), (5.5,6.5,'5.5-6.5'),
                          (6.5,7.5,'6.5-7.5'), (7.5,8.5,'7.5-8.5'), (8.5,10,'8.5+')]:
        items = [d for d in rated if lo <= d['rating'] < hi]
        if not items:
            continue
        ar = sum(d['rating'] for d in items) / len(items)
        adr = sum(d['dir_ratio'] for d in items) / len(items)
        adv = sum(abs(d['dir_ratio'] - optimal) for d in items) / len(items)
        ag = sum(d['gap'] for d in items) / len(items)
        print(f"  {name:>10s} {len(items):4d} {ar:7.1f} {adr:7.1%} {adv:7.1%} {ag:+8.4f}")

    # Save full results
    with open(f'{BASE}/screenplay/mass_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_analyzed': len(valid),
            'total_rated': len(rated),
            'correlation_linear': round(r, 4),
            'correlation_deviation': round(r2, 4),
            'rated': rated,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: mass_v2_results.json")


if __name__ == "__main__":
    main()
