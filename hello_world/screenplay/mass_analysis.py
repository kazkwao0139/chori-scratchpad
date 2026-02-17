"""
Mass screenplay analysis: download ALL IMSDB scripts + IMDB ratings.
Classify by rating tier and compare on 2D+3D axes.

Phase 1: Download all IMSDB scripts (cache to disk)
Phase 2: Get IMDB ratings via suggestion API
Phase 3: Analyze each screenplay (character diversity + narrative flow)
Phase 4: Compare tiers
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
from typing import Optional

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CACHE_DIR = Path(f"{BASE}/screenplay/_script_cache")
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT = f"{BASE}/screenplay/mass_checkpoint.json"
N_WINDOWS = 20


# --- Entropy ---

def bigram_entropy(text: str) -> float:
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
    for (a, b), count in bigrams.items():
        p_ab = count / total
        p_b_given_a = count / unigrams[a]
        H -= p_ab * math.log2(p_b_given_a)
    return H


def zlib_entropy(text: str) -> float:
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


# --- IMSDB ---

def fetch_url(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def get_all_imsdb_scripts():
    """Get list of all scripts from IMSDB alphabetical pages."""
    all_scripts = []
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0':
        url = f'https://imsdb.com/alphabetical/{letter}'
        html = fetch_url(url)
        if not html:
            continue
        links = re.findall(r'<a[^>]*href="(/Movie Scripts/[^"]+)"[^>]*>\s*([^<]+)', html)
        for href, title in links:
            # Convert href to script URL
            # /Movie Scripts/Alien Script.html -> https://imsdb.com/scripts/Alien.html
            m = re.match(r'/Movie Scripts/(.+) Script\.html', href)
            if m:
                slug = m.group(1).replace(' ', '-')
                script_url = f'https://imsdb.com/scripts/{slug}.html'
                all_scripts.append((title.strip(), script_url))
        time.sleep(0.15)
    return all_scripts


def extract_full_text(html: str) -> str:
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


def parse_dialogue(text: str) -> dict:
    lines = text.split('\n')
    characters = defaultdict(list)
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
                and re.match(r'^[A-Z][A-Z\s\.\'-]+$', clean)):
            current_char = clean
            continue
        if current_char and len(stripped) > 1:
            if not stripped.isupper():
                characters[current_char].append(stripped)
    return {char: ' '.join(lines) for char, lines in characters.items()}


# --- IMDB rating ---

def get_imdb_rating(title: str) -> Optional[float]:
    """Get IMDB rating via suggestion API."""
    clean = re.sub(r'[^\w\s]', '', title).strip()
    query = clean.replace(' ', '%20')[:50]
    url = f'https://v2.sg.media-imdb.com/suggestion/x/{query}.json'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        # Find best match (movie, not TV)
        for item in data.get('d', []):
            if item.get('qid') in ('movie', 'feature'):
                # Check title similarity
                item_title = item.get('l', '').lower()
                if title.lower()[:20] in item_title or item_title in title.lower():
                    return item.get('yr'), item.get('id')
        # Fallback: first movie result
        for item in data.get('d', []):
            if item.get('qid') in ('movie', 'feature'):
                return item.get('yr'), item.get('id')
    except Exception:
        pass
    return None, None


def get_rating_from_imdb_page(imdb_id: str) -> Optional[float]:
    """Get rating from IMDB page."""
    url = f'https://www.imdb.com/title/{imdb_id}/'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='replace')
        # Look for rating in JSON-LD
        m = re.search(r'"ratingValue"\s*:\s*"?(\d+\.?\d*)"?', html)
        if m:
            return float(m.group(1))
        m = re.search(r'"aggregateRating".*?"ratingValue".*?(\d+\.?\d*)', html, re.DOTALL)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# --- Analysis ---

def compute_flow(text, n_windows=N_WINDOWS):
    text = text.strip()
    if len(text) < n_windows * 100:
        return None
    window_size = len(text) // n_windows
    zlibs = []
    bigrams = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(text)
        chunk = text[start:end]
        bigrams.append(bigram_entropy(chunk))
        zlibs.append(zlib_entropy(chunk))
    return {'bigram_flow': bigrams, 'zlib_flow': zlibs}


def flow_metrics(flow):
    n = len(flow)
    mean = sum(flow) / n
    var = sum((v - mean) ** 2 for v in flow) / n
    std = var ** 0.5
    x_mean = (n - 1) / 2
    num = sum((i - x_mean) * (flow[i] - mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0
    diffs = [abs(flow[i+1] - flow[i]) for i in range(n - 1)]
    mean_jump = sum(diffs) / len(diffs)
    q1 = flow[:n//4]
    q4 = flow[3*n//4:]
    late_ratio = (sum(q4)/len(q4)) / (sum(q1)/len(q1)) if sum(q1) > 0 else 1.0
    # Autocorrelation
    if var > 0:
        ac = sum((flow[i] - mean) * (flow[i+1] - mean) for i in range(n-1)) / ((n-1) * var)
    else:
        ac = 0
    return {
        'mean': round(mean, 6), 'std': round(std, 6), 'var': round(var, 6),
        'slope': round(slope, 6), 'late_ratio': round(late_ratio, 4),
        'autocorr': round(ac, 4), 'mean_jump': round(mean_jump, 6),
        'range': round(max(flow) - min(flow), 6),
        'cv': round(std / mean if mean > 0 else 0, 6),
    }


def char_variance(text_full):
    dialogue = parse_dialogue(text_full)
    vals = []
    for char, text in dialogue.items():
        if len(text) < 500:
            continue
        vals.append(bigram_entropy(text))
    if len(vals) < 3:
        return None, 0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(var, 6), len(vals)


# --- Checkpoint ---

def load_checkpoint():
    try:
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'scripts': {}, 'ratings': {}, 'results': {}}


def save_checkpoint(cp):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False)


# --- Main ---

def main():
    print("=" * 70)
    print("  MASS SCREENPLAY ANALYSIS")
    print("  All IMSDB scripts + IMDB ratings")
    print("=" * 70)

    cp = load_checkpoint()

    # Phase 1: Get script list
    if 'script_list' not in cp or len(cp['script_list']) < 100:
        print("\n  Phase 1: Getting IMSDB script list...")
        scripts = get_all_imsdb_scripts()
        cp['script_list'] = scripts
        save_checkpoint(cp)
        print(f"  Found {len(scripts)} scripts")
    else:
        scripts = cp['script_list']
        print(f"\n  Phase 1: {len(scripts)} scripts in cache")

    # Phase 2: Download scripts + get ratings + analyze
    print(f"\n  Phase 2: Download, rate, and analyze...")
    total = len(scripts)
    done = len(cp['results'])
    skipped = len([k for k, v in cp['results'].items() if v is None])

    for i, (title, url) in enumerate(scripts):
        if title in cp['results']:
            continue

        print(f"  [{i+1}/{total}] {title:40s}... ", end='', flush=True)

        # Download script
        if title not in cp['scripts']:
            html = fetch_url(url)
            if not html:
                cp['scripts'][title] = None
                cp['results'][title] = None
                save_checkpoint(cp)
                print("FAIL (fetch)")
                continue
            full_text = extract_full_text(html)
            if len(full_text) < 5000:
                cp['scripts'][title] = None
                cp['results'][title] = None
                save_checkpoint(cp)
                print(f"SHORT ({len(full_text)})")
                continue
            cp['scripts'][title] = full_text
            time.sleep(0.2)
        else:
            full_text = cp['scripts'][title]
            if full_text is None:
                cp['results'][title] = None
                continue

        # Get IMDB info
        if title not in cp['ratings']:
            yr, imdb_id = get_imdb_rating(title)
            rating = None
            if imdb_id:
                rating = get_rating_from_imdb_page(imdb_id)
                time.sleep(0.2)
            cp['ratings'][title] = {'year': yr, 'imdb_id': imdb_id, 'rating': rating}
            time.sleep(0.1)

        rating_info = cp['ratings'].get(title, {})
        rating = rating_info.get('rating')

        # Analyze
        flow = compute_flow(full_text)
        if not flow:
            cp['results'][title] = None
            save_checkpoint(cp)
            print("FAIL (flow)")
            continue

        cvar, n_chars = char_variance(full_text)
        if cvar is None:
            cp['results'][title] = None
            save_checkpoint(cp)
            print("FAIL (chars)")
            continue

        zl = flow_metrics(flow['zlib_flow'])
        bg = flow_metrics(flow['bigram_flow'])

        cp['results'][title] = {
            'title': title,
            'char_var': cvar,
            'n_chars': n_chars,
            'text_len': len(full_text),
            'narr_std': zl['std'],
            'narr_slope': zl['slope'],
            'narr_cv': zl['cv'],
            'narr_autocorr': zl['autocorr'],
            'narr_late_ratio': zl['late_ratio'],
            'narr_mean': zl['mean'],
            'narr_range': zl['range'],
            'narr_mean_jump': zl['mean_jump'],
            'bg_narr_std': bg['std'],
            'bg_narr_slope': bg['slope'],
            'bg_narr_autocorr': bg['autocorr'],
            'imdb_rating': rating,
            'imdb_year': rating_info.get('year'),
            'zlib_flow': [round(v, 6) for v in flow['zlib_flow']],
        }

        save_checkpoint(cp)
        r_str = f"r={rating:.1f}" if rating else "r=?"
        print(f"OK ({r_str}, cv={cvar:.4f}, ns={zl['std']:.4f})")

    # Phase 3: Analysis
    print(f"\n{'=' * 70}")
    print("  Phase 3: ANALYSIS BY IMDB RATING TIER")
    print(f"{'=' * 70}")

    valid = [v for v in cp['results'].values() if v is not None]
    rated = [v for v in valid if v['imdb_rating'] is not None]

    print(f"\n  Total analyzed: {len(valid)}")
    print(f"  With IMDB rating: {len(rated)}")

    if not rated:
        print("  No rated movies found!")
        return

    # Tier split
    bad = [v for v in rated if v['imdb_rating'] < 5.5]
    mid = [v for v in rated if 5.5 <= v['imdb_rating'] < 7.0]
    good = [v for v in rated if 7.0 <= v['imdb_rating'] < 8.0]
    great = [v for v in rated if v['imdb_rating'] >= 8.0]

    print(f"\n  Tiers:")
    print(f"    Bad   (< 5.5): {len(bad)}")
    print(f"    Mid   (5.5-7): {len(mid)}")
    print(f"    Good  (7-8):   {len(good)}")
    print(f"    Great (8+):    {len(great)}")

    # Compare tiers
    def tier_stats(name, items):
        if not items:
            return
        cv = [v['char_var'] for v in items]
        ns = [v['narr_std'] for v in items]
        sl = [v['narr_slope'] for v in items]
        ac = [v['narr_autocorr'] for v in items]
        nm = [v['narr_mean'] for v in items]
        nj = [v['narr_mean_jump'] for v in items]
        nr = [v['narr_range'] for v in items]
        print(f"\n  {name} (n={len(items)}):")
        print(f"    char_var:    {sum(cv)/len(cv):.4f}")
        print(f"    narr_std:    {sum(ns)/len(ns):.4f}")
        print(f"    narr_slope:  {sum(sl)/len(sl):+.6f}")
        print(f"    narr_autocr: {sum(ac)/len(ac):+.4f}")
        print(f"    narr_mean:   {sum(nm)/len(nm):.4f}")
        print(f"    narr_jump:   {sum(nj)/len(nj):.4f}")
        print(f"    narr_range:  {sum(nr)/len(nr):.4f}")

    tier_stats("Bad (< 5.5)", bad)
    tier_stats("Mid (5.5-7.0)", mid)
    tier_stats("Good (7.0-8.0)", good)
    tier_stats("Great (8.0+)", great)

    # Full comparison table
    print(f"\n{'=' * 70}")
    print("  TIER COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"  {'Tier':15s} {'n':>4s} {'char_var':>10s} {'narr_std':>10s} {'slope':>10s} {'autocorr':>10s} {'mean':>10s}")
    print(f"  {'-'*70}")
    for name, items in [('Bad <5.5', bad), ('Mid 5.5-7', mid), ('Good 7-8', good), ('Great 8+', great)]:
        if not items:
            continue
        cv = sum(v['char_var'] for v in items) / len(items)
        ns = sum(v['narr_std'] for v in items) / len(items)
        sl = sum(v['narr_slope'] for v in items) / len(items)
        ac = sum(v['narr_autocorr'] for v in items) / len(items)
        nm = sum(v['narr_mean'] for v in items) / len(items)
        print(f"  {name:15s} {len(items):4d} {cv:10.4f} {ns:10.4f} {sl:+10.6f} {ac:+10.4f} {nm:10.4f}")

    # Save results
    out = {
        'total_analyzed': len(valid),
        'total_rated': len(rated),
        'tiers': {
            'bad': bad, 'mid': mid, 'good': good, 'great': great,
        },
    }
    with open(f'{BASE}/screenplay/mass_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: mass_results.json")


if __name__ == "__main__":
    main()
