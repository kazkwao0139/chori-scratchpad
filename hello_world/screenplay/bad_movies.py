from pathlib import Path
"""
Bad movies comparison — add notoriously bad screenplays to the 2D map.
Same analysis as narrative_flow.py + sweetspot_check.py but for bad movies.
"""

import re
import math
import sys
import time
import json
import zlib
import urllib.request
from collections import Counter, defaultdict
from typing import Optional, Dict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
N_WINDOWS = 20


# --- Entropy functions ---

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


# --- IMSDB functions ---

def fetch_script(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"    FETCH ERROR: {e}")
        return None


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


def parse_dialogue(text: str) -> Dict[str, str]:
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


# --- Analysis ---

def compute_flow(text, n_windows=N_WINDOWS):
    text = text.strip()
    if len(text) < n_windows * 100:
        return None
    window_size = len(text) // n_windows
    bigrams = []
    zlibs = []
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
    return {
        'mean': round(mean, 6), 'std': round(std, 6), 'var': round(var, 6),
        'cv': round(std / mean if mean > 0 else 0, 6),
        'slope': round(slope, 6), 'mean_jump': round(mean_jump, 6),
        'late_ratio': round(late_ratio, 4),
        'range': round(max(flow) - min(flow), 6),
    }


def char_variance(text_full):
    """Character diversity: bigram entropy variance across characters."""
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


def imsdb(slug):
    return f"https://imsdb.com/scripts/{slug}.html"


# --- Bad movies dataset ---
# Famously terrible movies available on IMSDB
BAD_MOVIES = [
    # Round 1 (worked)
    ("Twilight", imsdb("Twilight")),
    ("Fantastic Four 2005", imsdb("Fantastic-Four")),
    ("Resident Evil", imsdb("Resident-Evil")),
    ("Wild Wild West", imsdb("Wild-Wild-West")),
    # Round 2 (newly found)
    ("Catwoman", "https://imsdb.com/scripts/Catwoman.html"),
    ("Ghost Rider", "https://imsdb.com/scripts/Ghost-Rider.html"),
    ("Alone in the Dark", "https://imsdb.com/scripts/Alone-in-the-Dark.html"),
    ("Alien Resurrection", "https://imsdb.com/scripts/Alien-Resurrection.html"),
    ("Lake Placid", "https://imsdb.com/scripts/Lake-Placid.html"),
    ("Armageddon", "https://imsdb.com/scripts/Armageddon.html"),
    ("Pearl Harbor", "https://imsdb.com/scripts/Pearl-Harbor.html"),
]


def main():
    print("=" * 70)
    print("  BAD MOVIES ANALYSIS")
    print("  Same metrics as the 85 Oscar screenplays")
    print("=" * 70)

    results = []

    for title, url in BAD_MOVIES:
        print(f"  {title:35s}... ", end='', flush=True)
        html = fetch_script(url)
        if not html:
            print("FAIL (fetch)")
            continue

        full_text = extract_full_text(html)
        if not full_text or len(full_text) < 5000:
            print(f"FAIL (short: {len(full_text)} chars)")
            continue

        # Narrative flow
        flow = compute_flow(full_text)
        if not flow:
            print("FAIL (flow)")
            continue

        zl_metrics = flow_metrics(flow['zlib_flow'])
        bg_flow_metrics = flow_metrics(flow['bigram_flow'])

        # Character diversity
        cvar, n_chars = char_variance(full_text)

        if cvar is None:
            print(f"FAIL (< 3 chars with 500+ text)")
            continue

        results.append({
            'title': title,
            'is_bad': True,
            'text_len': len(full_text),
            'char_var': cvar,
            'n_chars': n_chars,
            'narr_std': zl_metrics['std'],
            'narr_slope': zl_metrics['slope'],
            'narr_cv': zl_metrics['cv'],
            'narr_late_ratio': zl_metrics['late_ratio'],
            'narr_range': zl_metrics['range'],
            'zlib_flow': [round(v, 6) for v in flow['zlib_flow']],
        })

        print(f"OK ({len(full_text):,d} chars, {n_chars} characters) "
              f"char_var={cvar:.4f} narr_std={zl_metrics['std']:.4f} slope={zl_metrics['slope']:+.6f}")
        time.sleep(0.3)

    print(f"\n  Successfully analyzed: {len(results)}/{len(BAD_MOVIES)}")

    if not results:
        print("  No bad movies could be analyzed!")
        return

    # Load existing 2D data for comparison
    with open(f'{BASE}/screenplay/screenplay_2d_data.json', 'r', encoding='utf-8') as f:
        good_data = json.load(f)

    good = good_data['matched']

    # Compare
    print(f"\n{'=' * 70}")
    print("  COMPARISON: OSCAR SCREENPLAYS vs BAD MOVIES")
    print(f"{'=' * 70}")

    g_char = [m['char_var'] for m in good]
    g_narr = [m['narr_std'] for m in good]
    b_char = [m['char_var'] for m in results]
    b_narr = [m['narr_std'] for m in results]

    print(f"\n  {'':25s} {'char_var':>10s} {'narr_std':>10s} {'slope':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Oscar avg':25s} {sum(g_char)/len(g_char):10.4f} {sum(g_narr)/len(g_narr):10.4f}")
    print(f"  {'Bad avg':25s} {sum(b_char)/len(b_char):10.4f} {sum(b_narr)/len(b_narr):10.4f}")

    b_slopes = [m['narr_slope'] for m in results]
    g_slopes = [m.get('narr_slope', 0) for m in good]
    # We need slopes from narrative_flow_results
    with open(f'{BASE}/screenplay/narrative_flow_results.json', 'r', encoding='utf-8') as f:
        flow_data = json.load(f)
    slope_lookup = {r['title']: r['zlib']['slope'] for r in flow_data['results']}
    g_slopes = [slope_lookup.get(m['title'], 0) for m in good if m['title'] in slope_lookup]
    if g_slopes:
        print(f"  {'Oscar avg slope':25s} {'':10s} {'':10s} {sum(g_slopes)/len(g_slopes):+10.6f}")
    print(f"  {'Bad avg slope':25s} {'':10s} {'':10s} {sum(b_slopes)/len(b_slopes):+10.6f}")

    # Per-movie detail
    print(f"\n  {'MOVIE':35s} {'char_var':>10s} {'narr_std':>10s} {'slope':>10s}")
    print(f"  {'-'*70}")
    for m in sorted(results, key=lambda x: x['char_var']):
        print(f"  {m['title']:35s} {m['char_var']:10.4f} {m['narr_std']:10.4f} {m['narr_slope']:+10.6f}")

    # Quadrant analysis (using Oscar medians)
    x_mid = good_data['quadrant_stats']['median_char_var']
    y_mid = good_data['quadrant_stats']['median_narr_std']

    print(f"\n  Quadrant placement (Oscar medians: char={x_mid:.4f}, narr={y_mid:.4f}):")
    for m in results:
        if m['char_var'] < x_mid and m['narr_std'] < y_mid:
            q = "Q1 (uniform+stable)"
        elif m['char_var'] >= x_mid and m['narr_std'] < y_mid:
            q = "Q2 (diverse+stable)"
        elif m['char_var'] < x_mid and m['narr_std'] >= y_mid:
            q = "Q3 (uniform+volatile)"
        else:
            q = "Q4 (diverse+volatile)"
        print(f"    {m['title']:35s} -> {q}")

    # Save
    out = {
        'bad_movies': results,
        'oscar_medians': {
            'char_var': x_mid,
            'narr_std': y_mid,
        },
    }
    with open(f'{BASE}/screenplay/bad_movies_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: bad_movies_results.json")


if __name__ == "__main__":
    main()
