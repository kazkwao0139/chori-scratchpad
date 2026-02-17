from pathlib import Path
"""
Narrative Flow Analysis — entropy over story time.

Instead of per-character dialogue entropy, we analyze the FULL screenplay
text (dialogue + stage directions) as a time series.

Split each screenplay into N equal windows → compute entropy per window
→ analyze the shape of the entropy curve.

Key question: do great screenplays have a distinctive entropy flow pattern?
"""

import re
import math
import sys
import time
import json
import zlib
import urllib.request
from collections import Counter
from typing import Optional, List

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
N_WINDOWS = 20  # split each screenplay into 20 segments (~5% each)


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


# --- Flow analysis ---

def compute_flow(text: str, n_windows: int = N_WINDOWS) -> dict:
    """Split text into windows and compute entropy for each."""
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

    return {
        'bigram_flow': bigrams,
        'zlib_flow': zlibs,
    }


def flow_metrics(flow: List[float]) -> dict:
    """Compute metrics from an entropy time series."""
    n = len(flow)
    mean = sum(flow) / n
    var = sum((v - mean) ** 2 for v in flow) / n
    std = var ** 0.5

    # Trend: simple linear regression slope
    x_mean = (n - 1) / 2
    num = sum((i - x_mean) * (flow[i] - mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0

    # First differences — how jumpy is the flow?
    diffs = [abs(flow[i+1] - flow[i]) for i in range(n - 1)]
    mean_jump = sum(diffs) / len(diffs)
    max_jump = max(diffs)
    max_jump_pos = diffs.index(max_jump) / (n - 1)  # normalized position [0,1]

    # Peak and valley positions
    peak_idx = flow.index(max(flow))
    valley_idx = flow.index(min(flow))

    # Range
    rng = max(flow) - min(flow)

    # Late-story ratio: mean of last 25% / mean of first 25%
    q1 = flow[:n//4]
    q4 = flow[3*n//4:]
    late_ratio = (sum(q4)/len(q4)) / (sum(q1)/len(q1)) if sum(q1) > 0 else 1.0

    return {
        'mean': round(mean, 6),
        'std': round(std, 6),
        'var': round(var, 6),
        'cv': round(std / mean if mean > 0 else 0, 6),
        'slope': round(slope, 6),
        'mean_jump': round(mean_jump, 6),
        'max_jump': round(max_jump, 6),
        'max_jump_pos': round(max_jump_pos, 3),
        'peak_pos': round(peak_idx / (n - 1), 3),
        'valley_pos': round(valley_idx / (n - 1), 3),
        'range': round(rng, 6),
        'late_ratio': round(late_ratio, 4),
    }


# --- IMSDB fetching (full text, no dialogue extraction) ---

def fetch_script(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def extract_full_text(html: str) -> str:
    """Extract FULL screenplay text (dialogue + stage directions)."""
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return ""
    text = m.group(1)
    text = re.sub(r'<b>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Clean up excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def imsdb(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"


# Same dataset as sweetspot_check.py
DATASET = [
    (1950, True, "Sunset Boulevard", imsdb("Sunset-Boulevard")),
    (1950, False, "All About Eve", imsdb("All-About-Eve")),
    (1953, True, "On the Waterfront", imsdb("On-the-Waterfront")),
    (1953, False, "The Barefoot Contessa", imsdb("Barefoot-Contessa,-The")),
    (1960, True, "The Apartment", imsdb("Apartment,-The")),
    (1960, False, "North by Northwest", imsdb("North-by-Northwest")),
    (1967, True, "In the Heat of the Night", imsdb("In-the-Heat-of-the-Night")),
    (1967, False, "Bonnie and Clyde", imsdb("Bonnie-and-Clyde")),
    (1967, False, "The Graduate", imsdb("Graduate,-The")),
    (1969, True, "Butch Cassidy", imsdb("Butch-Cassidy-and-the-Sundance-Kid")),
    (1969, False, "Easy Rider", imsdb("Easy-Rider")),
    (1972, True, "The Godfather", imsdb("Godfather")),
    (1972, False, "Cabaret", imsdb("Cabaret")),
    (1974, True, "Chinatown", imsdb("Chinatown")),
    (1974, False, "The Conversation", imsdb("Conversation,-The")),
    (1976, True, "Network", imsdb("Network")),
    (1976, False, "Rocky", imsdb("Rocky")),
    (1976, False, "Taxi Driver", imsdb("Taxi-Driver")),
    (1977, True, "Annie Hall", imsdb("Annie-Hall")),
    (1977, False, "Star Wars", imsdb("Star-Wars-A-New-Hope")),
    (1979, True, "Kramer vs. Kramer", imsdb("Kramer-vs-Kramer")),
    (1979, False, "Apocalypse Now", imsdb("Apocalypse-Now")),
    (1980, True, "Ordinary People", imsdb("Ordinary-People")),
    (1980, False, "Raging Bull", imsdb("Raging-Bull")),
    (1982, True, "Gandhi", imsdb("Gandhi")),
    (1982, False, "E.T.", imsdb("E-T--the-Extra-Terrestrial")),
    (1982, False, "Tootsie", imsdb("Tootsie")),
    (1984, True, "Amadeus", imsdb("Amadeus")),
    (1984, False, "The Killing Fields", imsdb("Killing-Fields,-The")),
    (1986, True, "Platoon", imsdb("Platoon")),
    (1986, False, "Top Gun", imsdb("Top-Gun")),
    (1988, True, "Rain Man", imsdb("Rain-Man")),
    (1988, False, "Die Hard", imsdb("Die-Hard")),
    (1989, True, "Dead Poets Society", imsdb("Dead-Poets-Society")),
    (1989, False, "When Harry Met Sally", imsdb("When-Harry-Met-Sally")),
    (1990, True, "Dances with Wolves", imsdb("Dances-with-Wolves")),
    (1990, False, "Goodfellas", imsdb("Goodfellas")),
    (1991, True, "Silence of the Lambs", imsdb("Silence-of-the-Lambs,-The")),
    (1991, False, "Thelma and Louise", imsdb("Thelma-and-Louise")),
    (1991, False, "JFK", imsdb("JFK")),
    (1992, True, "Unforgiven", imsdb("Unforgiven")),
    (1992, False, "A Few Good Men", imsdb("A-Few-Good-Men")),
    (1992, False, "Scent of a Woman", imsdb("Scent-of-a-Woman")),
    (1993, True, "Schindler's List", imsdb("Schindler's-List")),
    (1993, False, "The Fugitive", imsdb("Fugitive,-The")),
    (1993, False, "The Piano", imsdb("Piano,-The")),
    (1994, True, "Forrest Gump", imsdb("Forrest-Gump")),
    (1994, False, "Pulp Fiction", imsdb("Pulp-Fiction")),
    (1994, False, "The Shawshank Redemption", imsdb("Shawshank-Redemption,-The")),
    (1995, True, "Braveheart", imsdb("Braveheart")),
    (1995, False, "Sense and Sensibility", imsdb("Sense-and-Sensibility")),
    (1995, False, "Usual Suspects", imsdb("Usual-Suspects,-The")),
    (1996, True, "The English Patient", imsdb("English-Patient,-The")),
    (1996, False, "Fargo", imsdb("Fargo")),
    (1996, False, "Jerry Maguire", imsdb("Jerry-Maguire")),
    (1997, True, "Good Will Hunting", imsdb("Good-Will-Hunting")),
    (1997, False, "As Good as It Gets", imsdb("As-Good-As-It-Gets")),
    (1997, False, "Titanic", imsdb("Titanic")),
    (1998, True, "Shakespeare in Love", imsdb("Shakespeare-in-Love")),
    (1998, False, "Saving Private Ryan", imsdb("Saving-Private-Ryan")),
    (1998, False, "The Truman Show", imsdb("Truman-Show,-The")),
    (1999, True, "American Beauty", imsdb("American-Beauty")),
    (1999, False, "The Sixth Sense", imsdb("Sixth-Sense,-The")),
    (1999, False, "The Green Mile", imsdb("Green-Mile,-The")),
    (2000, True, "Almost Famous", imsdb("Almost-Famous")),
    (2000, False, "Gladiator", imsdb("Gladiator")),
    (2000, False, "Erin Brockovich", imsdb("Erin-Brockovich")),
    (2001, True, "A Beautiful Mind", imsdb("Beautiful-Mind,-A")),
    (2001, False, "Gosford Park", imsdb("Gosford-Park")),
    (2001, False, "Moulin Rouge", imsdb("Moulin-Rouge")),
    (2002, True, "The Pianist", imsdb("Pianist,-The")),
    (2002, False, "Gangs of New York", imsdb("Gangs-of-New-York")),
    (2002, False, "Minority Report", imsdb("Minority-Report")),
    (2003, True, "Lost in Translation", imsdb("Lost-in-Translation")),
    (2003, False, "Finding Nemo", imsdb("Finding-Nemo")),
    (2003, False, "Master and Commander", imsdb("Master-and-Commander")),
    (2004, True, "Eternal Sunshine", imsdb("Eternal-Sunshine-of-the-Spotless-Mind")),
    (2004, False, "Hotel Rwanda", imsdb("Hotel-Rwanda")),
    (2004, False, "The Aviator", imsdb("Aviator,-The")),
    (2005, True, "Crash", imsdb("Crash")),
    (2005, False, "Brokeback Mountain", imsdb("Brokeback-Mountain")),
    (2005, False, "Good Night and Good Luck", imsdb("Good-Night,-and-Good-Luck")),
    (2006, True, "The Departed", imsdb("Departed,-The")),
    (2006, False, "Little Miss Sunshine", imsdb("Little-Miss-Sunshine")),
    (2006, False, "Babel", imsdb("Babel")),
    (2007, True, "Juno", imsdb("Juno")),
    (2007, False, "Michael Clayton", imsdb("Michael-Clayton")),
    (2007, False, "Ratatouille", imsdb("Ratatouille")),
    (2008, True, "Slumdog Millionaire", imsdb("Slumdog-Millionaire")),
    (2008, False, "The Dark Knight", imsdb("Dark-Knight,-The")),
    (2008, False, "Milk", imsdb("Milk")),
    (2009, True, "The Hurt Locker", imsdb("Hurt-Locker,-The")),
    (2009, False, "Inglourious Basterds", imsdb("Inglourious-Basterds")),
    (2009, False, "Up", imsdb("Up")),
    (2010, True, "The King's Speech", imsdb("King's-Speech,-The")),
    (2010, False, "Black Swan", imsdb("Black-Swan")),
    (2010, False, "Inception", imsdb("Inception")),
    (2010, False, "The Fighter", imsdb("Fighter,-The")),
    (2010, False, "The Kids Are All Right", imsdb("Kids-Are-All-Right,-The")),
    (2011, True, "The Artist", imsdb("Artist,-The")),
    (2011, False, "The Descendants", imsdb("Descendants,-The")),
    (2011, False, "Moneyball", imsdb("Moneyball")),
    (2011, False, "Drive", imsdb("Drive")),
    (2012, True, "Argo", imsdb("Argo")),
    (2012, False, "Django Unchained", imsdb("Django-Unchained")),
    (2012, False, "Silver Linings Playbook", imsdb("Silver-Linings-Playbook")),
    (2012, False, "Lincoln", imsdb("Lincoln")),
    (2013, True, "12 Years a Slave", imsdb("12-Years-a-Slave")),
    (2013, False, "The Wolf of Wall Street", imsdb("Wolf-of-Wall-Street,-The")),
    (2013, False, "Her", imsdb("Her")),
    (2014, True, "Birdman", imsdb("Birdman")),
    (2014, False, "Whiplash", imsdb("Whiplash")),
    (2014, False, "Grand Budapest Hotel", imsdb("Grand-Budapest-Hotel,-The")),
    (2014, False, "Interstellar", imsdb("Interstellar")),
    (2015, True, "Spotlight", imsdb("Spotlight")),
    (2015, False, "The Big Short", imsdb("Big-Short,-The")),
    (2015, False, "The Revenant", imsdb("Revenant,-The")),
    (2015, False, "Ex Machina", imsdb("Ex-Machina")),
]


# --- Load our works (full text) ---

def load_dark_knight_full() -> str:
    with open(f'{BASE}/_copyrighted/dark_knight_script.txt', 'r', encoding='utf-8') as f:
        return f.read()


def load_parasite_en_full() -> str:
    import fitz
    pdf_path = f"{BASE}/_copyrighted/parasite_en_script.pdf"
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    full_text = full_text.replace('\u2018', "'").replace('\u2019', "'")
    full_text = full_text.replace('\u201c', '"').replace('\u201d', '"')
    return full_text


# --- Cache management ---

CACHE_FILE = f"{BASE}/screenplay/narrative_flow_cache.json"


def load_cache() -> dict:
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


# --- Main ---

def main():
    print("=" * 70)
    print("  NARRATIVE FLOW ANALYSIS")
    print("  Full screenplay text (dialogue + stage directions)")
    print(f"  {N_WINDOWS} windows per screenplay")
    print("=" * 70)

    cache = load_cache()
    results = []

    # 1. IMSDB screenplays
    print(f"\n  Phase 1: Fetching {len(DATASET)} IMSDB screenplays...")
    success = 0

    for year, is_winner, title, url in DATASET:
        marker = "W" if is_winner else " "

        # Check cache
        if title in cache:
            full_text = cache[title]
            print(f"  [{marker}] {year} {title}... CACHED ({len(full_text):,d} chars)")
        else:
            print(f"  [{marker}] {year} {title}... ", end='', flush=True)
            html = fetch_script(url)
            if not html:
                print("FAIL")
                continue
            full_text = extract_full_text(html)
            if not full_text or len(full_text) < 5000:
                print(f"SHORT ({len(full_text)} chars)")
                continue
            cache[title] = full_text
            save_cache(cache)
            print(f"OK ({len(full_text):,d} chars)")
            time.sleep(0.3)

        flow = compute_flow(full_text)
        if not flow:
            continue

        bg_metrics = flow_metrics(flow['bigram_flow'])
        zl_metrics = flow_metrics(flow['zlib_flow'])

        results.append({
            'title': title,
            'year': year,
            'is_winner': is_winner,
            'is_ours': False,
            'text_len': len(full_text),
            'bigram': bg_metrics,
            'bigram_flow': [round(v, 6) for v in flow['bigram_flow']],
            'zlib': zl_metrics,
            'zlib_flow': [round(v, 6) for v in flow['zlib_flow']],
        })
        success += 1

    print(f"\n  IMSDB: {success}/{len(DATASET)} screenplays analyzed")

    # 2. Our works
    print(f"\n{'=' * 70}")
    print("  Phase 2: Our works (full text)")
    print(f"{'=' * 70}")

    # Dark Knight
    print("\n  Dark Knight (OCR full text)...", end=' ', flush=True)
    dk_text = load_dark_knight_full()
    dk_flow = compute_flow(dk_text)
    if dk_flow:
        dk_bg = flow_metrics(dk_flow['bigram_flow'])
        dk_zl = flow_metrics(dk_flow['zlib_flow'])
        results.append({
            'title': 'The Dark Knight (ours)',
            'year': 2008,
            'is_winner': False,
            'is_ours': True,
            'text_len': len(dk_text),
            'bigram': dk_bg,
            'bigram_flow': [round(v, 6) for v in dk_flow['bigram_flow']],
            'zlib': dk_zl,
            'zlib_flow': [round(v, 6) for v in dk_flow['zlib_flow']],
        })
        print(f"OK ({len(dk_text):,d} chars)")
    else:
        print("FAIL")

    # Parasite EN
    print("  Parasite EN (PDF full text)...", end=' ', flush=True)
    try:
        p_text = load_parasite_en_full()
        p_flow = compute_flow(p_text)
        if p_flow:
            p_bg = flow_metrics(p_flow['bigram_flow'])
            p_zl = flow_metrics(p_flow['zlib_flow'])
            results.append({
                'title': 'Parasite (EN, ours)',
                'year': 2019,
                'is_winner': True,
                'is_ours': True,
                'text_len': len(p_text),
                'bigram': p_bg,
                'bigram_flow': [round(v, 6) for v in p_flow['bigram_flow']],
                'zlib': p_zl,
                'zlib_flow': [round(v, 6) for v in p_flow['zlib_flow']],
            })
            print(f"OK ({len(p_text):,d} chars)")
        else:
            print("FAIL")
    except Exception as e:
        print(f"ERROR: {e}")

    # 3. Analysis
    print(f"\n{'=' * 70}")
    print("  Phase 3: NARRATIVE FLOW COMPARISON")
    print(f"{'=' * 70}")

    imsdb_res = [r for r in results if not r['is_ours']]
    our_res = [r for r in results if r['is_ours']]
    winners = [r for r in imsdb_res if r['is_winner']]
    nominees = [r for r in imsdb_res if not r['is_winner']]

    # --- Bigram flow metrics ---
    print(f"\n  BIGRAM ENTROPY FLOW:")
    print(f"  {'':30s} {'std':>8s} {'slope':>8s} {'CV':>8s} {'peak':>6s} {'late_r':>8s} {'range':>8s}")
    print(f"  {'-'*80}")

    if winners:
        w_std = [r['bigram']['std'] for r in winners]
        w_slope = [r['bigram']['slope'] for r in winners]
        w_cv = [r['bigram']['cv'] for r in winners]
        print(f"  {'Winners (avg)':30s} {sum(w_std)/len(w_std):8.4f} {sum(w_slope)/len(w_slope):8.6f} "
              f"{sum(w_cv)/len(w_cv):8.4f}")

    if nominees:
        n_std = [r['bigram']['std'] for r in nominees]
        n_slope = [r['bigram']['slope'] for r in nominees]
        n_cv = [r['bigram']['cv'] for r in nominees]
        print(f"  {'Nominees (avg)':30s} {sum(n_std)/len(n_std):8.4f} {sum(n_slope)/len(n_slope):8.6f} "
              f"{sum(n_cv)/len(n_cv):8.4f}")

    print(f"  {'-'*80}")
    for r in our_res:
        m = r['bigram']
        print(f"  {'>>> ' + r['title']:30s} {m['std']:8.4f} {m['slope']:8.6f} "
              f"{m['cv']:8.4f} {m['peak_pos']:6.2f} {m['late_ratio']:8.4f} {m['range']:8.4f}")

    # --- Zlib flow metrics ---
    print(f"\n  ZLIB ENTROPY FLOW:")
    print(f"  {'':30s} {'std':>8s} {'slope':>8s} {'CV':>8s} {'peak':>6s} {'late_r':>8s} {'range':>8s}")
    print(f"  {'-'*80}")

    if winners:
        w_std = [r['zlib']['std'] for r in winners]
        w_slope = [r['zlib']['slope'] for r in winners]
        print(f"  {'Winners (avg)':30s} {sum(w_std)/len(w_std):8.4f} {sum(w_slope)/len(w_slope):8.6f}")

    if nominees:
        n_std = [r['zlib']['std'] for r in nominees]
        n_slope = [r['zlib']['slope'] for r in nominees]
        print(f"  {'Nominees (avg)':30s} {sum(n_std)/len(n_std):8.4f} {sum(n_slope)/len(n_slope):8.6f}")

    print(f"  {'-'*80}")
    for r in our_res:
        m = r['zlib']
        print(f"  {'>>> ' + r['title']:30s} {m['std']:8.4f} {m['slope']:8.6f} "
              f"{m['cv']:8.4f} {m['peak_pos']:6.2f} {m['late_ratio']:8.4f} {m['range']:8.4f}")

    # --- ASCII flow visualization ---
    print(f"\n{'=' * 70}")
    print("  ENTROPY FLOW CURVES (zlib, 20 windows)")
    print(f"{'=' * 70}")

    # Show our works + a few reference works
    show_list = our_res[:]
    # Add some reference: The Godfather, Pulp Fiction, Silence of the Lambs
    for ref_title in ['The Godfather', 'Pulp Fiction', 'Silence of the Lambs', 'American Beauty']:
        for r in imsdb_res:
            if r['title'] == ref_title:
                show_list.append(r)
                break

    for r in show_list:
        flow = r['zlib_flow']
        lo = min(flow)
        hi = max(flow)
        rng = hi - lo if hi > lo else 1
        label = r['title'][:35]
        marker = ">>>" if r['is_ours'] else "   "
        print(f"\n  {marker} {label} (zlib std={r['zlib']['std']:.4f}, slope={r['zlib']['slope']:.6f})")

        # Normalize to 0-40 width
        width = 40
        bars = []
        for v in flow:
            pos = int((v - lo) / rng * width)
            bars.append(pos)

        for i, pos in enumerate(bars):
            pct = (i + 1) / len(bars)
            act_label = ""
            if i == 0:
                act_label = " START"
            elif i == len(bars) // 4:
                act_label = " Act1-end"
            elif i == len(bars) // 2:
                act_label = " MIDPOINT"
            elif i == 3 * len(bars) // 4:
                act_label = " Act3-start"
            elif i == len(bars) - 1:
                act_label = " END"
            bar = '.' * pos + '#' + '.' * (width - pos)
            print(f"    {i+1:2d} |{bar}| {flow[i]:.4f}{act_label}")

    # --- Ranking by flow std ---
    print(f"\n{'=' * 70}")
    print("  RANKING by zlib flow std (narrative volatility)")
    print(f"{'=' * 70}")

    all_ranked = []
    for r in results:
        marker = ">>>" if r['is_ours'] else ("W" if r['is_winner'] else " ")
        all_ranked.append((r['zlib']['std'], f"[{marker}] {r['title']}", r['is_ours']))

    all_ranked.sort(key=lambda x: x[0])
    for i, (val, name, is_ours) in enumerate(all_ranked):
        flag = " <<<<" if is_ours else ""
        print(f"  {i+1:3d}. std={val:.4f}  {name}{flag}")

    # --- Winner vs Nominee comparison ---
    print(f"\n{'=' * 70}")
    print("  WINNER vs NOMINEE FLOW COMPARISON")
    print(f"{'=' * 70}")

    metrics_to_compare = ['std', 'slope', 'cv', 'mean_jump', 'range', 'late_ratio']
    for metric in metrics_to_compare:
        w_vals = [r['zlib'][metric] for r in winners]
        n_vals = [r['zlib'][metric] for r in nominees]
        if w_vals and n_vals:
            w_avg = sum(w_vals) / len(w_vals)
            n_avg = sum(n_vals) / len(n_vals)
            diff_pct = ((w_avg - n_avg) / n_avg * 100) if n_avg != 0 else 0
            print(f"  zlib {metric:12s}: winners={w_avg:8.4f}  nominees={n_avg:8.4f}  diff={diff_pct:+.1f}%")

    # 4. Save
    out = {
        'n_windows': N_WINDOWS,
        'results': results,
    }
    with open(f'{BASE}/screenplay/narrative_flow_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: narrative_flow_results.json")


if __name__ == "__main__":
    main()
