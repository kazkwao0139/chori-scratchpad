from pathlib import Path
"""
Sweet spot check: 83 IMSDB screenplays + our 3 works
Both bigram entropy AND zlib entropy per character → variance comparison.

The Dark Knight is already in the 83-screenplay IMSDB dataset (2008 nominee).
Parasite (Korean) and Code Geass (Japanese) are computed from our dialogue JSONs.
"""

import re
import math
import sys
import time
import json
import zlib
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, Optional

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


# ─── Entropy functions ─────────────────────────────

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


# ─── IMSDB ──────────────────────────────────────────

def fetch_script(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def extract_text(html: str) -> str:
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return ""
    text = m.group(1)
    text = re.sub(r'<b>', '\n<b>', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text


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


def analyze_screenplay(title: str, url: str, min_chars: int = 500) -> Optional[dict]:
    """Fetch and analyze a single screenplay from IMSDB."""
    html = fetch_script(url)
    if not html:
        return None
    text = extract_text(html)
    if not text or len(text) < 500:
        return None
    dialogue = parse_dialogue(text)

    chars_bigram = []
    chars_zlib = []
    for char, text in dialogue.items():
        if len(text) < min_chars:
            continue
        chars_bigram.append(bigram_entropy(text))
        chars_zlib.append(zlib_entropy(text))

    if len(chars_bigram) < 3:
        return None

    bg_mean = sum(chars_bigram) / len(chars_bigram)
    bg_var = sum((e - bg_mean) ** 2 for e in chars_bigram) / len(chars_bigram)

    zl_mean = sum(chars_zlib) / len(chars_zlib)
    zl_var = sum((e - zl_mean) ** 2 for e in chars_zlib) / len(chars_zlib)

    return {
        'title': title,
        'n_chars': len(chars_bigram),
        'bigram_var': round(bg_var, 6),
        'bigram_std': round(bg_var ** 0.5, 6),
        'zlib_var': round(zl_var, 6),
        'zlib_std': round(zl_var ** 0.5, 6),
    }


# ─── Our 3 works ──────────────────────────────────────

def analyze_our_work(title: str, dialogue: dict, min_lines: int = 20) -> dict:
    """Analyze one of our works using the same metrics."""
    chars_bigram = []
    chars_zlib = []
    char_names = []

    for char, lines in dialogue.items():
        if isinstance(lines, list):
            text = ' '.join(lines)
        else:
            text = lines
        if len(text) < 500:
            continue
        chars_bigram.append(bigram_entropy(text))
        chars_zlib.append(zlib_entropy(text))
        char_names.append(char)

    if len(chars_bigram) < 3:
        return None

    bg_mean = sum(chars_bigram) / len(chars_bigram)
    bg_var = sum((e - bg_mean) ** 2 for e in chars_bigram) / len(chars_bigram)

    zl_mean = sum(chars_zlib) / len(chars_zlib)
    zl_var = sum((e - zl_mean) ** 2 for e in chars_zlib) / len(chars_zlib)

    return {
        'title': title,
        'n_chars': len(chars_bigram),
        'bigram_var': round(bg_var, 6),
        'bigram_std': round(bg_var ** 0.5, 6),
        'zlib_var': round(zl_var, 6),
        'zlib_std': round(zl_var ** 0.5, 6),
        'per_char_bigram': {n: round(b, 6) for n, b in zip(char_names, chars_bigram)},
        'per_char_zlib': {n: round(z, 6) for n, z in zip(char_names, chars_zlib)},
    }


# ─── IMSDB Dataset (same as screenplay_entropy_full.py) ────

def imsdb(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"

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


# ─── Main ────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SWEET SPOT CHECK — 83 IMSDB screenplays + 3 works")
    print("  Bigram entropy variance + zlib entropy variance")
    print("=" * 70)

    # 1. Fetch and analyze IMSDB screenplays
    imsdb_results = []
    success = 0

    for year, is_winner, title, url in DATASET:
        marker = "W" if is_winner else " "
        print(f"  [{marker}] {year} {title}... ", end='', flush=True)
        a = analyze_screenplay(title, url)
        if a:
            a['year'] = year
            a['is_winner'] = is_winner
            imsdb_results.append(a)
            print(f"OK ({a['n_chars']} chars, bg_var={a['bigram_var']:.6f}, zl_var={a['zlib_var']:.6f})")
            success += 1
        else:
            print("SKIP")
        time.sleep(0.3)

    print(f"\n  IMSDB: {success}/{len(DATASET)} screenplays analyzed")

    # 2. Analyze our 3 works
    print(f"\n{'=' * 70}")
    print("  OUR 3 WORKS")
    print(f"{'=' * 70}")

    our_works = []

    # Dark Knight
    with open(f'{BASE}/_copyrighted/dark_knight_dialogue.json', 'r', encoding='utf-8') as f:
        dk_dialogue = json.load(f)
    dk = analyze_our_work("The Dark Knight (ours)", dk_dialogue)
    if dk:
        our_works.append(dk)
        print(f"  Dark Knight:  {dk['n_chars']} chars, bg_var={dk['bigram_var']:.6f}, zl_var={dk['zlib_var']:.6f}")

    # Parasite
    with open(f'{BASE}/_copyrighted/parasite_dialogue.json', 'r', encoding='utf-8') as f:
        p_dialogue = json.load(f)
    p = analyze_our_work("Parasite (ours)", p_dialogue)
    if p:
        our_works.append(p)
        print(f"  Parasite:     {p['n_chars']} chars, bg_var={p['bigram_var']:.6f}, zl_var={p['zlib_var']:.6f}")

    # Code Geass
    with open(f'{BASE}/_copyrighted/code_geass_dialogue_ja.json', 'r', encoding='utf-8') as f:
        cg_dialogue = json.load(f)
    cg = analyze_our_work("Code Geass (ours)", cg_dialogue)
    if cg:
        our_works.append(cg)
        print(f"  Code Geass:   {cg['n_chars']} chars, bg_var={cg['bigram_var']:.6f}, zl_var={cg['zlib_var']:.6f}")

    # 3. Sweet spot analysis
    print(f"\n{'=' * 70}")
    print("  SWEET SPOT COMPARISON")
    print(f"{'=' * 70}")

    winners = [r for r in imsdb_results if r['is_winner']]
    nominees = [r for r in imsdb_results if not r['is_winner']]

    if winners and nominees:
        # Bigram sweet spot (original)
        w_bg = [r['bigram_var'] for r in winners]
        n_bg = [r['bigram_var'] for r in nominees]
        all_bg = w_bg + n_bg

        print(f"\n  BIGRAM ENTROPY VARIANCE:")
        print(f"    Winners  (n={len(w_bg):2d}): mean={sum(w_bg)/len(w_bg):.6f}, "
              f"range=[{min(w_bg):.6f}, {max(w_bg):.6f}]")
        print(f"    Nominees (n={len(n_bg):2d}): mean={sum(n_bg)/len(n_bg):.6f}, "
              f"range=[{min(n_bg):.6f}, {max(n_bg):.6f}]")
        print(f"    Sweet spot: 0.024 ~ 0.039")

        for w in our_works:
            in_spot = "IN" if 0.024 <= w['bigram_var'] <= 0.039 else "OUT"
            print(f"    --> {w['title']:30s} bg_var={w['bigram_var']:.6f}  [{in_spot}]")

        # zlib sweet spot (new)
        w_zl = [r['zlib_var'] for r in winners]
        n_zl = [r['zlib_var'] for r in nominees]

        # Find zlib sweet spot using same percentile logic
        all_zl = sorted(w_zl + n_zl)
        p25 = all_zl[len(all_zl) // 4]
        p75 = all_zl[3 * len(all_zl) // 4]

        print(f"\n  ZLIB ENTROPY VARIANCE:")
        print(f"    Winners  (n={len(w_zl):2d}): mean={sum(w_zl)/len(w_zl):.6f}, "
              f"range=[{min(w_zl):.6f}, {max(w_zl):.6f}]")
        print(f"    Nominees (n={len(n_zl):2d}): mean={sum(n_zl)/len(n_zl):.6f}, "
              f"range=[{min(n_zl):.6f}, {max(n_zl):.6f}]")
        print(f"    IQR: [{p25:.6f}, {p75:.6f}]")

        for w in our_works:
            in_iqr = "IN" if p25 <= w['zlib_var'] <= p75 else "OUT"
            print(f"    --> {w['title']:30s} zl_var={w['zlib_var']:.6f}  [{in_iqr}]")

    # 4. Detailed ranking — where do our works fall?
    print(f"\n{'=' * 70}")
    print("  BIGRAM VAR RANKING (all screenplays + our works)")
    print(f"{'=' * 70}")

    all_ranked = []
    for r in imsdb_results:
        marker = "W" if r['is_winner'] else " "
        all_ranked.append((r['bigram_var'], f"[{marker}] {r['title']}", False))
    for w in our_works:
        all_ranked.append((w['bigram_var'], f">>> {w['title']}", True))

    all_ranked.sort(key=lambda x: x[0])
    for i, (var, name, is_ours) in enumerate(all_ranked):
        flag = " <<<<" if is_ours else ""
        print(f"  {i+1:3d}. var={var:.6f}  {name}{flag}")

    # 5. Save results
    out = {
        'imsdb': [{k: v for k, v in r.items()} for r in imsdb_results],
        'our_works': our_works,
    }
    with open(f'{BASE}/screenplay/sweetspot_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: sweetspot_results.json")


if __name__ == "__main__":
    main()
