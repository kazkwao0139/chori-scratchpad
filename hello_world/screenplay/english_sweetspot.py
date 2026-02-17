"""
English-only sweet spot comparison.
All 3 works in English → directly comparable with 83 IMSDB screenplays.

1. Parasite: English FYC screenplay (PDF)
2. Code Geass: English episode transcripts (wordpress)
3. Dark Knight: already English (our parsed OCR version)
"""

import re
import sys
import time
import json
import math
import zlib
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, Optional

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"


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


def calc_variance(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


# ─── 1. Parasite English (PDF) ─────────────────────

def parse_parasite_en():
    """Parse Parasite English screenplay from PDF."""
    import fitz
    pdf_path = f"{BASE}/_copyrighted/parasite_en_script.pdf"
    doc = fitz.open(pdf_path)

    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # Fix smart quotes
    full_text = full_text.replace('\u2018', "'").replace('\u2019', "'")
    full_text = full_text.replace('\u201c', '"').replace('\u201d', '"')

    # Parse screenplay format (same as IMSDB)
    lines = full_text.split('\n')
    characters = defaultdict(list)
    current_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_char = None
            continue

        # Remove parenthetical like (CONT'D), (O.S.), (V.O.)
        clean = re.sub(r'\(.*?\)', '', stripped).strip()

        # Character name: ALL CAPS, 2-30 chars, not scene heading
        if (clean and clean.isupper() and 2 <= len(clean) <= 30
                and not clean.startswith('INT') and not clean.startswith('EXT')
                and not clean.startswith('CUT') and not clean.startswith('FADE')
                and not clean.startswith('CLOSE') and not clean.startswith('ANGLE')
                and not clean.startswith('THE ')
                and not clean.startswith('LATER')
                and not clean.startswith('CONTINUOUS')
                and re.match(r'^[A-Z][A-Z\s\.\'-]+$', clean)):
            current_char = clean
            continue

        # Dialogue: indented text after character name
        if current_char and len(stripped) > 1:
            if not stripped.isupper():
                characters[current_char].append(stripped)

    # Join dialogue per character
    dialogue = {char: ' '.join(lines) for char, lines in characters.items()}

    print(f"  Parasite EN: {len(dialogue)} characters found")
    for char, text in sorted(dialogue.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {char:>20s}: {len(text):>6,d} chars")

    return dialogue


# ─── 2. Code Geass English (wordpress) ─────────────

CG_EPISODES = {
    'S1': [
        '1-the-day-a-new-demon-was-born',
        '2-his-name-is-zero',
        '3-the-false-classmate',
        '4-his-name-is-zero-2',
        '5-the-princess-and-the-witch',
        '6-the-stolen-mask',
        '7-attack-cornelia',
        '8-the-black-knights',
        '9-refrain',
        '10-guren-dances',
        '11-battle-for-narita',
        '12-the-messenger-from-kyoto',
        '13-shirley-at-gunpoint',
        '14-lelouch-vs-cornelia',
        '15-cheering-mao',
        '16-nunnally-held-hostage',
        '17-knight',
        '18-i-order-you-suzaku-kururugi',
        '19-island-of-the-gods',
        '20-battle-for-kyushu',
        '21-the-school-festival-declaration',
        '22-bloodstained-euphy',
        '23-at-least-with-sorrow',
        '24-the-collapsing-stage',
        '25-zero',
    ],
    'R2': [
        'r2-1-the-day-a-demon-awakens',
        'r2-2-plan-for-independent-japan',
        'r2-3-imprisoned-in-campus',
        'r2-4-counterattack-at-the-gallows',
        'r2-5-knight-of-rounds',
        'r2-6-surprise-attack-over-the-pacific',
        'r2-7-the-abandoned-mask',
        'r2-8-one-million-miracles',
        'r2-9-a-bride-in-the-vermillion-forbidden-city',
        'r2-10-when-shen-hu-wins-glory',
        'r2-11-power-of-passion',
        'r2-12-love-attack',
        'r2-13-assassin-from-the-past',
        'r2-14-geass-hunt',
        'r2-15-the-cs-world',
        'r2-16-united-federation-of-nations-resolution-number-one',
        'r2-17-the-taste-of-dirt',
        'r2-18-final-battle-tokyo-ii',
        'r2-19-betrayal',
        'r2-20-emperor-dismissed',
        'r2-21-the-ragnarok-connection',
        'r2-22-emperor-lelouch',
        'r2-23-schneizel-s-guise',
        'r2-24-the-grip-of-damocles',
        'r2-25-re',
    ],
}

CG_BASE_URL = "https://codegeassepisodetranscripts.wordpress.com/"


def fetch_cg_episode(slug: str) -> Optional[str]:
    url = f"{CG_BASE_URL}{slug}/"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"    FAIL: {slug} ({e})")
        return None


def parse_cg_transcript(html: str) -> Dict[str, list]:
    """Parse Code Geass transcript from wordpress HTML."""
    # Extract post content
    m = re.search(r'<div class="entry-content">(.*?)</div>', html, re.DOTALL)
    if not m:
        m = re.search(r'class="post-content">(.*?)</div>', html, re.DOTALL)
    if not m:
        # Try broader match
        m = re.search(r'<article[^>]*>(.*?)</article>', html, re.DOTALL)
    if not m:
        return {}

    content = m.group(1)
    # Remove HTML tags but preserve structure
    content = re.sub(r'<br\s*/?>', '\n', content)
    content = re.sub(r'</?p[^>]*>', '\n', content)
    content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)
    content = re.sub(r'<b>(.*?)</b>', r'**\1**', content)
    content = re.sub(r'<em>(.*?)</em>', r'', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    content = content.replace('&#8217;', "'").replace('&#8220;', '"').replace('&#8221;', '"')
    content = content.replace('&#8230;', '...')
    content = content.replace('\u2018', "'").replace('\u2019', "'")
    content = content.replace('\u201c', '"').replace('\u201d', '"')
    content = content.replace('\u2026', '...')

    # Parse **Character**: dialogue format
    chars = defaultdict(list)
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Match **Name**: dialogue or **Name**: "dialogue"
        m = re.match(r'\*\*([^*]+)\*\*\s*:\s*(.+)', line)
        if m:
            name = m.group(1).strip().upper()
            dialogue = m.group(2).strip().strip('"').strip()
            if dialogue and len(dialogue) > 1:
                chars[name].append(dialogue)

    return dict(chars)


def parse_code_geass_en():
    """Fetch and parse all 50 Code Geass episodes in English."""
    all_dialogue = defaultdict(list)
    total_eps = 0

    for season, episodes in CG_EPISODES.items():
        for slug in episodes:
            print(f"    {season} {slug}... ", end='', flush=True)
            html = fetch_cg_episode(slug)
            if not html:
                continue
            chars = parse_cg_transcript(html)
            if chars:
                for char, lines in chars.items():
                    all_dialogue[char].extend(lines)
                total_eps += 1
                total_lines = sum(len(v) for v in chars.values())
                print(f"OK ({len(chars)} chars, {total_lines} lines)")
            else:
                print("EMPTY")
            time.sleep(0.3)

    # Join dialogue per character
    dialogue = {char: ' '.join(lines) for char, lines in all_dialogue.items()}

    print(f"\n  Code Geass EN: {len(dialogue)} characters, {total_eps} episodes")
    for char, text in sorted(dialogue.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {char:>20s}: {len(text):>6,d} chars")

    return dialogue


# ─── 3. Dark Knight (already English) ──────────────

def parse_dark_knight_en():
    """Load our existing Dark Knight dialogue."""
    with open(f'{BASE}/_copyrighted/dark_knight_dialogue.json', 'r', encoding='utf-8') as f:
        raw = json.load(f)

    dialogue = {}
    for char, lines in raw.items():
        text = ' '.join(lines) if isinstance(lines, list) else lines
        dialogue[char] = text

    print(f"  Dark Knight EN: {len(dialogue)} characters")
    for char, text in sorted(dialogue.items(), key=lambda x: -len(x[1]))[:8]:
        print(f"    {char:>20s}: {len(text):>6,d} chars")

    return dialogue


# ─── Analysis ──────────────────────────────────────

def analyze_work(title: str, dialogue: dict, min_chars: int = 500):
    """Analyze one work: bigram + zlib variance."""
    bigrams = []
    zlibs = []
    char_data = []

    for char, text in dialogue.items():
        if len(text) < min_chars:
            continue
        bg = bigram_entropy(text)
        zl = zlib_entropy(text)
        bigrams.append(bg)
        zlibs.append(zl)
        char_data.append({'name': char, 'bigram': round(bg, 6),
                          'zlib': round(zl, 6), 'chars': len(text)})

    if len(bigrams) < 3:
        return None

    return {
        'title': title,
        'n_chars': len(bigrams),
        'bigram_var': round(calc_variance(bigrams), 6),
        'bigram_std': round(calc_variance(bigrams) ** 0.5, 6),
        'zlib_var': round(calc_variance(zlibs), 6),
        'zlib_std': round(calc_variance(zlibs) ** 0.5, 6),
        'characters': sorted(char_data, key=lambda x: -x['chars'])[:10],
    }


# ─── Main ──────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ENGLISH-ONLY SWEET SPOT COMPARISON")
    print("  All 3 works in English vs 83 IMSDB screenplays")
    print("=" * 70)

    # Load existing IMSDB results
    print("\n  Loading IMSDB results from sweetspot_check...")
    with open(f'{BASE}/screenplay/sweetspot_results.json', 'r', encoding='utf-8') as f:
        imsdb_data = json.load(f)
    imsdb = imsdb_data['imsdb']
    print(f"  {len(imsdb)} IMSDB screenplays loaded")

    # Parse our 3 works in English
    print(f"\n{'=' * 70}")
    print("  PARSING OUR 3 WORKS (ALL ENGLISH)")
    print(f"{'=' * 70}")

    # Parasite
    print("\n  1. Parasite (English FYC screenplay)")
    p_dialogue = parse_parasite_en()
    p_result = analyze_work("Parasite (EN)", p_dialogue)
    if p_result:
        print(f"  => bigram_var={p_result['bigram_var']:.6f}, zlib_var={p_result['zlib_var']:.6f}")

    # Code Geass
    print("\n  2. Code Geass (English transcripts)")
    cg_dialogue = parse_code_geass_en()
    cg_result = analyze_work("Code Geass (EN)", cg_dialogue)
    if cg_result:
        print(f"  => bigram_var={cg_result['bigram_var']:.6f}, zlib_var={cg_result['zlib_var']:.6f}")

    # Dark Knight
    print("\n  3. Dark Knight (already English)")
    dk_dialogue = parse_dark_knight_en()
    dk_result = analyze_work("Dark Knight (EN)", dk_dialogue)
    if dk_result:
        print(f"  => bigram_var={dk_result['bigram_var']:.6f}, zlib_var={dk_result['zlib_var']:.6f}")

    # Comparison
    our_works = [r for r in [p_result, cg_result, dk_result] if r]

    print(f"\n{'=' * 70}")
    print("  BIGRAM ENTROPY SWEET SPOT (0.024 ~ 0.039)")
    print(f"{'=' * 70}")

    for w in our_works:
        in_spot = "IN" if 0.024 <= w['bigram_var'] <= 0.039 else "OUT"
        print(f"  {w['title']:30s} bg_var={w['bigram_var']:.6f}  [{in_spot}]  ({w['n_chars']} chars)")

    # Full ranking
    print(f"\n{'=' * 70}")
    print("  FULL BIGRAM VAR RANKING")
    print(f"{'=' * 70}")

    all_ranked = []
    for r in imsdb:
        marker = "W" if r.get('is_winner') else " "
        all_ranked.append((r['bigram_var'], f"[{marker}] {r['title']}", False))
    for w in our_works:
        all_ranked.append((w['bigram_var'], f">>> {w['title']}", True))

    all_ranked.sort(key=lambda x: x[0])
    for i, (var, name, is_ours) in enumerate(all_ranked):
        flag = " <<<<" if is_ours else ""
        print(f"  {i+1:3d}. var={var:.6f}  {name}{flag}")

    # Save
    out = {
        'our_works_english': our_works,
        'sweet_spot': [0.024, 0.039],
    }
    with open(f'{BASE}/screenplay/english_sweetspot_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: english_sweetspot_results.json")


if __name__ == "__main__":
    main()
