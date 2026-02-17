from pathlib import Path
"""
Third axis candidate: Dialogue vs Direction ratio + entropy gap.
Film is a VISUAL medium. Good screenplays should drive story through
stage directions (mise-en-scene), not just dialogue.
"""

import re
import json
import sys
import math
import zlib
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


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
    for (a, b), c in bigrams.items():
        p_ab = c / total
        p_ba = c / unigrams[a]
        H -= p_ab * math.log2(p_ba)
    return H


def zlib_entropy(text):
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def split_dialogue_direction(full_text):
    """Split screenplay into dialogue vs stage directions."""
    lines = full_text.split('\n')
    dialogue_lines = []
    direction_lines = []
    current_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_char = None
            continue

        clean = re.sub(r'\(.*?\)', '', stripped).strip()

        # Character name
        if (clean.isupper() and 2 <= len(clean) <= 30
                and not clean.startswith('INT') and not clean.startswith('EXT')
                and not clean.startswith('CUT') and not clean.startswith('FADE')
                and not clean.startswith('CLOSE') and not clean.startswith('ANGLE')
                and not clean.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", clean)):
            current_char = clean
            continue

        if current_char and len(stripped) > 1 and not stripped.isupper():
            dialogue_lines.append(stripped)
        else:
            direction_lines.append(stripped)
            current_char = None

    return ' '.join(dialogue_lines), ' '.join(direction_lines)


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


def main():
    print("=" * 80)
    print("  DIALOGUE vs DIRECTION: THE VISUAL STORYTELLING AXIS")
    print("=" * 80)

    # Load all available full texts
    # 1. narrative_flow_cache (Oscar screenplays)
    with open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8') as f:
        flow_cache = json.load(f)

    # 2. mass_checkpoint (additional scripts with ratings)
    with open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8') as f:
        mass_cp = json.load(f)

    # 3. Our works
    dk_text = open(f'{BASE}/_copyrighted/dark_knight_script.txt', 'r', encoding='utf-8').read()

    import fitz
    doc = fitz.open(f'{BASE}/_copyrighted/parasite_en_script.pdf')
    p_text = ''
    for page in doc:
        p_text += page.get_text() + '\n'

    # Load Oscar metadata
    with open(f'{BASE}/screenplay/screenplay_2d_data.json', 'r', encoding='utf-8') as f:
        d2 = json.load(f)
    oscar_info = {}
    for m in d2['matched']:
        oscar_info[m['title']] = m

    # Combine all texts
    all_texts = {}
    for title, text in flow_cache.items():
        if text and len(text) > 5000:
            all_texts[title] = {'text': text, 'source': 'oscar'}

    for title, text in mass_cp.get('scripts', {}).items():
        if text and len(text) > 5000 and title not in all_texts:
            rating_info = mass_cp.get('ratings', {}).get(title, {})
            all_texts[title] = {
                'text': text, 'source': 'imsdb',
                'rating': rating_info.get('rating'),
            }

    all_texts['The Dark Knight (ours)'] = {'text': dk_text, 'source': 'ours'}
    all_texts['Parasite (EN, ours)'] = {'text': p_text, 'source': 'ours'}

    print(f"\n  Total screenplays: {len(all_texts)}")

    # Analyze each
    results = []
    for title, info in all_texts.items():
        text = info['text']
        dialogue, direction = split_dialogue_direction(text)

        if len(dialogue) < 1000 or len(direction) < 1000:
            continue

        total_len = len(dialogue) + len(direction)
        dir_ratio = len(direction) / total_len

        dial_zlib = zlib_entropy(dialogue)
        dir_zlib = zlib_entropy(direction)
        zlib_gap = dir_zlib - dial_zlib  # positive = direction more complex

        is_oscar = title in oscar_info or info['source'] == 'ours'
        is_ours = info['source'] == 'ours'
        rating = info.get('rating')

        oi = oscar_info.get(title, {})
        is_winner = oi.get('is_winner', False)

        results.append({
            'title': title,
            'dir_ratio': round(dir_ratio, 4),
            'dial_zlib': round(dial_zlib, 4),
            'dir_zlib': round(dir_zlib, 4),
            'zlib_gap': round(zlib_gap, 4),
            'is_oscar': is_oscar,
            'is_winner': is_winner,
            'is_ours': is_ours,
            'rating': rating,
            'dial_len': len(dialogue),
            'dir_len': len(direction),
        })

    print(f"  Analyzed: {len(results)}")

    # Groups
    oscar = [r for r in results if r['is_oscar'] and not r['is_ours']]
    ours = [r for r in results if r['is_ours']]
    winners = [r for r in oscar if r['is_winner']]
    nominees = [r for r in oscar if not r['is_winner']]
    rated = [r for r in results if not r['is_oscar'] and r['rating'] is not None]
    bad_rated = [r for r in rated if r['rating'] < 6.0]
    good_rated = [r for r in rated if r['rating'] >= 7.0]

    print(f"  Oscar: {len(oscar)} (W:{len(winners)}, N:{len(nominees)})")
    print(f"  Ours: {len(ours)}")
    print(f"  Rated non-Oscar: {len(rated)} (bad<6: {len(bad_rated)}, good>=7: {len(good_rated)})")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  DIALOGUE vs DIRECTION COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Group':30s} {'dir%':>7s} {'dial_zlib':>10s} {'dir_zlib':>10s} {'gap(d-d)':>10s}")
    print(f"  {'-'*62}")

    groups = [
        ('Oscar Winners', winners),
        ('Oscar Nominees', nominees),
        ('IMDB >= 7.0', good_rated),
        ('IMDB < 6.0', bad_rated),
    ]
    for name, items in groups:
        if not items:
            continue
        dr = avg([r['dir_ratio'] for r in items])
        dz = avg([r['dial_zlib'] for r in items])
        rz = avg([r['dir_zlib'] for r in items])
        gap = avg([r['zlib_gap'] for r in items])
        print(f"  {name:30s} {dr:7.1%} {dz:10.4f} {rz:10.4f} {gap:+10.4f}")

    print(f"  {'-'*62}")
    for r in ours:
        print(f"  {'>>> ' + r['title']:30s} {r['dir_ratio']:7.1%} {r['dial_zlib']:10.4f} {r['dir_zlib']:10.4f} {r['zlib_gap']:+10.4f}")

    # Ranking by dir_ratio
    print(f"\n{'='*80}")
    print(f"  DIRECTION RATIO RANKING (Oscar + Ours)")
    print(f"{'='*80}")
    ranked = sorted(oscar + ours, key=lambda r: r['dir_ratio'], reverse=True)
    for i, r in enumerate(ranked):
        if r['is_ours']:
            w = '>>>'
        elif r['is_winner']:
            w = '[W]'
        else:
            w = '[ ]'
        print(f"  {i+1:3d}. {w} {r['title']:35s} dir={r['dir_ratio']:5.1%} "
              f"gap={r['zlib_gap']:+.4f}")

    # Correlation: dir_ratio vs is_winner
    if oscar:
        w_dr = [r['dir_ratio'] for r in winners]
        n_dr = [r['dir_ratio'] for r in nominees]
        print(f"\n  Winners avg dir_ratio: {avg(w_dr):.1%}")
        print(f"  Nominees avg dir_ratio: {avg(n_dr):.1%}")
        print(f"  Diff: {(avg(w_dr) - avg(n_dr))/avg(n_dr)*100:+.1f}%")

        w_gap = [r['zlib_gap'] for r in winners]
        n_gap = [r['zlib_gap'] for r in nominees]
        print(f"\n  Winners avg zlib_gap: {avg(w_gap):+.4f}")
        print(f"  Nominees avg zlib_gap: {avg(n_gap):+.4f}")

    # Save
    with open(f'{BASE}/screenplay/dialogue_direction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: dialogue_direction_results.json")


if __name__ == "__main__":
    main()
