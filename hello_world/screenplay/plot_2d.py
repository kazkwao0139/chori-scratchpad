from pathlib import Path
"""
2D Scatter: Character Diversity (X) vs Narrative Consistency (Y)

X = bigram entropy variance across characters (from sweetspot_results)
Y = zlib flow std over story time (from narrative_flow_results)

Then feed to LLM for cluster analysis.
"""

import sys
import json

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)

# Load both datasets
with open(f'{BASE}/screenplay/sweetspot_results.json', 'r', encoding='utf-8') as f:
    sweet = json.load(f)

with open(f'{BASE}/screenplay/narrative_flow_results.json', 'r', encoding='utf-8') as f:
    flow = json.load(f)

# Build lookups
char_data = {}  # title -> bigram_var (character diversity)
for r in sweet['imsdb']:
    char_data[r['title']] = {
        'bigram_var': r['bigram_var'],
        'is_winner': r['is_winner'],
        'year': r['year'],
    }

# Also add our works from sweetspot
for r in sweet.get('our_works', []):
    char_data[r['title']] = {
        'bigram_var': r['bigram_var'],
        'is_winner': False,
        'year': 0,
    }

narr_data = {}  # title -> zlib flow std (narrative consistency)
for r in flow['results']:
    narr_data[r['title']] = {
        'zlib_std': r['zlib']['std'],
        'is_winner': r['is_winner'],
        'is_ours': r.get('is_ours', False),
        'year': r['year'],
    }

# Match titles present in BOTH datasets
matched = []
for title in char_data:
    # Try exact match first
    if title in narr_data:
        matched.append({
            'title': title,
            'char_var': char_data[title]['bigram_var'],
            'narr_std': narr_data[title]['zlib_std'],
            'is_winner': char_data[title]['is_winner'],
            'is_ours': narr_data[title].get('is_ours', False),
            'year': char_data[title]['year'],
        })

# Handle our works specially (title mismatch between datasets)
# sweetspot has "The Dark Knight (ours)", flow has "The Dark Knight (ours)"
# But English sweetspot had different format. Let's check manually.
# For Dark Knight: sweetspot likely used IMSDB version, flow used our OCR
# Let's add Parasite EN if available
for r in flow['results']:
    if r.get('is_ours') and r['title'] not in [m['title'] for m in matched]:
        # Try to find character data
        # Dark Knight (ours) should match
        if 'Dark Knight' in r['title']:
            # Use IMSDB Dark Knight character data if available
            for t, cd in char_data.items():
                if 'Dark Knight' in t:
                    matched.append({
                        'title': 'The Dark Knight (ours)',
                        'char_var': cd['bigram_var'],
                        'narr_std': r['zlib']['std'],
                        'is_winner': False,
                        'is_ours': True,
                        'year': 2008,
                    })
                    break

# For Parasite, use english_sweetspot_results if exists
try:
    with open(f'{BASE}/screenplay/english_sweetspot_results.json', 'r', encoding='utf-8') as f:
        en_sweet = json.load(f)
    for w in en_sweet.get('our_works_english', []):
        if 'Parasite' in w['title']:
            # Find narrative flow data
            for r in flow['results']:
                if 'Parasite' in r['title']:
                    matched.append({
                        'title': 'Parasite (EN)',
                        'char_var': w['bigram_var'],
                        'narr_std': r['zlib']['std'],
                        'is_winner': True,
                        'is_ours': True,
                        'year': 2019,
                    })
                    break
except FileNotFoundError:
    pass

# Deduplicate
seen = set()
deduped = []
for m in matched:
    if m['title'] not in seen:
        seen.add(m['title'])
        deduped.append(m)
matched = deduped

print(f"Matched {len(matched)} screenplays in both datasets")

# Separate groups
winners = [m for m in matched if m['is_winner'] and not m['is_ours']]
nominees = [m for m in matched if not m['is_winner'] and not m['is_ours']]
ours = [m for m in matched if m['is_ours']]

print(f"  Winners: {len(winners)}, Nominees: {len(nominees)}, Ours: {len(ours)}")

# --- Matplotlib scatter plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot nominees (gray)
if nominees:
    nx = [m['char_var'] for m in nominees]
    ny = [m['narr_std'] for m in nominees]
    ax.scatter(nx, ny, c='#aaaaaa', s=50, alpha=0.6, label=f'Nominees (n={len(nominees)})', zorder=2)
    for m in nominees:
        if m['title'] in ['Pulp Fiction', 'Star Wars', 'The Shawshank Redemption',
                          'Inception', 'Die Hard', 'Taxi Driver', 'Raging Bull',
                          'The Graduate', 'Finding Nemo']:
            ax.annotate(m['title'], (m['char_var'], m['narr_std']),
                       fontsize=7, alpha=0.6, ha='left', va='bottom')

# Plot winners (gold)
if winners:
    wx = [m['char_var'] for m in winners]
    wy = [m['narr_std'] for m in winners]
    ax.scatter(wx, wy, c='#FFD700', s=80, alpha=0.8, edgecolors='#B8860B',
              linewidth=1, label=f'Winners (n={len(winners)})', zorder=3)
    for m in winners:
        ax.annotate(m['title'], (m['char_var'], m['narr_std']),
                   fontsize=7, alpha=0.7, ha='left', va='bottom')

# Plot our works (red stars)
if ours:
    ox = [m['char_var'] for m in ours]
    oy = [m['narr_std'] for m in ours]
    ax.scatter(ox, oy, c='red', s=200, marker='*', alpha=1.0,
              label=f'Our Works (n={len(ours)})', zorder=5)
    for m in ours:
        ax.annotate(m['title'], (m['char_var'], m['narr_std']),
                   fontsize=9, fontweight='bold', color='red',
                   ha='left', va='bottom')

# Sweet spot zone (character axis)
ax.axvspan(0.024, 0.039, alpha=0.08, color='blue', label='Character sweet spot (0.024-0.039)')

# Winner narrative mean line
if winners:
    w_narr_mean = sum(m['narr_std'] for m in winners) / len(winners)
    ax.axhline(y=w_narr_mean, color='gold', linestyle='--', alpha=0.5,
              label=f'Winner narrative avg ({w_narr_mean:.4f})')

# Quadrant labels
all_char = [m['char_var'] for m in matched]
all_narr = [m['narr_std'] for m in matched]
x_mid = np.median(all_char)
y_mid = np.median(all_narr)

ax.axvline(x=x_mid, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=y_mid, color='gray', linestyle=':', alpha=0.3)

# Quadrant annotations
x_range = max(all_char) - min(all_char)
y_range = max(all_narr) - min(all_narr)
ax.text(min(all_char) + x_range * 0.02, max(all_narr) - y_range * 0.02,
        'Uniform voices\nVolatile narrative', fontsize=9, alpha=0.4, va='top')
ax.text(max(all_char) - x_range * 0.02, max(all_narr) - y_range * 0.02,
        'Diverse voices\nVolatile narrative', fontsize=9, alpha=0.4, va='top', ha='right')
ax.text(min(all_char) + x_range * 0.02, min(all_narr) + y_range * 0.02,
        'Uniform voices\nStable narrative\n(Bong Joon-ho zone)', fontsize=9, alpha=0.4, va='bottom')
ax.text(max(all_char) - x_range * 0.02, min(all_narr) + y_range * 0.02,
        'Diverse voices\nStable narrative\n(ideal?)', fontsize=9, alpha=0.4, va='bottom', ha='right')

ax.set_xlabel('Character Diversity (bigram entropy variance across characters)', fontsize=11)
ax.set_ylabel('Narrative Volatility (zlib entropy flow std over time)', fontsize=11)
ax.set_title('Screenplay 2D Map: Character Diversity vs Narrative Consistency\n'
             '83 IMSDB screenplays (1950-2015) + Parasite + Dark Knight',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{BASE}/screenplay/screenplay_2d_map.png', dpi=150)
print(f"\nSaved: screenplay_2d_map.png")

# --- Print data for LLM analysis ---
print(f"\n{'='*70}")
print("  DATA FOR LLM ANALYSIS")
print(f"{'='*70}")

# Quadrant counts
q1 = [m for m in matched if m['char_var'] < x_mid and m['narr_std'] < y_mid]
q2 = [m for m in matched if m['char_var'] >= x_mid and m['narr_std'] < y_mid]
q3 = [m for m in matched if m['char_var'] < x_mid and m['narr_std'] >= y_mid]
q4 = [m for m in matched if m['char_var'] >= x_mid and m['narr_std'] >= y_mid]

print(f"\n  Median split: char_var={x_mid:.4f}, narr_std={y_mid:.4f}")
print(f"\n  Q1 (uniform voice, stable narrative): {len(q1)}")
w1 = sum(1 for m in q1 if m['is_winner'])
print(f"     Winners: {w1}/{len(q1)} ({w1/len(q1)*100:.0f}%)" if q1 else "")
for m in q1:
    tag = "W" if m['is_winner'] else (" " if not m['is_ours'] else "*")
    print(f"     [{tag}] {m['title']:35s} char={m['char_var']:.4f} narr={m['narr_std']:.4f}")

print(f"\n  Q2 (diverse voice, stable narrative): {len(q2)}")
w2 = sum(1 for m in q2 if m['is_winner'])
print(f"     Winners: {w2}/{len(q2)} ({w2/len(q2)*100:.0f}%)" if q2 else "")
for m in q2:
    tag = "W" if m['is_winner'] else (" " if not m['is_ours'] else "*")
    print(f"     [{tag}] {m['title']:35s} char={m['char_var']:.4f} narr={m['narr_std']:.4f}")

print(f"\n  Q3 (uniform voice, volatile narrative): {len(q3)}")
w3 = sum(1 for m in q3 if m['is_winner'])
print(f"     Winners: {w3}/{len(q3)} ({w3/len(q3)*100:.0f}%)" if q3 else "")
for m in q3:
    tag = "W" if m['is_winner'] else (" " if not m['is_ours'] else "*")
    print(f"     [{tag}] {m['title']:35s} char={m['char_var']:.4f} narr={m['narr_std']:.4f}")

print(f"\n  Q4 (diverse voice, volatile narrative): {len(q4)}")
w4 = sum(1 for m in q4 if m['is_winner'])
print(f"     Winners: {w4}/{len(q4)} ({w4/len(q4)*100:.0f}%)" if q4 else "")
for m in q4:
    tag = "W" if m['is_winner'] else (" " if not m['is_ours'] else "*")
    print(f"     [{tag}] {m['title']:35s} char={m['char_var']:.4f} narr={m['narr_std']:.4f}")

# Winner distribution across quadrants
print(f"\n{'='*70}")
print("  WINNER DISTRIBUTION BY QUADRANT")
print(f"{'='*70}")
total_w = w1 + w2 + w3 + w4
print(f"  Q1 (uniform + stable):   {w1:2d} winners ({w1/total_w*100:.0f}%)" if total_w else "")
print(f"  Q2 (diverse + stable):   {w2:2d} winners ({w2/total_w*100:.0f}%)" if total_w else "")
print(f"  Q3 (uniform + volatile): {w3:2d} winners ({w3/total_w*100:.0f}%)" if total_w else "")
print(f"  Q4 (diverse + volatile): {w4:2d} winners ({w4/total_w*100:.0f}%)" if total_w else "")

# Correlation
n = len(matched)
x_vals = [m['char_var'] for m in matched]
y_vals = [m['narr_std'] for m in matched]
x_mean = sum(x_vals) / n
y_mean = sum(y_vals) / n
cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / n
sx = (sum((x - x_mean)**2 for x in x_vals) / n) ** 0.5
sy = (sum((y - y_mean)**2 for y in y_vals) / n) ** 0.5
corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0
print(f"\n  Correlation (char_var vs narr_std): r = {corr:.4f}")

# Save combined data
combined = {
    'matched': matched,
    'quadrant_stats': {
        'median_char_var': round(x_mid, 6),
        'median_narr_std': round(y_mid, 6),
        'Q1_uniform_stable': {'count': len(q1), 'winners': w1},
        'Q2_diverse_stable': {'count': len(q2), 'winners': w2},
        'Q3_uniform_volatile': {'count': len(q3), 'winners': w3},
        'Q4_diverse_volatile': {'count': len(q4), 'winners': w4},
    },
    'correlation': round(corr, 6),
}
with open(f'{BASE}/screenplay/screenplay_2d_data.json', 'w', encoding='utf-8') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(f"\n  Saved: screenplay_2d_data.json")
