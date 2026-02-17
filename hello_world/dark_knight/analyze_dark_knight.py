"""
The Dark Knight — zlib entropy + cosine similarity analysis
"Is the Joker truly chaotic? Does Two-Face transform? Does Batman have dual persona?"

Pipeline identical to Parasite / Code Geass analysis.
"""

import json
import zlib
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Font setup ──
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world/dark_knight"

# ── Load data ──
with open(f'{BASE}/dark_knight_dialogue.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

with open(f'{BASE}/dark_knight_dialogue_detail.json', 'r', encoding='utf-8') as f:
    detail = json.load(f)

# ══════════════════════════════════════
# Main characters
# ══════════════════════════════════════
# Keep BATMAN and WAYNE separate to test dual persona
MAIN_CHARS = ['THE JOKER', 'DENT', 'BATMAN', 'WAYNE', 'GORDON', 'RACHEL', 'ALFRED', 'FOX']

CHAR_COLORS = {
    'THE JOKER': '#9400D3',   # dark violet (chaos)
    'DENT':      '#FF8C00',   # dark orange (two-face)
    'BATMAN':    '#1C1C1C',   # near-black
    'WAYNE':     '#4169E1',   # royal blue
    'GORDON':    '#8B4513',   # saddle brown
    'RACHEL':    '#DC143C',   # crimson
    'ALFRED':    '#2E8B57',   # sea green
    'FOX':       '#708090',   # slate gray
}

CHAR_LABELS = {
    'THE JOKER': 'Joker',
    'DENT':      'Dent',
    'BATMAN':    'Batman',
    'WAYNE':     'Wayne',
    'GORDON':    'Gordon',
    'RACHEL':    'Rachel',
    'ALFRED':    'Alfred',
    'FOX':       'Fox',
}

# ── Analysis functions ──
WINDOW = 12
STRIDE = 4


def text_entropy(text: str) -> float:
    """zlib compression ratio = surface entropy"""
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def char_frequency(text: str) -> dict:
    """Character frequency vector"""
    freq = {}
    for ch in text.lower():
        if ch.strip():
            freq[ch] = freq.get(ch, 0) + 1
    return freq


def cosine_sim(v1: dict, v2: dict) -> float:
    """Cosine similarity between two frequency vectors"""
    all_keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in all_keys)
    m1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    m2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


# ══════════════════════════════════════
# 1. Per-character analysis
# ══════════════════════════════════════
print("=" * 70)
print("The Dark Knight — zlib Entropy Analysis")
print("=" * 70)

results = {}

for char in MAIN_CHARS:
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW:
        print(f"  {char}: insufficient data ({len(lines)} lines)")
        continue

    full_text = '\n'.join(lines)
    total_entropy = text_entropy(full_text)

    # Sliding window entropy
    entropies = []
    positions = []
    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        e = text_entropy(chunk)
        pos = (i + WINDOW / 2) / len(lines)
        entropies.append(e)
        positions.append(pos)

    # Sliding window cosine (compare adjacent windows)
    cosines = []
    cos_positions = []
    prev_vec = None
    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        vec = char_frequency(chunk)
        if prev_vec is not None:
            sim = cosine_sim(prev_vec, vec)
            cosines.append(sim)
            cos_positions.append((i + WINDOW / 2) / len(lines))
        prev_vec = vec

    ent_arr = np.array(entropies)
    cos_arr = np.array(cosines) if cosines else np.array([0])

    results[char] = {
        'total_lines': len(lines),
        'total_entropy': round(total_entropy, 6),
        'entropy_mean': round(float(np.mean(ent_arr)), 6),
        'entropy_std': round(float(np.std(ent_arr)), 6),
        'cosine_mean': round(float(np.mean(cos_arr)), 6),
        'cosine_std': round(float(np.std(cos_arr)), 6),
        'timeline': {
            'positions': [round(p, 4) for p in positions],
            'entropy': [round(float(e), 6) for e in entropies],
        },
        'cosine_timeline': {
            'positions': [round(p, 4) for p in cos_positions],
            'cosine': [round(float(c), 6) for c in cosines],
        },
    }

    print(f"\n  {CHAR_LABELS.get(char, char)} ({len(lines)} lines)")
    print(f"    Total zlib:  {total_entropy:.4f}")
    print(f"    Window mean: {np.mean(ent_arr):.4f} +/- {np.std(ent_arr):.4f}")
    print(f"    Cosine mean: {np.mean(cos_arr):.4f} +/- {np.std(cos_arr):.4f}")


# ══════════════════════════════════════
# 2. Harvey Dent: Before vs After (Two-Face transition)
# ══════════════════════════════════════
print("\n" + "=" * 70)
print("Harvey Dent — Before vs After Rachel's death")
print("=" * 70)

dent_lines = dialogue.get('DENT', [])
if dent_lines:
    # Find approximate split point: Rachel dies around ~60-65% of script
    # In the script, the hospital scene where Dent becomes Two-Face
    # We'll split at the halfway point of Dent's dialogue, which roughly
    # corresponds to pre/post transformation
    dent_detail = [d for d in detail if d['character'] == 'DENT']

    # Find the line_num where "The coin" or Two-Face behavior starts
    # Look for keywords indicating transformation
    split_idx = len(dent_lines) // 2  # default: midpoint

    # Try to find a better split: look for "fair" or "chance" which
    # Two-Face uses obsessively
    for idx, line in enumerate(dent_lines):
        if 'coin' in line.lower() and idx > len(dent_lines) * 0.4:
            split_idx = idx
            break

    pre_lines = dent_lines[:split_idx]
    post_lines = dent_lines[split_idx:]

    if len(pre_lines) >= 10 and len(post_lines) >= 10:
        pre_text = '\n'.join(pre_lines)
        post_text = '\n'.join(post_lines)

        pre_ent = text_entropy(pre_text)
        post_ent = text_entropy(post_text)

        # Window analysis for each half
        pre_ents = []
        for i in range(0, len(pre_lines) - WINDOW + 1, STRIDE):
            chunk = '\n'.join(pre_lines[i:i + WINDOW])
            pre_ents.append(text_entropy(chunk))

        post_ents = []
        for i in range(0, len(post_lines) - WINDOW + 1, STRIDE):
            chunk = '\n'.join(post_lines[i:i + WINDOW])
            post_ents.append(text_entropy(chunk))

        pre_arr = np.array(pre_ents)
        post_arr = np.array(post_ents)

        print(f"  Split at dialogue line {split_idx}/{len(dent_lines)}")
        print(f"  Pre-transformation (Harvey Dent, DA):")
        print(f"    {len(pre_lines)} lines, zlib mean={np.mean(pre_arr):.4f} +/- {np.std(pre_arr):.4f}")
        print(f"  Post-transformation (Two-Face):")
        print(f"    {len(post_lines)} lines, zlib mean={np.mean(post_arr):.4f} +/- {np.std(post_arr):.4f}")
        print(f"  Entropy shift: {np.mean(post_arr) - np.mean(pre_arr):+.4f}")

        results['DENT_PRE'] = {
            'total_lines': len(pre_lines),
            'entropy_mean': round(float(np.mean(pre_arr)), 6),
            'entropy_std': round(float(np.std(pre_arr)), 6),
        }
        results['DENT_POST'] = {
            'total_lines': len(post_lines),
            'entropy_mean': round(float(np.mean(post_arr)), 6),
            'entropy_std': round(float(np.std(post_arr)), 6),
        }


# ══════════════════════════════════════
# 3. Batman vs Wayne comparison
# ══════════════════════════════════════
print("\n" + "=" * 70)
print("Bruce Wayne — BATMAN vs WAYNE persona comparison")
print("=" * 70)

for persona in ['BATMAN', 'WAYNE']:
    if persona in results:
        r = results[persona]
        print(f"  {persona:>8}: {r['total_lines']} lines, "
              f"zlib={r['entropy_mean']:.4f}+/-{r['entropy_std']:.4f}, "
              f"cos={r['cosine_mean']:.4f}+/-{r['cosine_std']:.4f}")

# Cross-cosine: compare Batman's overall character frequency to Wayne's
if 'BATMAN' in dialogue and 'WAYNE' in dialogue:
    bat_text = '\n'.join(dialogue['BATMAN'])
    wayne_text = '\n'.join(dialogue['WAYNE'])
    bat_vec = char_frequency(bat_text)
    wayne_vec = char_frequency(wayne_text)
    cross_cos = cosine_sim(bat_vec, wayne_vec)
    print(f"\n  Cross-cosine (Batman <-> Wayne): {cross_cos:.4f}")
    print(f"  (1.0 = identical vocabulary, lower = more distinct personas)")
    results['BATMAN_WAYNE_CROSS_COSINE'] = round(cross_cos, 6)


# ══════════════════════════════════════
# 4. Graph: Entropy Timeline
# ══════════════════════════════════════
chars_with_data = [c for c in MAIN_CHARS if c in results]
n_chars = len(chars_with_data)

fig, axes = plt.subplots(n_chars, 1, figsize=(16, 2.5 * n_chars), sharex=True)
if n_chars == 1:
    axes = [axes]

fig.suptitle('The Dark Knight — zlib Entropy Timeline\n'
             'Character voice consistency across the screenplay',
             fontsize=16, fontweight='bold', y=0.995)

for idx, char in enumerate(chars_with_data):
    ax = axes[idx]
    r = results[char]
    pos = r['timeline']['positions']
    ent = r['timeline']['entropy']
    color = CHAR_COLORS.get(char, 'gray')
    label = CHAR_LABELS.get(char, char)

    ax.plot(pos, ent, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, ent, alpha=0.15, color=color)

    mean_e = r['entropy_mean']
    ax.axhline(y=mean_e, color=color, linestyle='--', linewidth=1, alpha=0.5)

    ax.text(0.98, 0.95,
            f'mean={r["entropy_mean"]:.4f}\n'
            f'\u03C3={r["entropy_std"]:.4f}\n'
            f'lines={r["total_lines"]}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel(label, fontsize=11, fontweight='bold', color=color,
                  rotation=0, labelpad=45)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)

axes[-1].set_xlabel('Story progression (0=start, 1=end)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{BASE}/dark_knight_zlib_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: dark_knight_zlib_timeline.png")


# ══════════════════════════════════════
# 5. Graph: Cosine Similarity Timeline
# ══════════════════════════════════════
fig, axes = plt.subplots(n_chars, 1, figsize=(16, 2.5 * n_chars), sharex=True)
if n_chars == 1:
    axes = [axes]

fig.suptitle('The Dark Knight — Cosine Similarity Timeline\n'
             'How much does vocabulary shift between adjacent windows?',
             fontsize=16, fontweight='bold', y=0.995)

for idx, char in enumerate(chars_with_data):
    ax = axes[idx]
    r = results[char]
    pos = r['cosine_timeline']['positions']
    cos = r['cosine_timeline']['cosine']
    color = CHAR_COLORS.get(char, 'gray')
    label = CHAR_LABELS.get(char, char)

    if pos:
        ax.plot(pos, cos, color=color, linewidth=2, alpha=0.8)
        ax.fill_between(pos, cos, alpha=0.15, color=color)

    mean_c = r['cosine_mean']
    ax.axhline(y=mean_c, color=color, linestyle='--', linewidth=1, alpha=0.5)

    ax.text(0.98, 0.05,
            f'mean={r["cosine_mean"]:.4f}\n'
            f'\u03C3={r["cosine_std"]:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel(label, fontsize=11, fontweight='bold', color=color,
                  rotation=0, labelpad=45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.15)

axes[-1].set_xlabel('Story progression (0=start, 1=end)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{BASE}/dark_knight_cosine_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dark_knight_cosine_timeline.png")


# ══════════════════════════════════════
# 6. Scatter Plot: Voice Fingerprint
# ══════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

for char in chars_with_data:
    r = results[char]
    color = CHAR_COLORS.get(char, 'gray')
    label = CHAR_LABELS.get(char, char)

    ax.scatter(r['entropy_std'], r['cosine_std'],
               s=r['total_lines'] * 2, color=color, alpha=0.7,
               edgecolors='white', linewidth=2, zorder=10)
    ax.annotate(label, (r['entropy_std'], r['cosine_std']),
                xytext=(10, 8), textcoords='offset points',
                fontsize=12, fontweight='bold', color=color)

ax.set_xlabel('Entropy \u03C3 (voice instability)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cosine \u03C3 (vocabulary shift)', fontsize=13, fontweight='bold')
ax.set_title('The Dark Knight — Character Voice Fingerprint\n'
             'Size = dialogue count | Position = instability',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{BASE}/dark_knight_voice_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dark_knight_voice_scatter.png")


# ══════════════════════════════════════
# 7. Special: Dent Transformation Chart
# ══════════════════════════════════════
if 'DENT' in results and len(dent_lines) > WINDOW * 2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    r = results['DENT']
    pos = r['timeline']['positions']
    ent = r['timeline']['entropy']

    # Color gradient: orange (Harvey) -> dark red (Two-Face)
    split_pos = split_idx / len(dent_lines)

    for i in range(len(pos)):
        color = '#FF8C00' if pos[i] < split_pos else '#8B0000'
        if i > 0:
            ax1.plot(pos[i-1:i+1], ent[i-1:i+1], color=color, linewidth=2.5, alpha=0.8)

    ax1.axvline(x=split_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(split_pos - 0.02, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else min(ent),
             'Harvey Dent\n(White Knight)', ha='right', fontsize=10, color='#FF8C00', fontweight='bold')
    ax1.text(split_pos + 0.02, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else min(ent),
             'Two-Face\n(Fallen)', ha='left', fontsize=10, color='#8B0000', fontweight='bold')

    ax1.axhline(y=r['entropy_mean'], color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('zlib Entropy', fontsize=12, fontweight='bold')
    ax1.set_title('Harvey Dent → Two-Face: Entropy Transformation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.15)

    # Cosine timeline
    cos_pos = r['cosine_timeline']['positions']
    cos_val = r['cosine_timeline']['cosine']
    for i in range(len(cos_pos)):
        color = '#FF8C00' if cos_pos[i] < split_pos else '#8B0000'
        if i > 0:
            ax2.plot(cos_pos[i-1:i+1], cos_val[i-1:i+1], color=color, linewidth=2.5, alpha=0.8)

    ax2.axvline(x=split_pos, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=r['cosine_mean'], color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dent dialogue progression', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'{BASE}/dark_knight_dent_transform.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: dark_knight_dent_transform.png")


# ══════════════════════════════════════
# 8. Save results
# ══════════════════════════════════════
save_results = {}
for k, v in results.items():
    if isinstance(v, dict):
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if kk not in ('timeline', 'cosine_timeline')}
    else:
        save_results[k] = v

with open(f'{BASE}/dark_knight_zlib_results.json', 'w', encoding='utf-8') as f:
    json.dump(save_results, f, ensure_ascii=False, indent=2)
print("Saved: dark_knight_zlib_results.json")

# Summary table
print("\n" + "=" * 80)
print(f"{'Character':>12} {'Lines':>6} {'Entropy':>8} {'sigma_ent':>10} {'Cosine':>8} {'sigma_cos':>10}")
print("-" * 80)
for char in MAIN_CHARS:
    if char not in results:
        continue
    r = results[char]
    label = CHAR_LABELS.get(char, char)
    print(f"{label:>12} {r['total_lines']:>6} {r['entropy_mean']:>8.4f} {r['entropy_std']:>10.4f} "
          f"{r['cosine_mean']:>8.4f} {r['cosine_std']:>10.4f}")

# Rank by sigma
print("\n  Entropy sigma ranking (voice instability):")
ranked = sorted([(CHAR_LABELS.get(c, c), results[c]['entropy_std'])
                 for c in chars_with_data], key=lambda x: -x[1])
for rank, (name, sigma) in enumerate(ranked, 1):
    marker = " <<<" if rank == 1 else ""
    print(f"    {rank}. {name}: {sigma:.4f}{marker}")

print("\n  Cosine sigma ranking (vocabulary shift):")
ranked_cos = sorted([(CHAR_LABELS.get(c, c), results[c]['cosine_std'])
                     for c in chars_with_data], key=lambda x: -x[1])
for rank, (name, sigma) in enumerate(ranked_cos, 1):
    marker = " <<<" if rank == 1 else ""
    print(f"    {rank}. {name}: {sigma:.4f}{marker}")
