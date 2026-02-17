"""
Plot-level entropy v2: 9 event types, Claude classifies directly.
Top 15 vs Bottom 15 pilot.

Event types (Scene-Sequel cycle):
  SETUP, INCITING, GOAL, ACTION, OBSTACLE, DISASTER, DISCOVERY, RESOLUTION, EMOTION
"""

import json
import sys
import re
import math
import time
import urllib.request
import urllib.error

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
API_KEY = open(f'{BASE}/claude apikey.txt').read().strip()
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

EVENT_TYPES = [
    "SETUP",       # Calm state, background, character introduction
    "INCITING",    # Event that breaks the status quo
    "GOAL",        # Character decides/declares what to do (internal)
    "ACTION",      # Character physically acts toward goal (external)
    "OBSTACLE",    # External force blocks the action
    "DISASTER",    # Action fails, situation worsens
    "DISCOVERY",   # New information, secret revealed, twist
    "RESOLUTION",  # Obstacle overcome, small victory
    "EMOTION",     # Reaction, bonding, romance, inner change, pacing breath
]

SYSTEM_PROMPT = """You are a screenplay scene classifier. You classify each scene into exactly ONE of these 9 types:

SETUP - Calm state, background exposition, character introduction. The "before" state.
INCITING - An event that disrupts the status quo (a phone rings, a body is found, a stranger arrives).
GOAL - A character decides or declares what they will do. Internal commitment, planning.
ACTION - A character physically acts toward their goal (infiltrates, runs, builds, fights).
OBSTACLE - An external force blocks progress (enemy appears, door is locked, argument erupts).
DISASTER - The action fails or the situation worsens. The protagonist suffers a setback.
DISCOVERY - New information is gained, a secret is revealed, an unexpected truth emerges.
RESOLUTION - An obstacle is overcome, a small victory is achieved, tension releases.
EMOTION - Emotional reaction, bonding, comfort, romance, grief, inner transformation. The breathing room.

Rules:
- Pick the SINGLE most dominant type for each scene.
- If a scene has multiple elements, pick the one that DRIVES the scene forward.
- Respond with ONLY the type label, one per line, numbered."""


def call_claude(prompt, system=SYSTEM_PROMPT, max_tokens=400):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}]
    }).encode('utf-8')

    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01"
    })

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data['content'][0]['text']
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            err_body = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
            print(f"    API error {e.code}: {err_body[:200]}")
            raise
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            raise
    return None


def split_scenes(text):
    pattern = re.compile(r'\n\s*((?:INT|EXT|INT\./EXT|I/E)[\.\s].+)', re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if len(matches) < 5:
        chunks = re.split(r'\n\s*\n\s*\n', text)
        scenes = []
        for i in range(0, len(chunks), 3):
            block = '\n'.join(chunks[i:i+3]).strip()
            if len(block) > 100:
                scenes.append(('BLOCK ' + str(i//3 + 1), block))
        return scenes

    scenes = []
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) > 50:
            scenes.append((header, body))
    return scenes


def classify_scenes_batch(scenes, title):
    """Have Claude classify each scene into one of 9 types."""
    labels = []
    batch_size = 10

    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i+batch_size]
        scene_texts = []
        for j, (header, body) in enumerate(batch):
            truncated = body[:600] + ('...' if len(body) > 600 else '')
            scene_texts.append(f"SCENE {i+j+1} [{header[:60]}]:\n{truncated}")

        prompt = (
            f"Movie: {title}\n\n"
            + "\n---\n".join(scene_texts)
            + "\n\n---\nClassify each scene above into exactly ONE type. "
            "Respond with ONLY numbered lines like:\n"
            "1. ACTION\n2. DISCOVERY\n3. EMOTION\netc."
        )

        resp = call_claude(prompt)
        if resp:
            for line in resp.strip().split('\n'):
                line = line.strip()
                m = re.match(r'\d+[\.\)]\s*(\w+)', line)
                if m:
                    label = m.group(1).upper()
                    # Normalize
                    if label in EVENT_TYPES:
                        labels.append(label)
                    else:
                        # Fuzzy match
                        best = min(EVENT_TYPES, key=lambda t: abs(len(t) - len(label)))
                        for et in EVENT_TYPES:
                            if et.startswith(label[:4]):
                                best = et
                                break
                        labels.append(best)

    return labels


def compute_metrics(labels):
    """Compute entropy and pattern metrics from event label sequence."""
    n = len(labels)
    if n < 5:
        return {}

    # ── Unigram entropy ──
    freq = {}
    for t in labels:
        freq[t] = freq.get(t, 0) + 1
    uni_entropy = -sum((c/n) * math.log2(c/n) for c in freq.values() if c > 0)

    # ── Bigram transition entropy ──
    transitions = {}
    for i in range(n - 1):
        pair = (labels[i], labels[i+1])
        transitions[pair] = transitions.get(pair, 0) + 1
    nt = sum(transitions.values())
    bi_entropy = -sum((c/nt) * math.log2(c/nt) for c in transitions.values() if c > 0) if nt > 0 else 0

    # ── Coverage ──
    coverage = len(freq) / len(EVENT_TYPES)

    # ── Repeat ratio ──
    repeats = sum(1 for i in range(n-1) if labels[i] == labels[i+1])
    repeat_ratio = repeats / (n - 1)

    # ── Scene-Sequel cycle detection ──
    # Scene cycle: GOAL → ACTION → OBSTACLE → DISASTER
    # Sequel cycle: EMOTION → DISCOVERY → GOAL
    scene_cycle = ['GOAL', 'ACTION', 'OBSTACLE', 'DISASTER']
    sequel_cycle = ['EMOTION', 'DISCOVERY', 'GOAL']

    def count_subsequences(seq, pattern):
        """Count how many times pattern appears as subsequence (with gaps ≤ 3)."""
        count = 0
        for start in range(len(seq)):
            if seq[start] == pattern[0]:
                pos = start
                matched = 1
                for pi in range(1, len(pattern)):
                    found = False
                    for look in range(pos + 1, min(pos + 4, len(seq))):  # max gap 3
                        if seq[look] == pattern[pi]:
                            pos = look
                            matched += 1
                            found = True
                            break
                    if not found:
                        break
                if matched == len(pattern):
                    count += 1
        return count

    scene_cycles = count_subsequences(labels, scene_cycle)
    sequel_cycles = count_subsequences(labels, sequel_cycle)
    cycle_density = (scene_cycles + sequel_cycles) / max(n / 4, 1)  # normalize by expected

    # ── Agency ratio ──
    # GOAL + ACTION + DECISION vs passive events
    active = sum(1 for t in labels if t in ('GOAL', 'ACTION', 'RESOLUTION'))
    passive = sum(1 for t in labels if t in ('OBSTACLE', 'DISASTER', 'SETUP'))
    agency_ratio = active / max(passive, 1)

    # ── Disaster density ──
    # "Put your hero up a tree and throw rocks"
    disaster_ratio = freq.get('DISASTER', 0) / n

    # ── Arc: first half vs second half type shift ──
    mid = n // 2
    fh = labels[:mid]
    sh = labels[mid:]
    fh_freq = {}
    sh_freq = {}
    for t in fh: fh_freq[t] = fh_freq.get(t, 0) + 1
    for t in sh: sh_freq[t] = sh_freq.get(t, 0) + 1
    all_types = set(list(fh_freq.keys()) + list(sh_freq.keys()))
    arc_shift = sum(abs(fh_freq.get(t, 0)/max(len(fh), 1) - sh_freq.get(t, 0)/max(len(sh), 1))
                    for t in all_types) / max(len(all_types), 1)

    # ── SETUP front-loading ──
    # Good screenplays should have SETUP concentrated at the beginning
    if freq.get('SETUP', 0) > 0:
        setup_positions = [i/n for i, t in enumerate(labels) if t == 'SETUP']
        setup_front = sum(1 for p in setup_positions if p < 0.25) / len(setup_positions)
    else:
        setup_front = 0

    # ── INCITING position ──
    inciting_positions = [i/n for i, t in enumerate(labels) if t == 'INCITING']
    inciting_pos = inciting_positions[0] if inciting_positions else 1.0  # first inciting event

    return {
        'n_scenes': n,
        'uni_entropy': round(uni_entropy, 4),
        'bi_entropy': round(bi_entropy, 4),
        'coverage': round(coverage, 4),
        'repeat_ratio': round(repeat_ratio, 4),
        'cycle_density': round(cycle_density, 4),
        'scene_cycles': scene_cycles,
        'sequel_cycles': sequel_cycles,
        'agency_ratio': round(agency_ratio, 4),
        'disaster_ratio': round(disaster_ratio, 4),
        'arc_shift': round(arc_shift, 4),
        'setup_front': round(setup_front, 4),
        'inciting_pos': round(inciting_pos, 4),
        'type_freq': freq,
    }


def main():
    print("=" * 70)
    print("  PLOT ENTROPY v2 — 9 Event Types, Claude Classification")
    print("=" * 70)

    # Load ratings
    mv2c = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))
    done = {k: v for k, v in mv2c['done'].items() if v and isinstance(v, dict) and v.get('rating')}

    # Load texts
    print("\n  Loading screenplay texts...")
    texts = {}
    mcp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
    for k, v in mcp.get('scripts', {}).items():
        if v and len(v) > 5000:
            texts[k] = v
    fc = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
    for k, v in fc.items():
        if v and len(v) > 5000 and k not in texts:
            texts[k] = v

    rated = []
    for title, info in done.items():
        if title in texts:
            rated.append({'title': title, 'rating': info['rating']})
    rated.sort(key=lambda x: x['rating'])
    print(f"  Rated with text: {len(rated)}")

    bottom15 = rated[:15]
    top15 = rated[-15:]

    print(f"  Top 15: {top15[0]['title']}({top15[0]['rating']}) ~ {top15[-1]['title']}({top15[-1]['rating']})")
    print(f"  Bottom 15: {bottom15[0]['title']}({bottom15[0]['rating']}) ~ {bottom15[-1]['title']}({bottom15[-1]['rating']})")

    results = []
    total_api = 0

    for group_name, group in [("BOTTOM 15", bottom15), ("TOP 15", top15)]:
        print(f"\n{'=' * 70}")
        print(f"  {group_name}")
        print(f"{'=' * 70}")

        for movie in group:
            title = movie['title']
            text = texts[title]
            rating = movie['rating']

            print(f"\n  [{title}] (rating={rating})")

            scenes = split_scenes(text)
            print(f"    Scenes: {len(scenes)}")

            if len(scenes) < 5:
                print(f"    SKIP — too few scenes")
                continue

            if len(scenes) > 60:
                scenes = scenes[:60]

            n_batches = math.ceil(len(scenes) / 10)
            total_api += n_batches

            labels = classify_scenes_batch(scenes, title)
            print(f"    Labels: {len(labels)}")

            if len(labels) < 5:
                print(f"    SKIP — too few labels")
                continue

            # Show sequence preview
            abbrev = [t[:3] for t in labels[:20]]
            print(f"    Seq: {' → '.join(abbrev)}{'...' if len(labels) > 20 else ''}")

            metrics = compute_metrics(labels)
            metrics['title'] = title
            metrics['rating'] = rating
            metrics['group'] = group_name
            metrics['labels'] = labels
            results.append(metrics)

            print(f"    H={metrics['uni_entropy']:.3f}  biH={metrics['bi_entropy']:.3f}  "
                  f"cycles={metrics['scene_cycles']}+{metrics['sequel_cycles']}  "
                  f"agency={metrics['agency_ratio']:.2f}  disaster={metrics['disaster_ratio']:.3f}")

    # ── Analysis ──
    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS (API calls: {total_api})")
    print(f"{'=' * 70}")

    top_r = [r for r in results if r['group'] == 'TOP 15']
    bot_r = [r for r in results if r['group'] == 'BOTTOM 15']

    compare_metrics = [
        'uni_entropy', 'bi_entropy', 'coverage', 'repeat_ratio',
        'cycle_density', 'agency_ratio', 'disaster_ratio',
        'arc_shift', 'setup_front', 'inciting_pos'
    ]

    print(f"\n  {'Metric':>15s} {'Top15':>8s} {'Bot15':>8s} {'Δ':>8s} {'Cohen_d':>8s} {'Effect':>8s}")
    print(f"  {'-'*60}")

    for metric in compare_metrics:
        top_vals = [r[metric] for r in top_r if metric in r]
        bot_vals = [r[metric] for r in bot_r if metric in r]
        if not top_vals or not bot_vals:
            continue

        top_mean = sum(top_vals) / len(top_vals)
        bot_mean = sum(bot_vals) / len(bot_vals)
        delta = top_mean - bot_mean

        top_var = sum((v - top_mean)**2 for v in top_vals) / max(len(top_vals)-1, 1)
        bot_var = sum((v - bot_mean)**2 for v in bot_vals) / max(len(bot_vals)-1, 1)
        pooled_sd = ((top_var + bot_var) / 2) ** 0.5
        d = delta / pooled_sd if pooled_sd > 0 else 0

        effect = "LARGE" if abs(d) > 0.8 else "MEDIUM" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "-"
        print(f"  {metric:>15s} {top_mean:8.4f} {bot_mean:8.4f} {delta:+8.4f} {d:+8.3f} {effect:>8s}")

    # ── Type distribution ──
    print(f"\n  Event type distribution:")
    top_freq = {}
    bot_freq = {}
    for r in top_r:
        for t, c in r['type_freq'].items():
            top_freq[t] = top_freq.get(t, 0) + c
    for r in bot_r:
        for t, c in r['type_freq'].items():
            bot_freq[t] = bot_freq.get(t, 0) + c

    top_total = sum(top_freq.values())
    bot_total = sum(bot_freq.values())

    print(f"\n  {'Type':>12s} {'Top15%':>8s} {'Bot15%':>8s} {'Δ%':>8s}")
    print(f"  {'-'*40}")
    for t in EVENT_TYPES:
        tp = top_freq.get(t, 0) / top_total * 100 if top_total else 0
        bp = bot_freq.get(t, 0) / bot_total * 100 if bot_total else 0
        marker = " <<<" if abs(tp - bp) > 5 else " <<" if abs(tp - bp) > 3 else ""
        print(f"  {t:>12s} {tp:7.1f}% {bp:7.1f}% {tp-bp:+7.1f}%{marker}")

    # ── Sequence pattern examples ──
    print(f"\n  Example sequences (top-rated):")
    for r in sorted(top_r, key=lambda x: -x['rating'])[:3]:
        seq = ' '.join(t[:3] for t in r['labels'][:30])
        print(f"    {r['title']} ({r['rating']}): {seq}")

    print(f"\n  Example sequences (bottom-rated):")
    for r in sorted(bot_r, key=lambda x: x['rating'])[:3]:
        seq = ' '.join(t[:3] for t in r['labels'][:30])
        print(f"    {r['title']} ({r['rating']}): {seq}")

    # Save
    save_path = f'{BASE}/screenplay/plot_entropy_v2_results.json'
    json.dump(results, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n  Saved to {save_path}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
