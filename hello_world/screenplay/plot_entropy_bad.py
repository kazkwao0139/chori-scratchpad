"""
Plot entropy analysis on newly collected BAD movie scripts.
Classify scenes → compute metrics → compare with existing >=8.0 data.
"""

import json
import sys
import re
import math
import time
import os
import urllib.request
import urllib.error

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
API_KEY = open(f'{BASE}/claude apikey.txt').read().strip()
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
CHECKPOINT = f'{BASE}/screenplay/plot_entropy_checkpoint.json'

BAD_SCRIPTS_DIR = f'{BASE}/screenplay/bad_scripts'

# Movie info: filename -> (title, imdb_rating)
BAD_MOVIES = {
    'the_room.txt': ('The Room', 3.6),
    'gigli.txt': ('Gigli', 2.5),
    'disaster_movie.txt': ('Disaster Movie', 1.9),
    'from_justin_to_kelly.txt': ('From Justin to Kelly', 2.0),
    'epic_movie.txt': ('Epic Movie', 2.4),
    'battlefield_earth.txt': ('Battlefield Earth', 2.5),
    'catwoman.txt': ('Catwoman_bad', 3.4),  # _bad to not clash with existing
    'dragonball_evolution.txt': ('Dragonball Evolution', 2.5),
    'mortal_kombat_annihilation.txt': ('Mortal Kombat Annihilation', 3.6),
    'showgirls.txt': ('Showgirls', 4.6),
    'jaws_revenge.txt': ('Jaws The Revenge', 3.1),
    'plan_9.txt': ('Plan 9 from Outer Space', 3.9),
    'alone_in_the_dark_v2.txt': ('Alone in the Dark v2', 2.4),
}

EVENT_TYPES = [
    "SETUP", "INCITING", "GOAL", "ACTION", "OBSTACLE",
    "DISASTER", "DISCOVERY", "RESOLUTION", "EMOTION",
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


def call_claude(prompt, max_tokens=400):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    }).encode('utf-8')
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01"
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data['content'][0]['text']
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if e.code == 529:
                time.sleep(15)
                continue
            if attempt < 4:
                time.sleep(5)
                continue
            return None
        except Exception as e:
            if attempt < 4:
                time.sleep(5)
                continue
            return None
    return None


def split_scenes_screenplay(text):
    """Split by INT./EXT. markers."""
    pattern = re.compile(r'\n\s*((?:INT|EXT|INT\./EXT|I/E)[\.\s].+)', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    scenes = []
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) > 50:
            scenes.append((header, body))
    return scenes


def split_scenes_transcript(text):
    """Split transcript into chunks by paragraph breaks or page-like boundaries."""
    # Try splitting by large gaps (multiple newlines)
    chunks = re.split(r'\n\s*\n\s*\n', text)
    scenes = []
    current = []
    current_len = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        current.append(chunk)
        current_len += len(chunk)
        # Target ~2000 chars per "scene" (roughly one screenplay page)
        if current_len >= 2000:
            block = '\n\n'.join(current)
            scenes.append(('CHUNK ' + str(len(scenes)+1), block))
            current = []
            current_len = 0

    if current:
        block = '\n\n'.join(current)
        if len(block) > 200:
            scenes.append(('CHUNK ' + str(len(scenes)+1), block))

    return scenes


def classify_scenes(scenes, title):
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
            "Respond with ONLY numbered lines like:\n1. ACTION\n2. DISCOVERY\n3. EMOTION"
        )
        resp = call_claude(prompt)
        if resp:
            for line in resp.strip().split('\n'):
                line = line.strip()
                m = re.match(r'\d+[\.\)]\s*(\w+)', line)
                if m:
                    label = m.group(1).upper()
                    if label in EVENT_TYPES:
                        labels.append(label)
                    else:
                        for et in EVENT_TYPES:
                            if et.startswith(label[:4]):
                                labels.append(et)
                                break
                        else:
                            labels.append('SETUP')
    return labels


def compute_metrics(labels):
    n = len(labels)
    if n < 5:
        return None
    freq = {}
    for t in labels:
        freq[t] = freq.get(t, 0) + 1
    uni_entropy = -sum((c/n) * math.log2(c/n) for c in freq.values() if c > 0)
    transitions = {}
    for i in range(n - 1):
        pair = (labels[i], labels[i+1])
        transitions[pair] = transitions.get(pair, 0) + 1
    nt = sum(transitions.values())
    bi_entropy = -sum((c/nt) * math.log2(c/nt) for c in transitions.values() if c > 0) if nt > 0 else 0
    coverage = len(freq) / len(EVENT_TYPES)
    repeats = sum(1 for i in range(n-1) if labels[i] == labels[i+1])
    repeat_ratio = repeats / (n - 1)
    active = sum(1 for t in labels if t in ('GOAL', 'ACTION', 'RESOLUTION'))
    passive = sum(1 for t in labels if t in ('OBSTACLE', 'DISASTER', 'SETUP'))
    agency_ratio = active / max(passive, 1)
    disaster_ratio = freq.get('DISASTER', 0) / n
    action_ratio = freq.get('ACTION', 0) / n
    emotion_ratio = freq.get('EMOTION', 0) / n
    mid = n // 2
    fh, sh = labels[:mid], labels[mid:]
    fh_freq, sh_freq = {}, {}
    for t in fh: fh_freq[t] = fh_freq.get(t, 0) + 1
    for t in sh: sh_freq[t] = sh_freq.get(t, 0) + 1
    all_types = set(list(fh_freq.keys()) + list(sh_freq.keys()))
    arc_shift = sum(abs(fh_freq.get(t, 0)/max(len(fh),1) - sh_freq.get(t, 0)/max(len(sh),1))
                    for t in all_types) / max(len(all_types), 1)
    if freq.get('SETUP', 0) > 0:
        setup_positions = [i/n for i, t in enumerate(labels) if t == 'SETUP']
        setup_front = sum(1 for p in setup_positions if p < 0.25) / len(setup_positions)
    else:
        setup_front = 0

    def count_subseq(seq, pattern):
        count = 0
        for start in range(len(seq)):
            if seq[start] == pattern[0]:
                pos = start
                matched = 1
                for pi in range(1, len(pattern)):
                    found = False
                    for look in range(pos+1, min(pos+4, len(seq))):
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

    scene_cycles = count_subseq(labels, ['GOAL', 'ACTION', 'OBSTACLE', 'DISASTER'])
    sequel_cycles = count_subseq(labels, ['EMOTION', 'DISCOVERY', 'GOAL'])
    cycle_density = (scene_cycles + sequel_cycles) / max(n / 4, 1)

    return {
        'n_scenes': n,
        'uni_entropy': round(uni_entropy, 4),
        'bi_entropy': round(bi_entropy, 4),
        'coverage': round(coverage, 4),
        'repeat_ratio': round(repeat_ratio, 4),
        'agency_ratio': round(agency_ratio, 4),
        'disaster_ratio': round(disaster_ratio, 4),
        'action_ratio': round(action_ratio, 4),
        'emotion_ratio': round(emotion_ratio, 4),
        'arc_shift': round(arc_shift, 4),
        'setup_front': round(setup_front, 4),
        'cycle_density': round(cycle_density, 4),
        'type_freq': freq,
        'labels': labels,
    }


def main():
    print("=" * 70)
    print("  PLOT ENTROPY — BAD MOVIES COLLECTION")
    print("=" * 70)

    # Load existing checkpoint
    try:
        checkpoint = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
    except:
        checkpoint = {}

    t0 = time.time()
    total_api = 0
    processed = 0

    for fname, (title, rating) in BAD_MOVIES.items():
        if title in checkpoint and checkpoint[title].get('labels'):
            print(f"  SKIP (done): {title} ({rating})")
            continue

        fpath = os.path.join(BAD_SCRIPTS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP (no file): {fname}")
            continue

        text = open(fpath, 'r', encoding='utf-8', errors='replace').read()
        print(f"\n  [{title}] ({rating}) — {len(text)} chars")

        # Determine split method
        n_int_ext = len(re.findall(r'INT\.|EXT\.', text, re.IGNORECASE))
        if n_int_ext >= 10:
            scenes = split_scenes_screenplay(text)
            method = 'screenplay'
        else:
            scenes = split_scenes_transcript(text)
            method = 'transcript'

        print(f"    Split method: {method}, scenes: {len(scenes)}")

        if len(scenes) < 5:
            print(f"    SKIP — too few scenes")
            checkpoint[title] = {'rating': rating, 'skip': True}
            continue

        if len(scenes) > 60:
            scenes = scenes[:60]

        n_batches = math.ceil(len(scenes) / 10)
        total_api += n_batches

        labels = classify_scenes(scenes, title)
        print(f"    Labels: {len(labels)}")

        if len(labels) < 5:
            print(f"    SKIP — too few labels")
            checkpoint[title] = {'rating': rating, 'skip': True}
            continue

        metrics = compute_metrics(labels)
        if metrics:
            metrics['rating'] = rating
            metrics['source'] = 'bad_collection'
            checkpoint[title] = metrics
            processed += 1

        abbrev = ' '.join(t[:3] for t in labels[:25])
        print(f"    Seq: {abbrev}...")
        print(f"    agency={metrics['agency_ratio']:.2f} action={metrics['action_ratio']:.1%} "
              f"emotion={metrics['emotion_ratio']:.1%} disaster={metrics['disaster_ratio']:.1%}")

        # Save checkpoint
        if processed % 5 == 0:
            json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # Final save
    json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Processed: {processed}, API calls: {total_api}, time: {elapsed:.0f}s")

    # ── FINAL COMPARISON ──
    print(f"\n{'=' * 70}")
    print("  FINAL: BAD (<4.0) vs GOOD (>=8.0)")
    print(f"{'=' * 70}")

    all_valid = [(k, v) for k, v in checkpoint.items()
                 if isinstance(v, dict) and v.get('labels') and v.get('rating')]

    bad = [(k, v) for k, v in all_valid if v['rating'] < 4.0]
    good = [(k, v) for k, v in all_valid if v['rating'] >= 8.0]

    print(f"  Bad (<4.0): {len(bad)}")
    print(f"  Good (>=8.0): {len(good)}")

    for k, v in sorted(bad, key=lambda x: x[1]['rating']):
        print(f"    {v['rating']:.1f} {k}: agency={v['agency_ratio']:.2f} action={v['action_ratio']:.1%} emotion={v['emotion_ratio']:.1%}")

    metrics_names = [
        'uni_entropy', 'bi_entropy', 'coverage', 'repeat_ratio',
        'agency_ratio', 'disaster_ratio', 'action_ratio', 'emotion_ratio',
        'arc_shift', 'setup_front', 'cycle_density'
    ]

    print(f"\n  {'Metric':>15s} {'Good':>8s} {'Bad':>8s} {'delta':>8s} {'d':>8s} {'effect':>8s}")
    print(f"  {'-'*55}")

    for m in metrics_names:
        gv = [v[m] for _, v in good]
        bv = [v[m] for _, v in bad]
        gm = sum(gv) / len(gv)
        bm = sum(bv) / len(bv)
        delta = gm - bm
        g_var = sum((x - gm) ** 2 for x in gv) / max(len(gv) - 1, 1)
        b_var = sum((x - bm) ** 2 for x in bv) / max(len(bv) - 1, 1)
        pooled = ((g_var + b_var) / 2) ** 0.5
        d = delta / pooled if pooled > 0 else 0
        effect = 'LARGE' if abs(d) > 0.8 else 'MED' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else '-'
        print(f"  {m:>15s} {gm:8.4f} {bm:8.4f} {delta:+8.4f} {d:+8.3f} {effect:>8s}")

    sig = sum(1 for m in metrics_names
              if abs(_d(good, bad, m)) > 0.2)
    med = sum(1 for m in metrics_names
              if abs(_d(good, bad, m)) > 0.5)
    large = sum(1 for m in metrics_names
                if abs(_d(good, bad, m)) > 0.8)
    print(f"\n  |d|>0.2: {sig}/{len(metrics_names)}  |d|>0.5: {med}  |d|>0.8: {large}")

    # Event type distribution
    print(f"\n  Event type distribution:")
    g_freq, b_freq = {}, {}
    for _, v in good:
        for t, c in v['type_freq'].items():
            g_freq[t] = g_freq.get(t, 0) + c
    for _, v in bad:
        for t, c in v['type_freq'].items():
            b_freq[t] = b_freq.get(t, 0) + c
    g_total = sum(g_freq.values())
    b_total = sum(b_freq.values())

    print(f"  {'Type':>12s} {'Good%':>8s} {'Bad%':>8s} {'Δ%':>8s}")
    print(f"  {'-'*40}")
    for t in EVENT_TYPES:
        gp = g_freq.get(t, 0) / g_total * 100 if g_total else 0
        bp = b_freq.get(t, 0) / b_total * 100 if b_total else 0
        marker = " <<<" if abs(gp - bp) > 5 else " <<" if abs(gp - bp) > 3 else ""
        print(f"  {t:>12s} {gp:7.1f}% {bp:7.1f}% {gp-bp:+7.1f}%{marker}")

    print(f"\n  Done.")


def _d(good, bad, metric):
    gv = [v[metric] for _, v in good]
    bv = [v[metric] for _, v in bad]
    gm = sum(gv) / len(gv)
    bm = sum(bv) / len(bv)
    g_var = sum((x - gm) ** 2 for x in gv) / max(len(gv) - 1, 1)
    b_var = sum((x - bm) ** 2 for x in bv) / max(len(bv) - 1, 1)
    pooled = ((g_var + b_var) / 2) ** 0.5
    return (gm - bm) / pooled if pooled > 0 else 0


if __name__ == "__main__":
    main()
