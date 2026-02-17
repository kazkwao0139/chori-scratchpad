from pathlib import Path
"""
Plot-level entropy FULL: all 171 movies with available text.
9 event types, Claude Haiku classification.
Checkpointing for crash recovery.
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

BASE = str(Path(__file__).resolve().parent.parent)
API_KEY = open(f'{BASE}/claude apikey.txt').read().strip()
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
CHECKPOINT = f'{BASE}/screenplay/plot_entropy_checkpoint.json'

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
                wait = 10 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 529:
                time.sleep(15)
                continue
            err_body = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
            print(f"      API error {e.code}: {err_body[:200]}")
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

    # Agency
    active = sum(1 for t in labels if t in ('GOAL', 'ACTION', 'RESOLUTION'))
    passive = sum(1 for t in labels if t in ('OBSTACLE', 'DISASTER', 'SETUP'))
    agency_ratio = active / max(passive, 1)

    # Disaster density
    disaster_ratio = freq.get('DISASTER', 0) / n

    # Action density
    action_ratio = freq.get('ACTION', 0) / n

    # Emotion density
    emotion_ratio = freq.get('EMOTION', 0) / n

    # Arc shift
    mid = n // 2
    fh, sh = labels[:mid], labels[mid:]
    fh_freq, sh_freq = {}, {}
    for t in fh: fh_freq[t] = fh_freq.get(t, 0) + 1
    for t in sh: sh_freq[t] = sh_freq.get(t, 0) + 1
    all_types = set(list(fh_freq.keys()) + list(sh_freq.keys()))
    arc_shift = sum(abs(fh_freq.get(t, 0)/max(len(fh),1) - sh_freq.get(t, 0)/max(len(sh),1))
                    for t in all_types) / max(len(all_types), 1)

    # Setup front-loading
    if freq.get('SETUP', 0) > 0:
        setup_positions = [i/n for i, t in enumerate(labels) if t == 'SETUP']
        setup_front = sum(1 for p in setup_positions if p < 0.25) / len(setup_positions)
    else:
        setup_front = 0

    # Scene cycle (GOAL→ACTION→OBSTACLE→DISASTER)
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


def pearson(a, b):
    n = len(a)
    if n < 5:
        return 0, 0
    ma, mb = sum(a)/n, sum(b)/n
    sa = (sum((v-ma)**2 for v in a)/n)**0.5
    sb = (sum((v-mb)**2 for v in b)/n)**0.5
    if sa == 0 or sb == 0:
        return 0, 0
    r = sum((ai-ma)*(bi-mb) for ai, bi in zip(a, b)) / (n*sa*sb)
    if abs(r) >= 1:
        return r, 999
    t = r * math.sqrt((n-2)/(1-r**2))
    return r, t


def main():
    print("=" * 70)
    print("  PLOT ENTROPY FULL — All 171 movies")
    print("=" * 70)

    # Load ratings
    mv2c = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))
    done = {k: v for k, v in mv2c['done'].items() if v and isinstance(v, dict) and v.get('rating')}

    # Load texts
    print("\n  Loading texts...")
    texts = {}
    mcp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
    for k, v in mcp.get('scripts', {}).items():
        if v and len(v) > 5000:
            texts[k] = v
    fc = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
    for k, v in fc.items():
        if v and len(v) > 5000 and k not in texts:
            texts[k] = v

    movies = []
    for title, info in done.items():
        if title in texts:
            movies.append({'title': title, 'rating': info['rating']})
    movies.sort(key=lambda x: x['rating'])
    print(f"  Movies: {len(movies)}")

    # Load checkpoint
    try:
        checkpoint = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
        print(f"  Checkpoint loaded: {len(checkpoint)} done")
    except:
        checkpoint = {}

    # Process all
    total_api = 0
    t0 = time.time()

    for idx, movie in enumerate(movies):
        title = movie['title']
        rating = movie['rating']

        if title in checkpoint and checkpoint[title].get('labels'):
            continue

        text = texts[title]
        scenes = split_scenes(text)

        if len(scenes) < 5:
            print(f"  [{idx+1}/{len(movies)}] {title} — SKIP (scenes={len(scenes)})")
            checkpoint[title] = {'rating': rating, 'skip': True}
            continue

        if len(scenes) > 60:
            scenes = scenes[:60]

        n_batches = math.ceil(len(scenes) / 10)
        total_api += n_batches

        labels = classify_scenes(scenes, title)

        if len(labels) < 5:
            print(f"  [{idx+1}/{len(movies)}] {title} — SKIP (labels={len(labels)})")
            checkpoint[title] = {'rating': rating, 'skip': True}
            continue

        metrics = compute_metrics(labels)
        if metrics:
            metrics['rating'] = rating
            checkpoint[title] = metrics

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{len(movies)}] {title} ({rating}) — "
              f"{len(labels)} labels, agency={metrics['agency_ratio']:.2f}, "
              f"action={metrics['action_ratio']:.1%} [{elapsed:.0f}s]")

        # Save checkpoint every 10 movies
        if (idx + 1) % 10 == 0:
            json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # Final save
    json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Processing done in {elapsed:.0f}s, API calls: {total_api}")

    # ── Analysis ──
    print(f"\n{'=' * 70}")
    print("  CORRELATION ANALYSIS (n={})".format(
        sum(1 for v in checkpoint.values() if isinstance(v, dict) and v.get('labels'))))
    print(f"{'=' * 70}")

    valid = [(k, v) for k, v in checkpoint.items()
             if isinstance(v, dict) and v.get('labels') and v.get('rating')]
    print(f"  Valid: {len(valid)}")

    metrics_list = [
        'uni_entropy', 'bi_entropy', 'coverage', 'repeat_ratio',
        'agency_ratio', 'disaster_ratio', 'action_ratio', 'emotion_ratio',
        'arc_shift', 'setup_front', 'cycle_density'
    ]

    ratings = [v['rating'] for _, v in valid]

    print(f"\n  {'Metric':>15s} {'r':>8s} {'t':>8s} {'p<0.05':>8s} {'direction':>20s}")
    print(f"  {'-'*65}")

    for metric in metrics_list:
        vals = [v[metric] for _, v in valid]
        r, t = pearson(vals, ratings)
        sig = "YES *" if abs(t) > 1.96 else "no"
        if abs(t) > 2.58:
            sig = "YES **"
        if abs(t) > 3.29:
            sig = "YES ***"
        direction = ""
        if abs(t) > 1.96:
            if r > 0:
                direction = "higher → better"
            else:
                direction = "lower → better"
        print(f"  {metric:>15s} {r:+8.4f} {t:+8.2f} {sig:>8s} {direction:>20s}")

    # ── Quintile analysis for agency_ratio ──
    print(f"\n{'=' * 70}")
    print("  QUINTILE ANALYSIS — agency_ratio")
    print(f"{'=' * 70}")

    valid_sorted = sorted(valid, key=lambda x: x[1]['agency_ratio'])
    qsize = len(valid_sorted) // 5

    for qi in range(5):
        start = qi * qsize
        end = start + qsize if qi < 4 else len(valid_sorted)
        q = valid_sorted[start:end]
        avg_agency = sum(v['agency_ratio'] for _, v in q) / len(q)
        avg_rating = sum(v['rating'] for _, v in q) / len(q)
        print(f"  Q{qi+1}: agency={avg_agency:.3f}  avg_rating={avg_rating:.2f}  (n={len(q)})")

    # ── Quintile for action_ratio ──
    print(f"\n  QUINTILE — action_ratio")
    valid_sorted = sorted(valid, key=lambda x: x[1]['action_ratio'])
    for qi in range(5):
        start = qi * qsize
        end = start + qsize if qi < 4 else len(valid_sorted)
        q = valid_sorted[start:end]
        avg_action = sum(v['action_ratio'] for _, v in q) / len(q)
        avg_rating = sum(v['rating'] for _, v in q) / len(q)
        print(f"  Q{qi+1}: action={avg_action:.1%}  avg_rating={avg_rating:.2f}  (n={len(q)})")

    # ── Quintile for emotion_ratio ──
    print(f"\n  QUINTILE — emotion_ratio")
    valid_sorted = sorted(valid, key=lambda x: x[1]['emotion_ratio'])
    for qi in range(5):
        start = qi * qsize
        end = start + qsize if qi < 4 else len(valid_sorted)
        q = valid_sorted[start:end]
        avg_emo = sum(v['emotion_ratio'] for _, v in q) / len(q)
        avg_rating = sum(v['rating'] for _, v in q) / len(q)
        print(f"  Q{qi+1}: emotion={avg_emo:.1%}  avg_rating={avg_rating:.2f}  (n={len(q)})")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
