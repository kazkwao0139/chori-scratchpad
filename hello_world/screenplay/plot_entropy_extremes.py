"""
Download missing extreme-rated screenplays from IMSDB,
run plot entropy classification on all extremes.
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


def fetch_url(url):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except:
        return None


def extract_text(html):
    if not html:
        return None
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return None
    text = m.group(1)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    return text.strip()


def get_imsdb_url(title):
    """Generate IMSDB URL from title."""
    clean = title.replace(', The', '').replace(', A', '').replace(', An', '')
    clean = re.sub(r'[^\w\s]', '', clean).strip()
    slug = clean.replace(' ', '-')
    return f"https://imsdb.com/scripts/{slug}.html"


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
    print("  PLOT ENTROPY — EXTREMES EXPANSION")
    print("  Download missing + classify >=8.0 and <=5.0")
    print("=" * 70)

    # Load ratings
    mv2c = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))
    done = {k: v for k, v in mv2c['done'].items() if v and isinstance(v, dict) and v.get('rating')}

    # Load existing texts
    texts = {}
    mcp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
    for k, v in mcp.get('scripts', {}).items():
        if v and len(v) > 5000:
            texts[k] = v
    fc = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
    for k, v in fc.items():
        if v and len(v) > 5000 and k not in texts:
            texts[k] = v

    # Load checkpoint
    try:
        checkpoint = json.load(open(CHECKPOINT, 'r', encoding='utf-8'))
    except:
        checkpoint = {}

    # Find extremes needing processing
    extremes = []
    for title, info in done.items():
        r = info['rating']
        if r >= 8.0 or r <= 5.0:
            if title in checkpoint and checkpoint[title].get('labels'):
                continue
            if title in checkpoint and checkpoint[title].get('skip'):
                # Retry skipped if we now have text
                if title not in texts:
                    continue
            extremes.append({'title': title, 'rating': r})

    extremes.sort(key=lambda x: (-abs(x['rating'] - 6.5), x['title']))
    print(f"\n  Extremes to process: {len(extremes)}")
    print(f"  Already done: {sum(1 for v in checkpoint.values() if isinstance(v,dict) and v.get('labels'))}")

    # ── Phase 1: Download missing texts from IMSDB ──
    print(f"\n  Phase 1: Downloading missing scripts...")
    downloaded = 0
    dl_failed = []

    for movie in extremes:
        title = movie['title']
        if title in texts:
            continue

        url = get_imsdb_url(title)
        html = fetch_url(url)
        text = extract_text(html) if html else None

        if text and len(text) > 5000:
            texts[title] = text
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"    Downloaded {downloaded}...")
        else:
            # Try alternate URL formats
            for prefix in ['The-', 'A-', 'An-']:
                if title.endswith(', ' + prefix[:-1]):
                    clean_title = prefix[:-1] + ' ' + title[:title.rfind(',')]
                    url2 = get_imsdb_url(clean_title)
                    html = fetch_url(url2)
                    text = extract_text(html) if html else None
                    if text and len(text) > 5000:
                        texts[title] = text
                        downloaded += 1
                        break
            else:
                dl_failed.append(title)

        time.sleep(0.3)  # polite

    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {len(dl_failed)}")

    # ── Phase 2: Classify all extremes ──
    print(f"\n  Phase 2: Classifying scenes...")
    total_api = 0
    processed = 0
    t0 = time.time()

    for movie in extremes:
        title = movie['title']
        rating = movie['rating']

        if title in checkpoint and checkpoint[title].get('labels'):
            continue
        if title not in texts:
            continue

        text = texts[title]
        scenes = split_scenes(text)

        if len(scenes) < 5:
            checkpoint[title] = {'rating': rating, 'skip': True, 'reason': 'too_few_scenes'}
            continue

        if len(scenes) > 60:
            scenes = scenes[:60]

        n_batches = math.ceil(len(scenes) / 10)
        total_api += n_batches

        labels = classify_scenes(scenes, title)

        if len(labels) < 5:
            checkpoint[title] = {'rating': rating, 'skip': True, 'reason': 'too_few_labels'}
            continue

        metrics = compute_metrics(labels)
        if metrics:
            metrics['rating'] = rating
            checkpoint[title] = metrics
            processed += 1

        elapsed = time.time() - t0
        tag = 'HIGH' if rating >= 8.0 else 'LOW'
        print(f"  [{processed}] {tag} {title} ({rating}) — "
              f"{len(labels)} labels, agency={metrics['agency_ratio']:.2f}, "
              f"action={metrics['action_ratio']:.1%} [{elapsed:.0f}s]")

        if processed % 10 == 0:
            json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # Final save
    json.dump(checkpoint, open(CHECKPOINT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Phase 2 done: {processed} new, {total_api} API calls, {elapsed:.0f}s")

    # ── Analysis ──
    print(f"\n{'=' * 70}")
    print("  FINAL ANALYSIS — EXTREMES ONLY")
    print(f"{'=' * 70}")

    all_valid = [(k, v) for k, v in checkpoint.items()
                 if isinstance(v, dict) and v.get('labels') and v.get('rating')]

    high = [(k, v) for k, v in all_valid if v['rating'] >= 8.0]
    low = [(k, v) for k, v in all_valid if v['rating'] <= 5.0]

    print(f"  High (>=8.0): {len(high)}")
    print(f"  Low (<=5.0): {len(low)}")

    if len(high) < 5 or len(low) < 5:
        print("  Not enough data.")
        # Try <=6.0
        low = [(k, v) for k, v in all_valid if v['rating'] <= 6.0]
        print(f"  Low (<=6.0): {len(low)}")

    metrics_names = [
        'uni_entropy', 'bi_entropy', 'coverage', 'repeat_ratio',
        'agency_ratio', 'disaster_ratio', 'action_ratio', 'emotion_ratio',
        'arc_shift', 'setup_front', 'cycle_density'
    ]

    print(f"\n  {'Metric':>15s} {'High':>8s} {'Low':>8s} {'delta':>8s} {'d':>8s} {'effect':>8s}")
    print(f"  {'-'*55}")

    for m in metrics_names:
        hv = [v[m] for _, v in high]
        lv = [v[m] for _, v in low]
        hm = sum(hv) / len(hv)
        lm = sum(lv) / len(lv)
        delta = hm - lm
        h_var = sum((x - hm) ** 2 for x in hv) / max(len(hv) - 1, 1)
        l_var = sum((x - lm) ** 2 for x in lv) / max(len(lv) - 1, 1)
        pooled = ((h_var + l_var) / 2) ** 0.5
        d = delta / pooled if pooled > 0 else 0
        effect = 'LARGE' if abs(d) > 0.8 else 'MED' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else '-'
        print(f"  {m:>15s} {hm:8.4f} {lm:8.4f} {delta:+8.4f} {d:+8.3f} {effect:>8s}")

    sig = sum(1 for m in metrics_names
              if abs(_cohen_d(high, low, m)) > 0.2)
    print(f"\n  |d|>0.2: {sig}/{len(metrics_names)}")

    # Also try within the extremes: correlation
    print(f"\n  Correlation within extremes (high only, n={len(high)}):")
    h_ratings = [v['rating'] for _, v in high]
    for m in metrics_names:
        vals = [v[m] for _, v in high]
        r, t = pearson(vals, h_ratings)
        sig_str = " *" if abs(t) > 1.96 else ""
        print(f"    {m:>15s}: r={r:+.4f} t={t:+.2f}{sig_str}")

    print(f"\n  Done.")


def _cohen_d(high, low, metric):
    hv = [v[metric] for _, v in high]
    lv = [v[metric] for _, v in low]
    hm = sum(hv) / len(hv)
    lm = sum(lv) / len(lv)
    h_var = sum((x - hm) ** 2 for x in hv) / max(len(hv) - 1, 1)
    l_var = sum((x - lm) ** 2 for x in lv) / max(len(lv) - 1, 1)
    pooled = ((h_var + l_var) / 2) ** 0.5
    return (hm - lm) / pooled if pooled > 0 else 0


if __name__ == "__main__":
    main()
