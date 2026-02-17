from pathlib import Path
"""
Plot-level entropy pilot: top 15 vs bottom 15 rated movies.
Use Claude Haiku to summarize each scene → measure entropy of event sequence.
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


def call_claude(prompt, system="You are a screenplay analyst.", max_tokens=300):
    """Call Claude API and return text response."""
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
    """Split screenplay into scenes using INT./EXT. markers."""
    # Find scene boundaries
    pattern = re.compile(r'\n\s*((?:INT|EXT|INT\./EXT|I/E)[\.\s].+)', re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if len(matches) < 5:
        # Fallback: split by double newlines into chunks
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


def summarize_scenes_batch(scenes, title):
    """Summarize scenes in batches to reduce API calls."""
    summaries = []
    batch_size = 10  # 10 scenes per API call

    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i+batch_size]
        scene_texts = []
        for j, (header, body) in enumerate(batch):
            # Truncate each scene to ~500 chars to fit context
            truncated = body[:500] + ('...' if len(body) > 500 else '')
            scene_texts.append(f"SCENE {i+j+1} [{header[:60]}]:\n{truncated}")

        prompt = (
            f"Movie: {title}\n\n"
            + "\n---\n".join(scene_texts)
            + "\n\n---\nFor each scene above, write ONE short sentence (max 15 words) "
            "describing the KEY PLOT EVENT that happens. "
            "Focus on WHAT HAPPENS (actions, decisions, revelations), not description.\n"
            "Format: one line per scene, numbered.\n"
            "Example:\n1. Detective discovers the victim was poisoned.\n2. Suspect flees the interrogation room."
        )

        resp = call_claude(prompt, max_tokens=400)
        if resp:
            # Parse numbered lines
            for line in resp.strip().split('\n'):
                line = line.strip()
                m = re.match(r'\d+[\.\)]\s*(.+)', line)
                if m:
                    summaries.append(m.group(1).strip())

    return summaries


def compute_event_entropy(summaries):
    """
    Compute entropy of the event sequence.
    Strategy: use bigram patterns of event types to measure predictability.
    Classify each event into categories, then measure transition entropy.
    """
    if len(summaries) < 5:
        return {}

    # Event type classification by keywords
    categories = {
        'confrontation': ['fight', 'attack', 'confront', 'argue', 'threat', 'kill', 'shoot', 'chase', 'battle'],
        'discovery': ['discover', 'find', 'learn', 'realize', 'reveal', 'uncover', 'notice', 'see'],
        'decision': ['decide', 'choose', 'agree', 'refuse', 'accept', 'reject', 'plan', 'commit'],
        'movement': ['arrive', 'leave', 'travel', 'enter', 'escape', 'flee', 'return', 'go', 'drive', 'walk'],
        'dialogue': ['tell', 'ask', 'explain', 'discuss', 'talk', 'confess', 'warn', 'convince', 'persuade'],
        'emotion': ['cry', 'laugh', 'mourn', 'celebrate', 'fear', 'love', 'kiss', 'embrace', 'grieve'],
        'setup': ['introduce', 'meet', 'begin', 'open', 'establish', 'prepare', 'wake', 'morning'],
        'twist': ['betray', 'surprise', 'shock', 'unexpected', 'turn', 'twist', 'secret', 'lie', 'trap'],
    }

    # Classify each summary
    event_types = []
    for s in summaries:
        s_lower = s.lower()
        best_cat = 'other'
        best_count = 0
        for cat, keywords in categories.items():
            count = sum(1 for kw in keywords if kw in s_lower)
            if count > best_count:
                best_count = count
                best_cat = cat
        event_types.append(best_cat)

    # Unigram entropy (event type distribution)
    n = len(event_types)
    freq = {}
    for t in event_types:
        freq[t] = freq.get(t, 0) + 1
    uni_entropy = -sum((c/n) * math.log2(c/n) for c in freq.values() if c > 0)

    # Bigram transition entropy
    transitions = {}
    for i in range(len(event_types) - 1):
        pair = (event_types[i], event_types[i+1])
        transitions[pair] = transitions.get(pair, 0) + 1
    nt = sum(transitions.values())
    bi_entropy = -sum((c/nt) * math.log2(c/nt) for c in transitions.values() if c > 0) if nt > 0 else 0

    # Repetition ratio (how often same type appears consecutively)
    repeats = sum(1 for i in range(len(event_types)-1) if event_types[i] == event_types[i+1])
    repeat_ratio = repeats / (n - 1) if n > 1 else 0

    # Surprise score: how often does the event type change?
    changes = sum(1 for i in range(len(event_types)-1) if event_types[i] != event_types[i+1])
    change_ratio = changes / (n - 1) if n > 1 else 0

    # Category coverage: how many different event types are used?
    coverage = len(freq) / len(categories)

    # Narrative arc: compare first-half vs second-half type distributions
    mid = n // 2
    first_half = event_types[:mid]
    second_half = event_types[mid:]
    fh_freq = {}
    sh_freq = {}
    for t in first_half:
        fh_freq[t] = fh_freq.get(t, 0) + 1
    for t in second_half:
        sh_freq[t] = sh_freq.get(t, 0) + 1
    all_types = set(list(fh_freq.keys()) + list(sh_freq.keys()))
    arc_shift = sum(abs(fh_freq.get(t, 0)/max(len(first_half),1) - sh_freq.get(t, 0)/max(len(second_half),1))
                    for t in all_types) / max(len(all_types), 1)

    return {
        'n_scenes': n,
        'n_summaries': len(summaries),
        'uni_entropy': round(uni_entropy, 4),
        'bi_entropy': round(bi_entropy, 4),
        'repeat_ratio': round(repeat_ratio, 4),
        'change_ratio': round(change_ratio, 4),
        'coverage': round(coverage, 4),
        'arc_shift': round(arc_shift, 4),
        'event_types': freq,
    }


def main():
    print("=" * 70)
    print("  PLOT ENTROPY PILOT — Top 15 vs Bottom 15")
    print("=" * 70)

    # Load ratings from mass_v2_checkpoint
    mv2c = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))
    done = {k: v for k, v in mv2c['done'].items() if v and isinstance(v, dict) and v.get('rating')}

    # Load screenplay texts
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

    print(f"  Texts available: {len(texts)}")

    # Get rated movies with text
    rated = []
    for title, info in done.items():
        if title in texts:
            rated.append({'title': title, 'rating': info['rating'], 'votes': info.get('votes', 0)})

    rated.sort(key=lambda x: x['rating'])
    print(f"  Rated with text: {len(rated)}")

    # Top 15 and bottom 15
    bottom15 = rated[:15]
    top15 = rated[-15:]

    print(f"\n  Top 15: {top15[0]['title']}({top15[0]['rating']}) ~ {top15[-1]['title']}({top15[-1]['rating']})")
    print(f"  Bottom 15: {bottom15[0]['title']}({bottom15[0]['rating']}) ~ {bottom15[-1]['title']}({bottom15[-1]['rating']})")

    # Process each movie
    results = []
    total_api_calls = 0

    for group_name, group in [("BOTTOM 15", bottom15), ("TOP 15", top15)]:
        print(f"\n{'=' * 70}")
        print(f"  {group_name}")
        print(f"{'=' * 70}")

        for movie in group:
            title = movie['title']
            text = texts[title]
            rating = movie['rating']

            print(f"\n  [{title}] (rating={rating})")

            # Split into scenes
            scenes = split_scenes(text)
            print(f"    Scenes found: {len(scenes)}")

            if len(scenes) < 5:
                print(f"    SKIP — too few scenes")
                continue

            # Cap at 60 scenes (most important ones)
            if len(scenes) > 60:
                scenes = scenes[:60]

            # Summarize via Claude API
            n_batches = math.ceil(len(scenes) / 10)
            total_api_calls += n_batches
            print(f"    API calls needed: {n_batches}")

            summaries = summarize_scenes_batch(scenes, title)
            print(f"    Summaries obtained: {len(summaries)}")

            if len(summaries) < 5:
                print(f"    SKIP — too few summaries")
                continue

            # Compute plot entropy
            metrics = compute_event_entropy(summaries)
            metrics['title'] = title
            metrics['rating'] = rating
            metrics['group'] = group_name
            results.append(metrics)

            print(f"    uni_H={metrics['uni_entropy']:.3f}  bi_H={metrics['bi_entropy']:.3f}  "
                  f"change={metrics['change_ratio']:.3f}  coverage={metrics['coverage']:.2f}  "
                  f"arc_shift={metrics['arc_shift']:.3f}")

    # ── Analysis ──
    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS — total API calls: {total_api_calls}")
    print(f"{'=' * 70}")

    top_results = [r for r in results if r['group'] == 'TOP 15']
    bot_results = [r for r in results if r['group'] == 'BOTTOM 15']

    if not top_results or not bot_results:
        print("  Not enough data for comparison.")
        return

    metrics_to_compare = ['uni_entropy', 'bi_entropy', 'repeat_ratio', 'change_ratio', 'coverage', 'arc_shift']

    print(f"\n  {'Metric':>15s} {'Top15':>8s} {'Bot15':>8s} {'Δ':>8s} {'Cohen_d':>8s}")
    print(f"  {'-'*50}")

    for metric in metrics_to_compare:
        top_vals = [r[metric] for r in top_results]
        bot_vals = [r[metric] for r in bot_results]

        top_mean = sum(top_vals) / len(top_vals)
        bot_mean = sum(bot_vals) / len(bot_vals)
        delta = top_mean - bot_mean

        # Cohen's d
        top_var = sum((v - top_mean)**2 for v in top_vals) / max(len(top_vals)-1, 1)
        bot_var = sum((v - bot_mean)**2 for v in bot_vals) / max(len(bot_vals)-1, 1)
        pooled_sd = ((top_var + bot_var) / 2) ** 0.5
        d = delta / pooled_sd if pooled_sd > 0 else 0

        sig = "<<<" if abs(d) > 0.8 else "<<" if abs(d) > 0.5 else "<" if abs(d) > 0.2 else ""
        print(f"  {metric:>15s} {top_mean:8.4f} {bot_mean:8.4f} {delta:+8.4f} {d:+8.3f} {sig}")

    # Save results
    save_path = f'{BASE}/screenplay/plot_entropy_pilot_results.json'
    json.dump(results, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n  Saved to {save_path}")

    # Event type distribution comparison
    print(f"\n  Event type distribution:")
    top_all_types = {}
    bot_all_types = {}
    for r in top_results:
        for t, c in r['event_types'].items():
            top_all_types[t] = top_all_types.get(t, 0) + c
    for r in bot_results:
        for t, c in r['event_types'].items():
            bot_all_types[t] = bot_all_types.get(t, 0) + c

    top_total = sum(top_all_types.values())
    bot_total = sum(bot_all_types.values())
    all_types = sorted(set(list(top_all_types.keys()) + list(bot_all_types.keys())))

    print(f"\n  {'Type':>15s} {'Top15%':>8s} {'Bot15%':>8s} {'Δ%':>8s}")
    print(f"  {'-'*42}")
    for t in all_types:
        tp = top_all_types.get(t, 0) / top_total * 100 if top_total else 0
        bp = bot_all_types.get(t, 0) / bot_total * 100 if bot_total else 0
        print(f"  {t:>15s} {tp:7.1f}% {bp:7.1f}% {tp-bp:+7.1f}%")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
