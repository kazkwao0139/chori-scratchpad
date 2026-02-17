from pathlib import Path
"""
LLM mass analysis: all ~1000 rated movies.
Download missing texts from IMSDB, run Qwen2.5-3B perplexity,
then cluster analysis with local minima.
"""

import re
import json
import sys
import math
import time
import zlib
import random
import urllib.request
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CHECKPOINT = f"{BASE}/screenplay/llm_mass_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(cp):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False)


def fetch_url(url):
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def extract_full_text(html):
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return ""
    text = m.group(1)
    text = re.sub(r'<b>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_dialogue_direction(full_text):
    lines = full_text.split('\n')
    dial, dirn = [], []
    cur = None
    for line in lines:
        s = line.strip()
        if not s:
            cur = None
            continue
        c = re.sub(r'\(.*?\)', '', s).strip()
        if (c.isupper() and 2 <= len(c) <= 30
                and not c.startswith('INT') and not c.startswith('EXT')
                and not c.startswith('CUT') and not c.startswith('FADE')
                and not c.startswith('CLOSE') and not c.startswith('ANGLE')
                and not c.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", c)):
            cur = c
            continue
        if cur and len(s) > 1 and not s.isupper():
            dial.append(s)
        else:
            dirn.append(s)
            cur = None
    return ' '.join(dial), ' '.join(dirn)


def parse_dialogue_chars(text):
    lines = text.split('\n')
    characters = defaultdict(list)
    cur = None
    for line in lines:
        s = line.strip()
        if not s:
            cur = None
            continue
        c = re.sub(r'\(.*?\)', '', s).strip()
        if (c.isupper() and 2 <= len(c) <= 30
                and not c.startswith('INT') and not c.startswith('EXT')
                and not c.startswith('CUT') and not c.startswith('FADE')
                and not c.startswith('CLOSE') and not c.startswith('ANGLE')
                and not c.startswith('THE ')
                and re.match(r"^[A-Z][A-Z\s\.'-]+$", c)):
            cur = c
            continue
        if cur and len(s) > 1 and not s.isupper():
            characters[cur].append(s)
    return {ch: ' '.join(ls) for ch, ls in characters.items()}


def main():
    print("=" * 70)
    print("  LLM MASS ANALYSIS — ALL RATED MOVIES")
    print("=" * 70)

    # Load existing texts
    texts = {}
    try:
        mcp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
        for k, v in mcp.get('scripts', {}).items():
            if v and len(v) > 5000:
                texts[k] = v
        script_list = mcp.get('script_list', [])
    except Exception:
        script_list = []
    try:
        fc = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
        for k, v in fc.items():
            if v and len(v) > 5000 and k not in texts:
                texts[k] = v
    except Exception:
        pass

    # Load v2 checkpoint for ratings + script URLs
    v2cp = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))

    # Build title -> URL map from script_list
    title_to_url = {}
    for title, url in script_list:
        title_to_url[title] = url

    # Get all rated titles
    rated = {}
    for title, info in v2cp['done'].items():
        if info and info.get('rating') and info.get('votes', 0) >= 1000:
            rated[title] = info

    print(f"\n  Rated movies: {len(rated)}")
    print(f"  Already have text: {len([t for t in rated if t in texts])}")
    print(f"  Need to download: {len([t for t in rated if t not in texts and t in title_to_url])}")

    # Load LLM model
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-3B"
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print("  Model loaded.")

    def compute_perplexity(text, max_tokens=1024):
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_tokens).to('cuda')
        if tokens.shape[1] < 10:
            return None
        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss.item()
        return math.exp(loss)

    cp = load_checkpoint()

    # Also load previous llm_full results
    try:
        prev = json.load(open(f'{BASE}/screenplay/llm_full_checkpoint.json', 'r', encoding='utf-8'))
        for title, data in prev.items():
            if title not in cp and data.get('llm_narr_std') is not None:
                cp[title] = data
        print(f"  Loaded {len(prev)} from previous LLM checkpoint")
    except Exception:
        pass

    t0 = time.time()
    total = len(rated)
    done = 0
    skipped = 0
    downloaded = 0

    for title, info in rated.items():
        done += 1

        # Already processed?
        if title in cp and cp[title].get('llm_narr_std') is not None:
            continue

        # Get text
        if title in texts:
            text = texts[title]
        elif title in title_to_url:
            url = title_to_url[title]
            html = fetch_url(url)
            if not html:
                cp[title] = {'llm_narr_std': None, 'skip': 'fetch_fail'}
                skipped += 1
                continue
            text = extract_full_text(html)
            if len(text) < 5000:
                cp[title] = {'llm_narr_std': None, 'skip': 'short'}
                skipped += 1
                continue
            downloaded += 1
            time.sleep(0.15)
        else:
            cp[title] = {'llm_narr_std': None, 'skip': 'no_url'}
            skipped += 1
            continue

        # LLM analysis
        # Axis Y: narrative flow (20 windows)
        n_windows = 20
        ws = len(text) // n_windows
        flow_ppl = []
        for i in range(n_windows):
            start = i * ws
            end = start + ws if i < n_windows - 1 else len(text)
            chunk = text[start:end]
            if len(chunk) > 2000:
                mid = len(chunk) // 2
                chunk = chunk[mid-1000:mid+1000]
            ppl = compute_perplexity(chunk)
            if ppl is not None:
                flow_ppl.append(ppl)

        llm_narr_std = None
        llm_narr_mean = None
        if len(flow_ppl) >= 10:
            mean_ppl = sum(flow_ppl) / len(flow_ppl)
            llm_narr_std = (sum((v - mean_ppl)**2 for v in flow_ppl) / len(flow_ppl)) ** 0.5
            llm_narr_mean = mean_ppl

        # Axis X: char diversity
        chars = parse_dialogue_chars(text)
        char_ppls = []
        for ch, t in sorted(chars.items(), key=lambda x: -len(x[1])):
            if len(t) < 500:
                continue
            sample = t[:2000]
            ppl = compute_perplexity(sample)
            if ppl is not None:
                char_ppls.append(ppl)
            if len(char_ppls) >= 8:
                break

        llm_char_var = None
        if len(char_ppls) >= 3:
            mean_cp = sum(char_ppls) / len(char_ppls)
            llm_char_var = sum((v - mean_cp)**2 for v in char_ppls) / len(char_ppls)

        # Dialogue vs direction perplexity
        dial, dirn = split_dialogue_direction(text)
        dial_sample = dial[:3000] if len(dial) > 3000 else dial
        dir_sample = dirn[:3000] if len(dirn) > 3000 else dirn
        llm_dial_ppl = compute_perplexity(dial_sample)
        llm_dir_ppl = compute_perplexity(dir_sample)
        llm_ppl_gap = (llm_dir_ppl - llm_dial_ppl) if (llm_dial_ppl and llm_dir_ppl) else None

        # Dir ratio
        total_len = len(dial) + len(dirn)
        dir_ratio = len(dirn) / total_len if total_len > 0 else 0.5

        cp[title] = {
            'llm_narr_std': round(llm_narr_std, 4) if llm_narr_std else None,
            'llm_narr_mean': round(llm_narr_mean, 4) if llm_narr_mean else None,
            'llm_char_var': round(llm_char_var, 4) if llm_char_var else None,
            'llm_dial_ppl': round(llm_dial_ppl, 4) if llm_dial_ppl else None,
            'llm_dir_ppl': round(llm_dir_ppl, 4) if llm_dir_ppl else None,
            'llm_ppl_gap': round(llm_ppl_gap, 4) if llm_ppl_gap else None,
            'dir_ratio': round(dir_ratio, 4),
            'rating': info['rating'],
            'votes': info['votes'],
        }

        elapsed = time.time() - t0
        processed = len([v for v in cp.values() if v.get('llm_narr_std') is not None])
        eta = elapsed / max(done, 1) * (total - done)
        r_str = f"r={info['rating']:.1f}"
        ns_str = f"{llm_narr_std:.1f}" if llm_narr_std else "?"
        dl_str = " DL" if title not in texts else ""

        print(f"  [{done}/{total}] {r_str} {title:35s} ns={ns_str:>5s}{dl_str} "
              f"({elapsed:.0f}s, ETA {eta:.0f}s, ok={processed})", flush=True)

        if done % 20 == 0:
            save_checkpoint(cp)

    save_checkpoint(cp)

    # ─── Cluster analysis ───
    print(f"\n{'=' * 70}")
    print("  CLUSTER ANALYSIS (k=3)")
    print(f"{'=' * 70}")

    complete = []
    for title, info in cp.items():
        if (info.get('llm_narr_std') is not None
                and info.get('llm_char_var') is not None
                and info.get('llm_ppl_gap') is not None
                and info.get('rating') is not None):
            complete.append({**info, 'title': title})

    print(f"\n  Complete movies: {len(complete)}")

    if len(complete) < 50:
        print("  Not enough data!")
        return

    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    # Filter: need all features
    complete = [m for m in complete if all(m.get(f) is not None for f in feat_names)]
    print(f"  With all features: {len(complete)}")

    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in complete]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    vectors = [[(m[f] - means[f]) / stds[f] if stds[f] > 0 else 0 for f in feat_names] for m in complete]

    def kmeans(data, k, max_iter=100):
        n = len(data)
        dim = len(data[0])
        centers = [data[random.randint(0, n-1)][:]]
        for _ in range(k - 1):
            dists = []
            for p in data:
                min_d = min(sum((p[d] - c[d])**2 for d in range(dim)) for c in centers)
                dists.append(min_d)
            total = sum(dists)
            if total == 0:
                centers.append(data[random.randint(0, n-1)][:])
                continue
            r = random.random() * total
            cum = 0
            for i, d in enumerate(dists):
                cum += d
                if cum >= r:
                    centers.append(data[i][:])
                    break
        labels = [0] * n
        for _ in range(max_iter):
            changed = 0
            for i, p in enumerate(data):
                best = min(range(k), key=lambda c: sum((p[d] - centers[c][d])**2 for d in range(dim)))
                if labels[i] != best:
                    changed += 1
                labels[i] = best
            for c in range(k):
                members = [data[i] for i in range(n) if labels[i] == c]
                if members:
                    centers[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
            if changed == 0:
                break
        return labels, centers

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
        t = r * math.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 999
        return r, t

    random.seed(42)

    best_inertia = float('inf')
    best_labels = None
    best_centers = None
    for trial in range(50):
        labels, centers = kmeans(vectors, 3)
        inertia = sum(sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names))) for i in range(len(complete)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]
            best_centers = [c[:] for c in centers]

    labels = best_labels

    # Sort clusters by dir_ratio
    cluster_info = []
    for c in range(3):
        members = [complete[i] for i in range(len(complete)) if labels[i] == c]
        if members:
            avg_dr = sum(m['dir_ratio'] for m in members) / len(members)
            cluster_info.append((c, avg_dr, members))
    cluster_info.sort(key=lambda x: x[1])

    names = ['A: 대사형', 'B: 균형형', 'C: 시각형']

    for ci, (c, avg_dr, members) in enumerate(cluster_info):
        n = len(members)
        avg_r = sum(m['rating'] for m in members) / n
        raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}

        # Compute distance
        for m in members:
            m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
        members.sort(key=lambda m: m['_dist'])

        half = n // 2
        close = members[:half]
        far = members[half:]
        close_avg = sum(m['rating'] for m in close) / len(close)
        far_avg = sum(m['rating'] for m in far) / len(far)

        # Within-cluster correlation
        devs = [m['_dist'] for m in members]
        rats = [m['rating'] for m in members]
        r_val, t_val = pearson(devs, rats)
        sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""

        print(f"\n  {names[ci]} (n={n}, avg_r={avg_r:.2f})")
        print(f"    Center: dr={raw_center['dir_ratio']:.1%}  narr={raw_center['llm_narr_std']:.1f}  "
              f"cvar={raw_center['llm_char_var']:.1f}  gap={raw_center['llm_ppl_gap']:+.1f}")
        print(f"    Close half: avg={close_avg:.2f}  |  Far half: avg={far_avg:.2f}  |  Δ={close_avg-far_avg:+.2f}")
        print(f"    Within r(dev, rating) = {r_val:+.4f}  t={t_val:+.2f} {sig}")

        # Top 5 closest
        print(f"    Center movies:")
        for m in members[:5]:
            print(f"      d={m['_dist']:.2f} r={m['rating']:.1f} {m['title']}")

        # Stats
        close_bad = sum(1 for m in close if m['rating'] < 6.0) / len(close)
        far_bad = sum(1 for m in far if m['rating'] < 6.0) / len(far)
        close_good = sum(1 for m in close if m['rating'] >= 8.0) / len(close)
        far_good = sum(1 for m in far if m['rating'] >= 8.0) / len(far)
        print(f"    Close: %<6={close_bad:.1%}  %≥8={close_good:.1%}")
        print(f"    Far:   %<6={far_bad:.1%}  %≥8={far_good:.1%}")

    # Save
    results = {
        'n': len(complete),
        'clusters': []
    }
    for ci, (c, avg_dr, members) in enumerate(cluster_info):
        n = len(members)
        half = n // 2
        close = members[:half]
        far = members[half:]
        results['clusters'].append({
            'name': names[ci],
            'n': n,
            'avg_rating': round(sum(m['rating'] for m in members) / n, 2),
            'close_avg': round(sum(m['rating'] for m in close) / len(close), 2),
            'far_avg': round(sum(m['rating'] for m in far) / len(far), 2),
            'center_dr': round(sum(m['dir_ratio'] for m in members) / n, 3),
        })

    with open(f'{BASE}/screenplay/llm_mass_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Saved: llm_mass_results.json")


if __name__ == "__main__":
    main()
