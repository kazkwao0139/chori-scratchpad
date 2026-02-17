from pathlib import Path
"""
Full LLM analysis: all 168 movies with text + rating.
Compute LLM-based three axes + ppl gap, then regression.
"""

import json
import sys
import math
import time
import re
import zlib
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CHECKPOINT = f"{BASE}/screenplay/llm_full_checkpoint.json"


def load_checkpoint():
    try:
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(cp):
    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False)


def zlib_entropy(text):
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


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


def pearson(a, b):
    n = len(a)
    if n < 5:
        return 0, 0
    ma = sum(a) / n
    mb = sum(b) / n
    sa = (sum((v - ma)**2 for v in a) / n) ** 0.5
    sb = (sum((v - mb)**2 for v in b) / n) ** 0.5
    if sa == 0 or sb == 0:
        return 0, 0
    cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b)) / n
    r = cov / (sa * sb)
    t = r * math.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 999
    return r, t


def main():
    print("=" * 70)
    print("  LLM FULL ANALYSIS: ALL 168 MOVIES")
    print("=" * 70)

    # Load texts
    texts = {}
    try:
        mcp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
        for k, v in mcp.get('scripts', {}).items():
            if v and len(v) > 5000:
                texts[k] = v
    except Exception:
        pass
    try:
        fc = json.load(open(f'{BASE}/screenplay/narrative_flow_cache.json', 'r', encoding='utf-8'))
        for k, v in fc.items():
            if v and len(v) > 5000 and k not in texts:
                texts[k] = v
    except Exception:
        pass

    # Load ratings
    v2cp = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))

    movies = []
    for title, info in v2cp['done'].items():
        if (info and info.get('rating') and info.get('votes', 0) >= 1000
                and title in texts):
            movies.append({
                'title': title,
                'rating': info['rating'],
                'votes': info['votes'],
                'text': texts[title],
                'zlib_data': info,
            })

    movies.sort(key=lambda x: x['rating'])
    print(f"\n  Movies with text + rating: {len(movies)}")
    print(f"  Rating range: {movies[0]['rating']:.1f} - {movies[-1]['rating']:.1f}")

    # Load model
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
    t0 = time.time()
    total = len(movies)

    for idx, m in enumerate(movies):
        title = m['title']
        if title in cp:
            m.update(cp[title])
            continue

        text = m['text']

        # Axis Y: LLM narrative flow (20 windows)
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

        if len(flow_ppl) >= 10:
            mean_ppl = sum(flow_ppl) / len(flow_ppl)
            m['llm_narr_std'] = (sum((v - mean_ppl)**2 for v in flow_ppl) / len(flow_ppl)) ** 0.5
            m['llm_narr_mean'] = mean_ppl
        else:
            m['llm_narr_std'] = None
            m['llm_narr_mean'] = None

        # Axis X: LLM char diversity
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

        if len(char_ppls) >= 3:
            mean_cp = sum(char_ppls) / len(char_ppls)
            m['llm_char_var'] = sum((v - mean_cp)**2 for v in char_ppls) / len(char_ppls)
            m['llm_char_mean'] = mean_cp
        else:
            m['llm_char_var'] = None
            m['llm_char_mean'] = None

        # Dialogue vs direction perplexity
        dial, dirn = split_dialogue_direction(text)
        dial_sample = dial[:3000] if len(dial) > 3000 else dial
        dir_sample = dirn[:3000] if len(dirn) > 3000 else dirn
        m['llm_dial_ppl'] = compute_perplexity(dial_sample)
        m['llm_dir_ppl'] = compute_perplexity(dir_sample)
        if m['llm_dial_ppl'] and m['llm_dir_ppl']:
            m['llm_ppl_gap'] = m['llm_dir_ppl'] - m['llm_dial_ppl']
        else:
            m['llm_ppl_gap'] = None

        # Axis Z: dir_ratio (same as before, no LLM needed)
        total_len = len(dial) + len(dirn)
        m['dir_ratio'] = len(dirn) / total_len if total_len > 0 else 0.5

        # zlib baselines
        m['zlib_narr_std'] = m['zlib_data'].get('narr_std')
        m['zlib_char_var'] = m['zlib_data'].get('char_var')

        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (total - idx - 1)
        print(f"  [{idx+1}/{total}] r={m['rating']:.1f} {title:35s} "
              f"ns={'%.1f'%m['llm_narr_std'] if m['llm_narr_std'] else '?':>5s} "
              f"cv={'%.1f'%m['llm_char_var'] if m['llm_char_var'] else '?':>5s} "
              f"gap={'%+.1f'%m['llm_ppl_gap'] if m['llm_ppl_gap'] else '?':>5s} "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        # Checkpoint
        cp[title] = {
            'llm_narr_std': m.get('llm_narr_std'),
            'llm_narr_mean': m.get('llm_narr_mean'),
            'llm_char_var': m.get('llm_char_var'),
            'llm_char_mean': m.get('llm_char_mean'),
            'llm_dial_ppl': m.get('llm_dial_ppl'),
            'llm_dir_ppl': m.get('llm_dir_ppl'),
            'llm_ppl_gap': m.get('llm_ppl_gap'),
            'dir_ratio': m.get('dir_ratio'),
            'zlib_narr_std': m.get('zlib_narr_std'),
            'zlib_char_var': m.get('zlib_char_var'),
        }
        if (idx + 1) % 10 == 0:
            save_checkpoint(cp)

    save_checkpoint(cp)

    # ─── Analysis ───
    print(f"\n{'=' * 70}")
    print("  ANALYSIS")
    print(f"{'=' * 70}")

    # Filter complete
    complete = [m for m in movies
                if m.get('llm_narr_std') is not None
                and m.get('llm_char_var') is not None
                and m.get('llm_ppl_gap') is not None]
    print(f"\n  Complete: {len(complete)}")

    ratings = [m['rating'] for m in complete]

    # Correlations
    print(f"\n  PEARSON CORRELATIONS with IMDB rating:")
    print(f"  {'Metric':>25s} {'r':>8s} {'t':>8s} {'sig':>5s}")
    print(f"  {'-'*50}")

    metrics = [
        ('zlib_narr_std', [m['zlib_narr_std'] for m in complete if m.get('zlib_narr_std')]),
        ('llm_narr_std', [m['llm_narr_std'] for m in complete]),
        ('llm_narr_mean', [m['llm_narr_mean'] for m in complete]),
        ('zlib_char_var', [m['zlib_char_var'] for m in complete if m.get('zlib_char_var')]),
        ('llm_char_var', [m['llm_char_var'] for m in complete]),
        ('llm_char_mean', [m['llm_char_mean'] for m in complete]),
        ('dir_ratio', [m['dir_ratio'] for m in complete]),
        ('|dev 57.5%|', [abs(m['dir_ratio'] - 0.575) for m in complete]),
        ('llm_ppl_gap', [m['llm_ppl_gap'] for m in complete]),
        ('llm_dial_ppl', [m['llm_dial_ppl'] for m in complete]),
        ('llm_dir_ppl', [m['llm_dir_ppl'] for m in complete]),
    ]

    for name, vals in metrics:
        if len(vals) != len(ratings):
            r_list = ratings[:len(vals)]
        else:
            r_list = ratings
        r_val, t_val = pearson(vals, r_list)
        sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""
        print(f"  {name:>25s} {r_val:+8.4f} {t_val:+8.2f} {sig:>5s}")

    # ─── Multiple regression with LLM features ───
    print(f"\n{'=' * 70}")
    print("  MULTIPLE REGRESSION (LLM features)")
    print(f"{'=' * 70}")

    # Features: llm_narr_std, llm_char_var, |dev|, llm_ppl_gap
    def standardize(vals):
        mu = sum(vals) / len(vals)
        sigma = (sum((v - mu)**2 for v in vals) / len(vals)) ** 0.5
        return [(v - mu) / sigma if sigma > 0 else 0 for v in vals], mu, sigma

    feats = {
        'llm_narr_std': [m['llm_narr_std'] for m in complete],
        'llm_char_var': [m['llm_char_var'] for m in complete],
        '|dev|': [abs(m['dir_ratio'] - 0.575) for m in complete],
        'llm_ppl_gap': [m['llm_ppl_gap'] for m in complete],
        'llm_dial_ppl': [m['llm_dial_ppl'] for m in complete],
    }

    feat_names = list(feats.keys())
    feat_std = {}
    for name, vals in feats.items():
        feat_std[name], _, _ = standardize(vals)

    n = len(complete)
    k = len(feat_names) + 1  # +1 for intercept

    # Design matrix
    X = [[1] + [feat_std[name][i] for name in feat_names] for i in range(n)]
    y = ratings

    # OLS
    XtX = [[sum(X[i][j]*X[i][l] for i in range(n)) for l in range(k)] for j in range(k)]
    Xty = [sum(X[i][j]*y[i] for i in range(n)) for j in range(k)]

    aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        max_row = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        for j in range(col, k+1):
            aug[col][j] /= pivot
        for row in range(k):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, k+1):
                aug[row][j] -= factor * aug[col][j]

    beta = [aug[i][k] for i in range(k)]
    y_pred = [sum(beta[j] * X[i][j] for j in range(k)) for i in range(n)]
    mr = sum(y) / n
    ss_res = sum((y[i] - y_pred[i])**2 for i in range(n))
    ss_tot = sum((y[i] - mr)**2 for i in range(n))
    r_sq = 1 - ss_res / ss_tot
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k)
    mse = ss_res / (n - k)

    labels = ['intercept'] + feat_names
    print(f"\n  {'Variable':>20s} {'β':>10s}")
    print(f"  {'-'*32}")
    for j in range(k):
        print(f"  {labels[j]:>20s} {beta[j]:+10.4f}")
    print(f"\n  R² = {r_sq:.4f}")
    print(f"  Adjusted R² = {adj_r_sq:.4f}")
    print(f"  RMSE = {mse**0.5:.3f}")
    print(f"  n = {n}")

    # ─── Compare zlib vs LLM regression ───
    print(f"\n{'=' * 70}")
    print("  COMPARISON: ZLIB vs LLM R²")
    print(f"{'=' * 70}")

    # zlib regression (3 features)
    complete_z = [m for m in movies
                  if m.get('zlib_narr_std') is not None
                  and m.get('zlib_char_var') is not None]
    if complete_z:
        z_feats = {
            'zlib_narr_std': [m['zlib_narr_std'] for m in complete_z],
            'zlib_char_var': [m['zlib_char_var'] for m in complete_z],
            '|dev|': [abs(m.get('dir_ratio', 0.575) - 0.575) for m in complete_z],
        }
        z_names = list(z_feats.keys())
        z_std = {}
        for name, vals in z_feats.items():
            z_std[name], _, _ = standardize(vals)

        nz = len(complete_z)
        kz = len(z_names) + 1
        Xz = [[1] + [z_std[name][i] for name in z_names] for i in range(nz)]
        yz = [m['rating'] for m in complete_z]

        XtXz = [[sum(Xz[i][j]*Xz[i][l] for i in range(nz)) for l in range(kz)] for j in range(kz)]
        Xtyz = [sum(Xz[i][j]*yz[i] for i in range(nz)) for j in range(kz)]

        augz = [XtXz[i][:] + [Xtyz[i]] for i in range(kz)]
        for col in range(kz):
            max_row = max(range(col, kz), key=lambda r: abs(augz[r][col]))
            augz[col], augz[max_row] = augz[max_row], augz[col]
            pivot = augz[col][col]
            if abs(pivot) < 1e-12:
                continue
            for j in range(col, kz+1):
                augz[col][j] /= pivot
            for row in range(kz):
                if row == col:
                    continue
                factor = augz[row][col]
                for j in range(col, kz+1):
                    augz[row][j] -= factor * augz[col][j]

        betaz = [augz[i][kz] for i in range(kz)]
        ypz = [sum(betaz[j]*Xz[i][j] for j in range(kz)) for i in range(nz)]
        mrz = sum(yz) / nz
        ss_res_z = sum((yz[i] - ypz[i])**2 for i in range(nz))
        ss_tot_z = sum((yz[i] - mrz)**2 for i in range(nz))
        r_sq_z = 1 - ss_res_z / ss_tot_z

        print(f"\n  ZLIB (3 features, n={nz}):  R² = {r_sq_z:.4f}")
        print(f"  LLM  (5 features, n={n}):   R² = {r_sq:.4f}")
        print(f"  Improvement: {r_sq/r_sq_z:.1f}x" if r_sq_z > 0 else "")

    # ─── Quintile test ───
    print(f"\n{'=' * 70}")
    print("  QUINTILE TEST (by predicted rating)")
    print(f"{'=' * 70}")

    indexed = list(range(n))
    indexed.sort(key=lambda i: y_pred[i])
    qs = 5
    qsize = n // qs

    print(f"\n  {'Quintile':>10s} {'n':>4s} {'pred':>7s} {'actual':>7s} {'Δ':>7s}")
    print(f"  {'-'*40}")
    for q in range(qs):
        start = q * qsize
        end = start + qsize if q < qs - 1 else n
        items = indexed[start:end]
        pred_avg = sum(y_pred[i] for i in items) / len(items)
        act_avg = sum(y[i] for i in items) / len(items)
        print(f"  Q{q+1:>9d} {len(items):4d} {pred_avg:7.2f} {act_avg:7.2f} {act_avg-pred_avg:+7.2f}")

    # Save
    results = []
    for m in complete:
        results.append({
            'title': m['title'],
            'rating': m['rating'],
            'llm_narr_std': m.get('llm_narr_std'),
            'llm_char_var': m.get('llm_char_var'),
            'llm_ppl_gap': m.get('llm_ppl_gap'),
            'llm_dial_ppl': m.get('llm_dial_ppl'),
            'llm_dir_ppl': m.get('llm_dir_ppl'),
            'dir_ratio': m.get('dir_ratio'),
            'zlib_narr_std': m.get('zlib_narr_std'),
            'zlib_char_var': m.get('zlib_char_var'),
        })
    with open(f'{BASE}/screenplay/llm_full_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n': n,
            'r_squared_llm': round(r_sq, 4),
            'r_squared_zlib': round(r_sq_z, 4) if complete_z else None,
            'movies': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: llm_full_results.json")
    print(f"  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
