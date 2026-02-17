"""
LLM pilot: top 15 vs bottom 15 rated movies.
Compare zlib vs LLM perplexity on the same axes.
Does LLM separate them better?
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

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world"
CHECKPOINT = f"{BASE}/screenplay/llm_pilot_checkpoint.json"


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


def main():
    print("=" * 70)
    print("  LLM PILOT: TOP 15 vs BOTTOM 15")
    print("=" * 70)

    # Load texts
    texts = {}
    try:
        cp = json.load(open(f'{BASE}/screenplay/mass_checkpoint.json', 'r', encoding='utf-8'))
        for k, v in cp.get('scripts', {}).items():
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

    # Load ratings from v2 checkpoint
    v2cp = json.load(open(f'{BASE}/screenplay/mass_v2_checkpoint.json', 'r', encoding='utf-8'))

    # Build list of movies with text + rating
    candidates = []
    for title, info in v2cp['done'].items():
        if (info and info.get('rating') and info.get('votes', 0) >= 1000
                and title in texts):
            candidates.append({
                'title': title,
                'rating': info['rating'],
                'votes': info['votes'],
                'text': texts[title],
            })

    candidates.sort(key=lambda x: x['rating'])
    print(f"\n  Candidates with text + rating: {len(candidates)}")

    # Pick bottom 15 and top 15
    bottom = candidates[:15]
    top = candidates[-15:]
    selected = bottom + top

    print(f"\n  Bottom 15 (rating {bottom[0]['rating']:.1f} - {bottom[-1]['rating']:.1f}):")
    for m in bottom:
        print(f"    r={m['rating']:.1f} v={m['votes']:,d} {m['title']}")

    print(f"\n  Top 15 (rating {top[0]['rating']:.1f} - {top[-1]['rating']:.1f}):")
    for m in top:
        print(f"    r={m['rating']:.1f} v={m['votes']:,d} {m['title']}")

    # ─── Phase 1: zlib baseline ───
    print(f"\n{'=' * 70}")
    print("  PHASE 1: ZLIB BASELINE")
    print(f"{'=' * 70}")

    for m in selected:
        text = m['text']
        # Axis Y: narrative flow (20 windows)
        n_windows = 20
        ws = len(text) // n_windows
        flow = []
        for i in range(n_windows):
            start = i * ws
            end = start + ws if i < n_windows - 1 else len(text)
            flow.append(zlib_entropy(text[start:end]))
        mean_flow = sum(flow) / len(flow)
        std_flow = (sum((v - mean_flow)**2 for v in flow) / len(flow)) ** 0.5
        m['zlib_narr_std'] = std_flow

        # Axis X: char diversity
        chars = parse_dialogue_chars(text)
        char_entropies = []
        for ch, t in chars.items():
            if len(t) < 500:
                continue
            char_entropies.append(zlib_entropy(t))
        if len(char_entropies) >= 3:
            mean_ce = sum(char_entropies) / len(char_entropies)
            m['zlib_char_var'] = sum((v - mean_ce)**2 for v in char_entropies) / len(char_entropies)
        else:
            m['zlib_char_var'] = None

        # Axis Z: direction ratio
        dial, dirn = split_dialogue_direction(text)
        total = len(dial) + len(dirn)
        m['dir_ratio'] = len(dirn) / total if total > 0 else 0.5

    # Report zlib
    top_z = [m for m in top if m.get('zlib_char_var') is not None]
    bot_z = [m for m in bottom if m.get('zlib_char_var') is not None]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n  {'Metric':>20s} {'Top 15':>10s} {'Bot 15':>10s} {'Δ':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'zlib_narr_std':>20s} {avg([m['zlib_narr_std'] for m in top]):10.4f} {avg([m['zlib_narr_std'] for m in bottom]):10.4f} {avg([m['zlib_narr_std'] for m in top]) - avg([m['zlib_narr_std'] for m in bottom]):+10.4f}")
    if top_z and bot_z:
        print(f"  {'zlib_char_var':>20s} {avg([m['zlib_char_var'] for m in top_z]):10.4f} {avg([m['zlib_char_var'] for m in bot_z]):10.4f} {avg([m['zlib_char_var'] for m in top_z]) - avg([m['zlib_char_var'] for m in bot_z]):+10.4f}")
    print(f"  {'dir_ratio':>20s} {avg([m['dir_ratio'] for m in top]):10.4f} {avg([m['dir_ratio'] for m in bottom]):10.4f} {avg([m['dir_ratio'] for m in top]) - avg([m['dir_ratio'] for m in bottom]):+10.4f}")
    print(f"  {'|dev 57.5%|':>20s} {avg([abs(m['dir_ratio']-0.575) for m in top]):10.4f} {avg([abs(m['dir_ratio']-0.575) for m in bottom]):10.4f} {avg([abs(m['dir_ratio']-0.575) for m in top]) - avg([abs(m['dir_ratio']-0.575) for m in bottom]):+10.4f}")

    # ─── Phase 2: LLM perplexity ───
    print(f"\n{'=' * 70}")
    print("  PHASE 2: LLM PERPLEXITY (Qwen2.5-3B)")
    print(f"{'=' * 70}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-3B"
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print("  Model loaded.")

    def compute_perplexity(text, max_tokens=1024):
        """Compute perplexity of a text chunk."""
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_tokens).to('cuda')
        if tokens.shape[1] < 10:
            return None
        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss.item()
        return math.exp(loss)

    cp = load_checkpoint()
    t0 = time.time()

    for idx, m in enumerate(selected):
        title = m['title']
        if title in cp:
            # Restore from checkpoint
            m['llm_narr_std'] = cp[title].get('llm_narr_std')
            m['llm_char_var'] = cp[title].get('llm_char_var')
            m['llm_dial_ppl'] = cp[title].get('llm_dial_ppl')
            m['llm_dir_ppl'] = cp[title].get('llm_dir_ppl')
            print(f"  [{idx+1}/30] {title:40s} (cached)")
            continue

        text = m['text']
        group = "TOP" if m in top else "BOT"

        # Axis Y: LLM narrative flow (20 windows, sample 2K chars each)
        n_windows = 20
        ws = len(text) // n_windows
        flow_ppl = []
        for i in range(n_windows):
            start = i * ws
            end = start + ws if i < n_windows - 1 else len(text)
            chunk = text[start:end]
            # Take middle 2K chars for efficiency
            if len(chunk) > 2000:
                mid = len(chunk) // 2
                chunk = chunk[mid-1000:mid+1000]
            ppl = compute_perplexity(chunk)
            if ppl is not None:
                flow_ppl.append(ppl)

        if len(flow_ppl) >= 10:
            mean_ppl = sum(flow_ppl) / len(flow_ppl)
            m['llm_narr_std'] = (sum((v - mean_ppl)**2 for v in flow_ppl) / len(flow_ppl)) ** 0.5
        else:
            m['llm_narr_std'] = None

        # Axis X: LLM char diversity (perplexity per character)
        chars = parse_dialogue_chars(text)
        char_ppls = []
        for ch, t in sorted(chars.items(), key=lambda x: -len(x[1])):
            if len(t) < 500:
                continue
            # Sample middle 2K
            sample = t[:2000] if len(t) > 2000 else t
            ppl = compute_perplexity(sample)
            if ppl is not None:
                char_ppls.append(ppl)
            if len(char_ppls) >= 8:
                break

        if len(char_ppls) >= 3:
            mean_cp = sum(char_ppls) / len(char_ppls)
            m['llm_char_var'] = sum((v - mean_cp)**2 for v in char_ppls) / len(char_ppls)
        else:
            m['llm_char_var'] = None

        # Bonus: dialogue vs direction perplexity
        dial, dirn = split_dialogue_direction(text)
        dial_sample = dial[:3000] if len(dial) > 3000 else dial
        dir_sample = dirn[:3000] if len(dirn) > 3000 else dirn
        m['llm_dial_ppl'] = compute_perplexity(dial_sample)
        m['llm_dir_ppl'] = compute_perplexity(dir_sample)

        elapsed = time.time() - t0
        print(f"  [{idx+1}/30] {group} {title:35s} r={m['rating']:.1f} "
              f"narr_std={'%.1f' % m['llm_narr_std'] if m['llm_narr_std'] else '?':>6s} "
              f"char_var={'%.1f' % m['llm_char_var'] if m['llm_char_var'] else '?':>6s} "
              f"d_ppl={'%.1f' % m['llm_dial_ppl'] if m['llm_dial_ppl'] else '?':>6s} "
              f"r_ppl={'%.1f' % m['llm_dir_ppl'] if m['llm_dir_ppl'] else '?':>6s} "
              f"({elapsed:.0f}s)")

        # Save checkpoint
        cp[title] = {
            'llm_narr_std': m.get('llm_narr_std'),
            'llm_char_var': m.get('llm_char_var'),
            'llm_dial_ppl': m.get('llm_dial_ppl'),
            'llm_dir_ppl': m.get('llm_dir_ppl'),
        }
        save_checkpoint(cp)

    # ─── Phase 3: Compare ───
    print(f"\n{'=' * 70}")
    print("  PHASE 3: COMPARISON — ZLIB vs LLM")
    print(f"{'=' * 70}")

    top_l = [m for m in top if m.get('llm_char_var') is not None]
    bot_l = [m for m in bottom if m.get('llm_char_var') is not None]

    print(f"\n  {'Metric':>25s} {'Top 15':>10s} {'Bot 15':>10s} {'Δ':>10s} {'Δ%':>8s}")
    print(f"  {'-'*65}")

    metrics = [
        ('zlib_narr_std', [m for m in top], [m for m in bottom]),
        ('llm_narr_std', [m for m in top if m.get('llm_narr_std')], [m for m in bottom if m.get('llm_narr_std')]),
        ('zlib_char_var', top_z, bot_z),
        ('llm_char_var', top_l, bot_l),
        ('dir_ratio', top, bottom),
    ]

    for name, t_list, b_list in metrics:
        if not t_list or not b_list:
            continue
        t_avg = avg([m[name] for m in t_list])
        b_avg = avg([m[name] for m in b_list])
        delta = t_avg - b_avg
        pct = delta / b_avg * 100 if b_avg != 0 else 0
        print(f"  {name:>25s} {t_avg:10.4f} {b_avg:10.4f} {delta:+10.4f} {pct:+8.1f}%")

    # LLM bonus: dialogue vs direction perplexity gap
    top_gap = [m for m in top if m.get('llm_dial_ppl') and m.get('llm_dir_ppl')]
    bot_gap = [m for m in bottom if m.get('llm_dial_ppl') and m.get('llm_dir_ppl')]
    if top_gap and bot_gap:
        t_dial = avg([m['llm_dial_ppl'] for m in top_gap])
        b_dial = avg([m['llm_dial_ppl'] for m in bot_gap])
        t_dir = avg([m['llm_dir_ppl'] for m in top_gap])
        b_dir = avg([m['llm_dir_ppl'] for m in bot_gap])
        t_gap = avg([m['llm_dir_ppl'] - m['llm_dial_ppl'] for m in top_gap])
        b_gap = avg([m['llm_dir_ppl'] - m['llm_dial_ppl'] for m in bot_gap])

        print(f"\n  LLM Perplexity (dialogue vs direction):")
        print(f"  {'':>25s} {'Top 15':>10s} {'Bot 15':>10s}")
        print(f"  {'dial_ppl':>25s} {t_dial:10.1f} {b_dial:10.1f}")
        print(f"  {'dir_ppl':>25s} {t_dir:10.1f} {b_dir:10.1f}")
        print(f"  {'gap (dir-dial)':>25s} {t_gap:+10.1f} {b_gap:+10.1f}")

    # Effect size (Cohen's d)
    print(f"\n  Effect sizes (Cohen's d):")
    for name in ['zlib_narr_std', 'llm_narr_std', 'zlib_char_var', 'llm_char_var', 'dir_ratio']:
        t_vals = [m[name] for m in top if m.get(name) is not None]
        b_vals = [m[name] for m in bottom if m.get(name) is not None]
        if len(t_vals) < 3 or len(b_vals) < 3:
            continue
        t_mean = avg(t_vals)
        b_mean = avg(b_vals)
        pooled_std = ((sum((v-t_mean)**2 for v in t_vals) + sum((v-b_mean)**2 for v in b_vals)) / (len(t_vals) + len(b_vals) - 2)) ** 0.5
        d = (t_mean - b_mean) / pooled_std if pooled_std > 0 else 0
        label = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"    {name:>25s}: d = {d:+.3f} ({label})")

    print(f"\n  Total time: {time.time() - t0:.0f}s")

    # Save results
    results = {
        'top': [{'title': m['title'], 'rating': m['rating'],
                 'zlib_narr_std': m.get('zlib_narr_std'), 'llm_narr_std': m.get('llm_narr_std'),
                 'zlib_char_var': m.get('zlib_char_var'), 'llm_char_var': m.get('llm_char_var'),
                 'dir_ratio': m.get('dir_ratio'),
                 'llm_dial_ppl': m.get('llm_dial_ppl'), 'llm_dir_ppl': m.get('llm_dir_ppl')}
                for m in top],
        'bottom': [{'title': m['title'], 'rating': m['rating'],
                    'zlib_narr_std': m.get('zlib_narr_std'), 'llm_narr_std': m.get('llm_narr_std'),
                    'zlib_char_var': m.get('zlib_char_var'), 'llm_char_var': m.get('llm_char_var'),
                    'dir_ratio': m.get('dir_ratio'),
                    'llm_dial_ppl': m.get('llm_dial_ppl'), 'llm_dir_ppl': m.get('llm_dir_ppl')}
                   for m in bottom],
    }
    with open(f'{BASE}/screenplay/llm_pilot_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: llm_pilot_results.json")


if __name__ == "__main__":
    main()
