from pathlib import Path
"""
Cluster analysis on n=845 LLM-analyzed movies.
"""

import json
import sys
import math
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


def kmeans(data, k, max_iter=100):
    n = len(data)
    dim = len(data[0])
    centers = [data[random.randint(0, n-1)][:]]
    for _ in range(k - 1):
        dists = [min(sum((p[d] - c[d])**2 for d in range(dim)) for c in centers) for p in data]
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


def main():
    print("=" * 70)
    print("  CLUSTER ANALYSIS — n=845 LLM MOVIES")
    print("=" * 70)

    cp = json.load(open(f'{BASE}/screenplay/llm_mass_checkpoint.json', 'r', encoding='utf-8'))

    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    complete = []
    for title, info in cp.items():
        if all(info.get(f) is not None for f in feat_names) and info.get('rating') is not None:
            complete.append({**info, 'title': title})

    print(f"\n  Complete: {len(complete)}")

    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in complete]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    vectors = [[(m[f] - means[f]) / stds[f] if stds[f] > 0 else 0 for f in feat_names] for m in complete]

    random.seed(42)

    for k in [3]:
        print(f"\n{'=' * 70}")
        print(f"  K = {k}")
        print(f"{'=' * 70}")

        best_inertia = float('inf')
        best_labels = None
        for trial in range(50):
            labels, centers = kmeans(vectors, k)
            inertia = sum(sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names))) for i in range(len(complete)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels[:]

        labels = best_labels

        # Sort clusters by dir_ratio
        cluster_groups = []
        for c in range(k):
            members = [complete[i] for i in range(len(complete)) if labels[i] == c]
            if not members:
                continue
            avg_dr = sum(m['dir_ratio'] for m in members) / len(members)
            cluster_groups.append((c, avg_dr, members))
        cluster_groups.sort(key=lambda x: x[1])

        names = ['A: 대사형', 'B: 균형형', 'C: 시각형']

        for ci, (c, avg_dr, members) in enumerate(cluster_groups):
            n = len(members)
            avg_r = sum(m['rating'] for m in members) / n
            raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}

            for m in members:
                m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
            members.sort(key=lambda m: m['_dist'])

            half = max(1, n // 2)
            close = members[:half]
            far = members[half:]

            close_avg = sum(m['rating'] for m in close) / len(close) if close else 0
            far_avg = sum(m['rating'] for m in far) / len(far) if far else 0

            devs = [m['_dist'] for m in members]
            rats = [m['rating'] for m in members]
            r_val, t_val = pearson(devs, rats)
            sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""

            name = names[ci] if ci < len(names) else f"C{ci+1}"
            print(f"\n  {name} (n={n}, avg_r={avg_r:.2f})")
            print(f"    Center: dr={raw_center['dir_ratio']:.1%}  narr={raw_center['llm_narr_std']:.1f}  "
                  f"cvar={raw_center['llm_char_var']:.1f}  gap={raw_center['llm_ppl_gap']:+.1f}  dppl={raw_center['llm_dial_ppl']:.1f}")
            print(f"    Close half (n={len(close)}): avg={close_avg:.2f}")
            print(f"    Far half  (n={len(far)}):  avg={far_avg:.2f}")
            print(f"    Δ = {close_avg - far_avg:+.2f}")
            print(f"    Within r(dev, rating) = {r_val:+.4f}  t={t_val:+.2f} {sig}")

            # Top 5 center movies
            print(f"    Center movies:")
            for m in members[:5]:
                print(f"      d={m['_dist']:.2f} r={m['rating']:.1f} {m['title']}")

            # Worst and best in cluster
            by_r = sorted(members, key=lambda m: m['rating'])
            bot3 = ', '.join(m['title'] + '(' + str(m['rating']) + ')' for m in by_r[:3])
            top3 = ', '.join(m['title'] + '(' + str(m['rating']) + ')' for m in by_r[-3:])
            print(f"    Worst: {bot3}")
            print(f"    Best:  {top3}")

            # Percentages
            close_bad = sum(1 for m in close if m['rating'] < 6.0) / len(close) if close else 0
            far_bad = sum(1 for m in far if m['rating'] < 6.0) / len(far) if far else 0
            close_good = sum(1 for m in close if m['rating'] >= 8.0) / len(close) if close else 0
            far_good = sum(1 for m in far if m['rating'] >= 8.0) / len(far) if far else 0
            print(f"    Close: %<6.0={close_bad:.1%}  %≥8.0={close_good:.1%}")
            print(f"    Far:   %<6.0={far_bad:.1%}  %≥8.0={far_good:.1%}")

    # ── Global comparison ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Type':>15s} {'n':>5s} {'avg_r':>6s} {'close':>6s} {'far':>6s} {'Δ':>6s} {'r(dev)':>8s} {'t':>6s} {'sig':>4s}")
    print(f"  {'-'*60}")
    for ci, (c, avg_dr, members) in enumerate(cluster_groups):
        n = len(members)
        avg_r = sum(m['rating'] for m in members) / n
        half = max(1, n // 2)
        close = members[:half]
        far = members[half:]
        close_avg = sum(m['rating'] for m in close) / len(close) if close else 0
        far_avg = sum(m['rating'] for m in far) / len(far) if far else 0
        devs = [m['_dist'] for m in members]
        rats = [m['rating'] for m in members]
        r_val, t_val = pearson(devs, rats)
        sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""
        name = names[ci] if ci < len(names) else f"C{ci+1}"
        print(f"  {name:>15s} {n:5d} {avg_r:6.2f} {close_avg:6.2f} {far_avg:6.2f} {close_avg-far_avg:+6.2f} {r_val:+8.4f} {t_val:+6.2f} {sig:>4s}")


if __name__ == "__main__":
    main()
