from pathlib import Path
"""
Local minima with LLM features.
Cluster on LLM perplexity features, then check WITHIN-cluster
deviation vs rating correlation.
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
        return 0, 0, n
    ma, mb = sum(a)/n, sum(b)/n
    sa = (sum((v-ma)**2 for v in a)/n)**0.5
    sb = (sum((v-mb)**2 for v in b)/n)**0.5
    if sa == 0 or sb == 0:
        return 0, 0, n
    r = sum((ai-ma)*(bi-mb) for ai, bi in zip(a, b)) / (n*sa*sb)
    t = r * math.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else 999
    return r, t, n


def main():
    print("=" * 70)
    print("  LOCAL MINIMA: LLM-BASED CLUSTERING")
    print("=" * 70)

    # Load LLM results
    with open(f'{BASE}/screenplay/llm_full_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    movies = [m for m in data['movies']
              if m.get('llm_narr_std') is not None
              and m.get('llm_char_var') is not None
              and m.get('llm_ppl_gap') is not None
              and m.get('llm_dial_ppl') is not None
              and m.get('llm_dir_ppl') is not None]

    print(f"\n  Movies: {len(movies)}")

    # Features for clustering
    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    # Standardize
    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in movies]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    vectors = [[(m[f] - means[f]) / stds[f] if stds[f] > 0 else 0 for f in feat_names] for m in movies]

    random.seed(42)

    for k in [2, 3, 4]:
        print(f"\n{'=' * 70}")
        print(f"  K = {k}")
        print(f"{'=' * 70}")

        best_inertia = float('inf')
        best_labels = None
        best_centers = None
        for trial in range(30):
            labels, centers = kmeans(vectors, k)
            inertia = sum(sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names))) for i in range(len(movies)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels[:]
                best_centers = [c[:] for c in centers]

        labels = best_labels
        centers = best_centers

        # Per-cluster analysis
        print(f"\n  {'C':>3s} {'n':>4s} {'avg_r':>6s} {'narr':>6s} {'cvar':>6s} {'gap':>7s} {'dppl':>6s} {'dr':>6s} | within r(dev,rat)")
        print(f"  {'-'*75}")

        all_devs = []
        all_rats = []

        for c in range(k):
            idx = [i for i in range(len(movies)) if labels[i] == c]
            if not idx:
                continue
            members = [movies[i] for i in idx]
            n = len(members)
            ar = sum(m['rating'] for m in members) / n

            # Cluster center in raw space
            raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}

            # Print cluster profile
            print(f"  {c+1:>3d} {n:4d} {ar:6.2f} "
                  f"{raw_center['llm_narr_std']:6.1f} "
                  f"{raw_center['llm_char_var']:6.1f} "
                  f"{raw_center['llm_ppl_gap']:+7.1f} "
                  f"{raw_center['llm_dial_ppl']:6.1f} "
                  f"{raw_center['dir_ratio']:6.1%}", end="")

            # Within-cluster deviation vs rating
            devs = []
            rats = []
            for m in members:
                dev = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
                devs.append(dev)
                rats.append(m['rating'])
                all_devs.append(dev)
                all_rats.append(m['rating'])

            r_val, t_val, _ = pearson(devs, rats)
            sig = "***" if abs(t_val) > 3.29 else "**" if abs(t_val) > 2.58 else "*" if abs(t_val) > 1.96 else ""
            print(f" | r={r_val:+.3f} t={t_val:+.2f} {sig}")

            # Show some movies in this cluster
            members_sorted = sorted(members, key=lambda m: m['rating'])
            print(f"       worst: {members_sorted[0]['title']} (r={members_sorted[0]['rating']:.1f})")
            print(f"       best:  {members_sorted[-1]['title']} (r={members_sorted[-1]['rating']:.1f})")
            # Most typical (closest to center)
            closest = min(members, key=lambda m: sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names))
            print(f"       typical: {closest['title']} (r={closest['rating']:.1f})")

        # Combined within-cluster
        r_all, t_all, _ = pearson(all_devs, all_rats)
        print(f"\n  Combined within-cluster: r={r_all:+.4f} t={t_all:+.2f}")

        # ── Per-cluster: close half vs far half ──
        print(f"\n  CLOSE vs FAR half per cluster:")
        print(f"  {'C':>3s} {'close_avg':>10s} {'far_avg':>10s} {'Δ':>8s} {'close_%<6':>10s} {'far_%<6':>10s} {'close_%≥8':>10s} {'far_%≥8':>10s}")
        print(f"  {'-'*75}")

        for c in range(k):
            idx = [i for i in range(len(movies)) if labels[i] == c]
            if len(idx) < 10:
                continue
            members = [movies[i] for i in idx]
            raw_center = {f: sum(m[f] for m in members) / len(members) for f in feat_names}

            mem_devs = []
            for m in members:
                dev = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
                mem_devs.append((dev, m))
            mem_devs.sort(key=lambda x: x[0])

            half = len(mem_devs) // 2
            close = [m for _, m in mem_devs[:half]]
            far = [m for _, m in mem_devs[half:]]

            ca = sum(m['rating'] for m in close) / len(close)
            fa = sum(m['rating'] for m in far) / len(far)
            cb = sum(1 for m in close if m['rating'] < 6.0) / len(close)
            fb = sum(1 for m in far if m['rating'] < 6.0) / len(far)
            cg = sum(1 for m in close if m['rating'] >= 8.0) / len(close)
            fg = sum(1 for m in far if m['rating'] >= 8.0) / len(far)

            print(f"  {c+1:>3d} {ca:10.2f} {fa:10.2f} {ca-fa:+8.2f} {cb:10.1%} {fb:10.1%} {cg:10.1%} {fg:10.1%}")

    # ── Individual feature within-cluster correlations ──
    print(f"\n{'=' * 70}")
    print("  PER-FEATURE WITHIN-CLUSTER CORRELATION (k=3)")
    print(f"{'=' * 70}")

    best_labels_k3 = best_labels  # reuse last k=4... let me redo k=3
    # Redo k=3
    best_inertia = float('inf')
    for trial in range(30):
        labels, centers = kmeans(vectors, 3)
        inertia = sum(sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names))) for i in range(len(movies)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels[:]
            best_centers = [c[:] for c in centers]

    labels = best_labels

    for f in feat_names + ['rating']:
        print(f"\n  {f}:")
        for c in range(3):
            members = [movies[i] for i in range(len(movies)) if labels[i] == c]
            if len(members) < 5:
                continue
            vals = [m[f] if f != 'rating' else m['rating'] for m in members]
            avg_v = sum(vals) / len(vals)
            std_v = (sum((v - avg_v)**2 for v in vals) / len(vals)) ** 0.5
            print(f"    C{c+1}: n={len(members):3d}  avg={avg_v:8.2f}  std={std_v:8.2f}")

    # ── What actually distinguishes the clusters? ──
    print(f"\n{'=' * 70}")
    print("  CLUSTER CHARACTERIZATION (k=3)")
    print(f"{'=' * 70}")

    for c in range(3):
        idx = [i for i in range(len(movies)) if labels[i] == c]
        members = [movies[i] for i in idx]
        if not members:
            continue
        print(f"\n  CLUSTER {c+1} (n={len(members)}):")
        ratings = sorted([m['rating'] for m in members])
        print(f"    Rating: {ratings[0]:.1f} - {ratings[-1]:.1f}, median={ratings[len(ratings)//2]:.1f}")

        # Top 5 and bottom 5
        by_r = sorted(members, key=lambda m: m['rating'])
        bot3 = ', '.join(m['title'] + '(' + str(m['rating']) + ')' for m in by_r[:3])
        top3 = ', '.join(m['title'] + '(' + str(m['rating']) + ')' for m in by_r[-3:])
        print(f"    Bottom 3: {bot3}")
        print(f"    Top 3:    {top3}")

        # Key distinguishing features
        for f in feat_names:
            vals = [m[f] for m in members]
            avg_v = sum(vals) / len(vals)
            all_vals = [m[f] for m in movies]
            all_avg = sum(all_vals) / len(all_vals)
            all_std = (sum((v - all_avg)**2 for v in all_vals) / len(all_vals)) ** 0.5
            z = (avg_v - all_avg) / all_std if all_std > 0 else 0
            print(f"    {f:>20s}: {avg_v:8.2f} (z={z:+.2f})")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
