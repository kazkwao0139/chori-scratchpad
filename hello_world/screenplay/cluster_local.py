from pathlib import Path
"""
Local minima hypothesis: formulaic, auteur, dialogue-heavy are DIFFERENT clusters.
Each has its own optimal point. Within-cluster deviation predicts quality better.
K-means on three axes, then within-cluster correlation with rating.
"""

import json
import sys
import math
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


def kmeans(data, k, max_iter=100):
    """Simple k-means on list of [x,y,z] vectors."""
    n = len(data)
    dim = len(data[0])

    # Init: k-means++
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
    for iteration in range(max_iter):
        # Assign
        changed = 0
        for i, p in enumerate(data):
            best = min(range(k), key=lambda c: sum((p[d] - centers[c][d])**2 for d in range(dim)))
            if labels[i] != best:
                changed += 1
            labels[i] = best
        # Update centers
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
    print("  LOCAL MINIMA: CLUSTERING SCREENPLAY TYPES")
    print("=" * 70)

    with open(f'{BASE}/screenplay/mass_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter complete data
    movies = []
    for d in data['rated']:
        if (d.get('char_var') is not None
                and d.get('narr_std') is not None
                and d.get('dir_ratio') is not None
                and d.get('rating') is not None
                and d.get('votes', 0) >= 1000):
            movies.append(d)

    print(f"\n  Total movies: {len(movies)}")

    # Standardize axes
    axes = ['char_var', 'narr_std', 'dir_ratio']
    means = {}
    stds = {}
    for ax in axes:
        vals = [m[ax] for m in movies]
        means[ax] = sum(vals) / len(vals)
        stds[ax] = (sum((v - means[ax])**2 for v in vals) / len(vals)) ** 0.5

    # Standardized vectors
    vectors = []
    for m in movies:
        vectors.append([(m[ax] - means[ax]) / stds[ax] for ax in axes])

    # Try different k values
    random.seed(42)

    for k in [2, 3, 4, 5]:
        print(f"\n{'=' * 70}")
        print(f"  K = {k} CLUSTERS")
        print(f"{'=' * 70}")

        # Run k-means multiple times, pick best
        best_inertia = float('inf')
        best_labels = None
        best_centers = None

        for trial in range(20):
            labels, centers = kmeans(vectors, k)
            inertia = sum(
                sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(3))
                for i in range(len(movies))
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels[:]
                best_centers = [c[:] for c in centers]

        labels = best_labels
        centers = best_centers

        # Analyze each cluster
        print(f"\n  {'Cluster':>10s} {'n':>5s} {'avg_r':>7s} {'min_r':>6s} {'cv':>8s} {'ns':>8s} {'dr':>7s} {'%<6':>6s} {'%≥7.5':>6s}")
        print(f"  {'-'*72}")

        global_r_devs = []
        global_ratings = []

        for c in range(k):
            members = [movies[i] for i in range(len(movies)) if labels[i] == c]
            if not members:
                continue

            ar = sum(m['rating'] for m in members) / len(members)
            mn = min(m['rating'] for m in members)
            acv = sum(m['char_var'] for m in members) / len(members)
            ans = sum(m['narr_std'] for m in members) / len(members)
            adr = sum(m['dir_ratio'] for m in members) / len(members)
            pct_bad = sum(1 for m in members if m['rating'] < 6.0) / len(members)
            pct_good = sum(1 for m in members if m['rating'] >= 7.5) / len(members)

            print(f"  C{c+1:>9d} {len(members):5d} {ar:7.2f} {mn:6.1f} {acv:8.4f} {ans:8.4f} {adr:7.1%} {pct_bad:6.1%} {pct_good:6.1%}")

            # Within-cluster deviation from cluster center
            c_center_raw = {
                'char_var': acv,
                'narr_std': ans,
                'dir_ratio': adr,
            }

            devs = []
            rats = []
            for m in members:
                dev = sum(((m[ax] - c_center_raw[ax]) / stds[ax])**2 for ax in axes) ** 0.5
                devs.append(dev)
                rats.append(m['rating'])
                global_r_devs.append(dev)
                global_ratings.append(m['rating'])

            r_corr, t_corr = pearson(devs, rats)
            print(f"           within-cluster r(dev, rating) = {r_corr:+.4f}  t={t_corr:.2f}")

        # Overall: within-cluster deviation vs rating
        r_all, t_all = pearson(global_r_devs, global_ratings)
        print(f"\n  Combined within-cluster r(dev, rating) = {r_all:+.4f}  t={t_all:.2f}  n={len(global_ratings)}")

        # Compare with global (single-cluster) approach
        global_center = [0, 0, 0]  # standardized mean = 0
        single_devs = [sum(vectors[i][d]**2 for d in range(3))**0.5 for i in range(len(movies))]
        r_single, t_single = pearson(single_devs, [m['rating'] for m in movies])
        print(f"  Single-cluster baseline  r(dev, rating) = {r_single:+.4f}  t={t_single:.2f}")

        # ── Floor test within clusters ──
        print(f"\n  FLOOR TEST per cluster:")
        for c in range(k):
            members_idx = [i for i in range(len(movies)) if labels[i] == c]
            if len(members_idx) < 20:
                continue
            members = [movies[i] for i in members_idx]
            acv = sum(m['char_var'] for m in members) / len(members)
            ans = sum(m['narr_std'] for m in members) / len(members)
            adr = sum(m['dir_ratio'] for m in members) / len(members)

            # Sort by within-cluster deviation
            mem_devs = []
            for m in members:
                dev = sum(((m[ax] - {'char_var': acv, 'narr_std': ans, 'dir_ratio': adr}[ax]) / stds[ax])**2 for ax in axes) ** 0.5
                mem_devs.append((dev, m['rating']))
            mem_devs.sort()

            half = len(mem_devs) // 2
            close = mem_devs[:half]
            far = mem_devs[half:]

            close_avg = sum(r for _, r in close) / len(close)
            far_avg = sum(r for _, r in far) / len(far)
            close_bad = sum(1 for _, r in close if r < 6.0) / len(close)
            far_bad = sum(1 for _, r in far if r < 6.0) / len(far)
            close_good = sum(1 for _, r in close if r >= 7.5) / len(close)
            far_good = sum(1 for _, r in far if r >= 7.5) / len(far)

            print(f"    C{c+1}: close_half avg={close_avg:.2f} %bad={close_bad:.1%} %good={close_good:.1%}"
                  f"  |  far_half avg={far_avg:.2f} %bad={far_bad:.1%} %good={far_good:.1%}"
                  f"  |  Δavg={close_avg-far_avg:+.2f}")

    # ─── Elbow / Silhouette ───
    print(f"\n{'=' * 70}")
    print("  ELBOW: inertia by k")
    print(f"{'=' * 70}")

    for k in range(1, 8):
        best_inertia = float('inf')
        for trial in range(20):
            if k == 1:
                inertia = sum(sum(v**2 for v in vec) for vec in vectors)
                best_inertia = min(best_inertia, inertia)
                break
            labels, centers = kmeans(vectors, k)
            inertia = sum(
                sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(3))
                for i in range(len(movies))
            )
            best_inertia = min(best_inertia, inertia)
        print(f"  k={k}: inertia={best_inertia:.1f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
