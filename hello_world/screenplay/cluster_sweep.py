from pathlib import Path
"""
Sweep k=2..15 on n=845 LLM movies.
Elbow, silhouette, and within-cluster Δ for each k.
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
    # k-means++ init
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


def silhouette(data, labels, k):
    """Average silhouette score. Subsample if too large."""
    n = len(data)
    dim = len(data[0])

    # For speed, subsample if n > 500
    if n > 500:
        idx = random.sample(range(n), 500)
        sub_data = [data[i] for i in idx]
        sub_labels = [labels[i] for i in idx]
    else:
        sub_data = data
        sub_labels = labels
        idx = list(range(n))

    sn = len(sub_data)
    scores = []

    # Precompute cluster membership
    cluster_members = {}
    for i in range(sn):
        c = sub_labels[i]
        if c not in cluster_members:
            cluster_members[c] = []
        cluster_members[c].append(i)

    for i in range(sn):
        ci = sub_labels[i]
        # a(i): avg distance to same cluster
        same = cluster_members[ci]
        if len(same) <= 1:
            scores.append(0)
            continue
        a_i = sum(sum((sub_data[i][d] - sub_data[j][d])**2 for d in range(dim))**0.5
                  for j in same if j != i) / (len(same) - 1)

        # b(i): min avg distance to other clusters
        b_i = float('inf')
        for ck in cluster_members:
            if ck == ci:
                continue
            others = cluster_members[ck]
            if not others:
                continue
            avg_d = sum(sum((sub_data[i][d] - sub_data[j][d])**2 for d in range(dim))**0.5
                        for j in others) / len(others)
            if avg_d < b_i:
                b_i = avg_d

        if b_i == float('inf'):
            scores.append(0)
        else:
            scores.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0)

    return sum(scores) / len(scores)


def main():
    print("=" * 70)
    print("  CLUSTER SWEEP — k=2..15, n=845 LLM MOVIES")
    print("=" * 70)

    cp = json.load(open(f'{BASE}/screenplay/llm_mass_checkpoint.json', 'r', encoding='utf-8'))

    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    complete = []
    for title, info in cp.items():
        if all(info.get(f) is not None for f in feat_names) and info.get('rating') is not None:
            complete.append({**info, 'title': title})

    print(f"\n  Complete: {len(complete)}")

    # Standardize
    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in complete]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    vectors = [[(m[f] - means[f]) / stds[f] if stds[f] > 0 else 0 for f in feat_names] for m in complete]

    random.seed(42)

    # ── Sweep k=2..15 ──
    results = []

    for k in range(2, 16):
        best_inertia = float('inf')
        best_labels = None
        for trial in range(30):
            labels, centers = kmeans(vectors, k)
            inertia = sum(
                sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names)))
                for i in range(len(complete))
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels[:]

        labels = best_labels

        # Silhouette
        sil = silhouette(vectors, labels, k)

        # Per-cluster analysis
        cluster_deltas = []
        cluster_corrs = []
        sig_count = 0

        for c in range(k):
            members_idx = [i for i in range(len(complete)) if labels[i] == c]
            if len(members_idx) < 10:
                continue
            members = [complete[i] for i in members_idx]
            n = len(members)
            raw_center = {f: sum(m[f] for m in members) / n for f in feat_names}

            for m in members:
                m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5
            members.sort(key=lambda m: m['_dist'])

            half = max(1, n // 2)
            close = members[:half]
            far = members[half:]
            close_avg = sum(m['rating'] for m in close) / len(close)
            far_avg = sum(m['rating'] for m in far) / len(far) if far else close_avg
            delta = close_avg - far_avg
            cluster_deltas.append(delta)

            devs = [m['_dist'] for m in members]
            rats = [m['rating'] for m in members]
            r_val, t_val = pearson(devs, rats)
            cluster_corrs.append((r_val, t_val, n))
            if abs(t_val) > 1.96:
                sig_count += 1

        # Best delta in any cluster
        max_delta = max(cluster_deltas) if cluster_deltas else 0
        min_delta = min(cluster_deltas) if cluster_deltas else 0
        avg_delta = sum(cluster_deltas) / len(cluster_deltas) if cluster_deltas else 0

        # Best correlation
        best_r = max(cluster_corrs, key=lambda x: abs(x[0])) if cluster_corrs else (0, 0, 0)

        results.append({
            'k': k, 'inertia': best_inertia, 'sil': sil,
            'max_delta': max_delta, 'min_delta': min_delta, 'avg_delta': avg_delta,
            'sig_count': sig_count, 'n_clusters_valid': len(cluster_deltas),
            'best_r': best_r[0], 'best_t': best_r[1], 'best_n': best_r[2],
            'deltas': cluster_deltas, 'corrs': cluster_corrs
        })

        print(f"\n  k={k:2d}  inertia={best_inertia:7.0f}  sil={sil:.3f}  "
              f"max_Δ={max_delta:+.3f}  min_Δ={min_delta:+.3f}  "
              f"sig={sig_count}/{len(cluster_deltas)}  best_r={best_r[0]:+.3f}(t={best_r[1]:+.2f},n={best_r[2]})")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'k':>3s} {'inertia':>8s} {'Δinertia':>9s} {'sil':>6s} {'max_Δ':>7s} {'min_Δ':>7s} {'avg_Δ':>7s} {'sig':>5s} {'best_r':>7s}")
    print(f"  {'-'*65}")

    for i, r in enumerate(results):
        d_inertia = results[i-1]['inertia'] - r['inertia'] if i > 0 else 0
        print(f"  {r['k']:3d} {r['inertia']:8.0f} {d_inertia:+9.0f} {r['sil']:6.3f} "
              f"{r['max_delta']:+7.3f} {r['min_delta']:+7.3f} {r['avg_delta']:+7.3f} "
              f"{r['sig_count']:2d}/{r['n_clusters_valid']:<2d} {r['best_r']:+7.3f}")

    # ── Elbow: find biggest drop ratio ──
    print(f"\n  Elbow analysis (inertia drop ratio):")
    for i in range(1, len(results)):
        prev = results[i-1]['inertia']
        curr = results[i]['inertia']
        ratio = (prev - curr) / prev * 100
        bar = "█" * int(ratio * 2)
        print(f"    k={results[i]['k']:2d}: -{ratio:.1f}% {bar}")

    # ── Any significant cluster at any k? ──
    print(f"\n{'=' * 70}")
    print("  SIGNIFICANT CLUSTERS (|t| > 1.96)")
    print(f"{'=' * 70}")

    any_sig = False
    for r in results:
        for ci, (rv, tv, cn) in enumerate(r['corrs']):
            if abs(tv) > 1.96:
                any_sig = True
                sig = "***" if abs(tv) > 3.29 else "**" if abs(tv) > 2.58 else "*"
                print(f"  k={r['k']:2d} C{ci+1}: r={rv:+.4f} t={tv:+.2f} {sig} (n={cn})")

    if not any_sig:
        print("  NONE.")

    # ── Detail for best silhouette k ──
    best_sil_k = max(results, key=lambda r: r['sil'])
    print(f"\n{'=' * 70}")
    print(f"  BEST SILHOUETTE: k={best_sil_k['k']} (sil={best_sil_k['sil']:.3f})")
    print(f"{'=' * 70}")
    print(f"  Deltas per cluster: {[f'{d:+.3f}' for d in best_sil_k['deltas']]}")
    print(f"  Correlations: {[f'r={rv:+.3f}(t={tv:+.2f},n={cn})' for rv, tv, cn in best_sil_k['corrs']]}")

    # ── Best max_delta k ──
    best_delta_k = max(results, key=lambda r: r['max_delta'])
    print(f"\n  BEST MAX Δ: k={best_delta_k['k']} (max_Δ={best_delta_k['max_delta']:+.3f})")
    print(f"  Deltas per cluster: {[f'{d:+.3f}' for d in best_delta_k['deltas']]}")
    print(f"  Correlations: {[f'r={rv:+.3f}(t={tv:+.2f},n={cn})' for rv, tv, cn in best_delta_k['corrs']]}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
