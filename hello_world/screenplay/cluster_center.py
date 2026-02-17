from pathlib import Path
"""
Show the CENTER movies of each cluster — the archetypes.
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


def main():
    with open(f'{BASE}/screenplay/llm_full_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    movies = [m for m in data['movies']
              if m.get('llm_narr_std') is not None
              and m.get('llm_char_var') is not None
              and m.get('llm_ppl_gap') is not None
              and m.get('llm_dial_ppl') is not None
              and m.get('llm_dir_ppl') is not None]

    feat_names = ['llm_narr_std', 'llm_char_var', 'llm_ppl_gap', 'llm_dial_ppl', 'dir_ratio']

    means = {}
    stds = {}
    for f in feat_names:
        vals = [m[f] for m in movies]
        means[f] = sum(vals) / len(vals)
        stds[f] = (sum((v - means[f])**2 for v in vals) / len(vals)) ** 0.5

    vectors = [[(m[f] - means[f]) / stds[f] if stds[f] > 0 else 0 for f in feat_names] for m in movies]

    random.seed(42)

    for k in [3]:
        best_inertia = float('inf')
        best_labels = None
        best_centers = None
        for trial in range(50):
            labels, centers = kmeans(vectors, k)
            inertia = sum(sum((vectors[i][d] - centers[labels[i]][d])**2 for d in range(len(feat_names))) for i in range(len(movies)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels[:]
                best_centers = [c[:] for c in centers]

        labels = best_labels
        centers = best_centers

        # Sort clusters by dir_ratio for consistent labeling
        cluster_dr = []
        for c in range(k):
            members = [movies[i] for i in range(len(movies)) if labels[i] == c]
            avg_dr = sum(m['dir_ratio'] for m in members) / len(members) if members else 0
            cluster_dr.append((c, avg_dr))
        cluster_dr.sort(key=lambda x: x[1])
        cluster_names = {cluster_dr[0][0]: 'A: 대사형', cluster_dr[1][0]: 'B: 균형형', cluster_dr[2][0]: 'C: 시각형'}

        for c_idx, (c, _) in enumerate(cluster_dr):
            members = [movies[i] for i in range(len(movies)) if labels[i] == c]
            if not members:
                continue

            cname = cluster_names[c]
            raw_center = {f: sum(m[f] for m in members) / len(members) for f in feat_names}

            # Compute distance to center for each movie
            for m in members:
                m['_dist'] = sum(((m[f] - raw_center[f]) / stds[f])**2 for f in feat_names) ** 0.5

            members.sort(key=lambda m: m['_dist'])

            avg_r = sum(m['rating'] for m in members) / len(members)
            med_r = sorted(m['rating'] for m in members)[len(members)//2]

            print(f"\n{'=' * 70}")
            print(f"  CLUSTER {cname} (n={len(members)}, avg_r={avg_r:.2f}, med_r={med_r:.1f})")
            print(f"{'=' * 70}")
            print(f"  Center: narr={raw_center['llm_narr_std']:.1f}  cvar={raw_center['llm_char_var']:.1f}  "
                  f"gap={raw_center['llm_ppl_gap']:+.1f}  dppl={raw_center['llm_dial_ppl']:.1f}  "
                  f"dr={raw_center['dir_ratio']:.1%}")

            print(f"\n  CLOSEST TO CENTER (archetypes):")
            print(f"  {'rank':>4s} {'dist':>5s} {'r':>4s} {'title':>35s} {'narr':>5s} {'cvar':>5s} {'gap':>6s} {'dr':>5s}")
            print(f"  {'-'*70}")
            for i, m in enumerate(members[:10]):
                print(f"  {i+1:4d} {m['_dist']:5.2f} {m['rating']:4.1f} {m['title']:>35s} "
                      f"{m['llm_narr_std']:5.1f} {m['llm_char_var']:5.1f} {m['llm_ppl_gap']:+6.1f} {m['dir_ratio']:5.1%}")

            print(f"\n  FARTHEST FROM CENTER (outliers in this cluster):")
            for i, m in enumerate(members[-5:]):
                rank = len(members) - 4 + i
                print(f"  {rank:4d} {m['_dist']:5.2f} {m['rating']:4.1f} {m['title']:>35s} "
                      f"{m['llm_narr_std']:5.1f} {m['llm_char_var']:5.1f} {m['llm_ppl_gap']:+6.1f} {m['dir_ratio']:5.1%}")

            # Rating distribution: center 50% vs edge 50%
            half = len(members) // 2
            close = members[:half]
            far = members[half:]
            close_r = sorted(m['rating'] for m in close)
            far_r = sorted(m['rating'] for m in far)

            print(f"\n  Close half (n={len(close)}): avg={sum(close_r)/len(close_r):.2f}  "
                  f"min={close_r[0]:.1f}  p25={close_r[len(close_r)//4]:.1f}  "
                  f"med={close_r[len(close_r)//2]:.1f}  p75={close_r[3*len(close_r)//4]:.1f}  max={close_r[-1]:.1f}")
            print(f"  Far half  (n={len(far)}):  avg={sum(far_r)/len(far_r):.2f}  "
                  f"min={far_r[0]:.1f}  p25={far_r[len(far_r)//4]:.1f}  "
                  f"med={far_r[len(far_r)//2]:.1f}  p75={far_r[3*len(far_r)//4]:.1f}  max={far_r[-1]:.1f}")

    print()


if __name__ == "__main__":
    main()
