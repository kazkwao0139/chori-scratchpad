from pathlib import Path
"""
Can we predict IMDB rating from screenplay alone?
Three axes: char_var (X), narr_std (Y), |dir_ratio - 0.575| (Z).
Check: normality of each axis, then regression.
"""

import json
import sys
import math
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)


def main():
    print("=" * 70)
    print("  PREDICT IMDB RATING FROM SCREENPLAY — THREE AXES")
    print("=" * 70)

    with open(f'{BASE}/screenplay/mass_v2_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    rated = data['rated']

    # Filter: need all three axes
    complete = []
    for d in rated:
        if (d.get('char_var') is not None
                and d.get('narr_std') is not None
                and d.get('dir_ratio') is not None
                and d.get('rating') is not None
                and d.get('votes', 0) >= 1000):
            complete.append(d)

    print(f"\n  Total rated: {len(rated)}")
    print(f"  With all 3 axes: {len(complete)}")

    # Extract axes
    x = [d['char_var'] for d in complete]       # character diversity
    y = [d['narr_std'] for d in complete]       # narrative volatility
    z = [abs(d['dir_ratio'] - 0.575) for d in complete]  # visual balance deviation
    r = [d['rating'] for d in complete]         # IMDB rating

    # ─── Normality check ───
    print(f"\n{'=' * 70}")
    print("  NORMALITY CHECK")
    print(f"{'=' * 70}")

    def stats(vals, name):
        n = len(vals)
        mu = sum(vals) / n
        sigma = (sum((v - mu)**2 for v in vals) / n) ** 0.5
        median = sorted(vals)[n // 2]
        # Skewness
        skew = sum((v - mu)**3 for v in vals) / (n * sigma**3) if sigma > 0 else 0
        # Kurtosis (excess)
        kurt = sum((v - mu)**4 for v in vals) / (n * sigma**4) - 3 if sigma > 0 else 0
        # Jarque-Bera test: JB = n/6 * (S^2 + K^2/4)
        jb = n / 6 * (skew**2 + kurt**2 / 4)
        # JB ~ chi2(2), critical value at 0.05 = 5.99
        jb_sig = "REJECT" if jb > 5.99 else "NORMAL"

        print(f"\n  {name}:")
        print(f"    n={n}  mean={mu:.6f}  std={sigma:.6f}  median={median:.6f}")
        print(f"    skewness={skew:+.3f}  kurtosis={kurt:+.3f}")
        print(f"    Jarque-Bera={jb:.2f}  → {jb_sig} (crit=5.99 at α=0.05)")

        # Histogram (ASCII)
        lo, hi = min(vals), max(vals)
        n_bins = 20
        bw = (hi - lo) / n_bins if hi > lo else 1
        bins = [0] * n_bins
        for v in vals:
            idx = min(int((v - lo) / bw), n_bins - 1)
            bins[idx] += 1
        max_count = max(bins)
        print(f"    [{lo:.4f} .. {hi:.4f}]")
        for i, cnt in enumerate(bins):
            bar = "█" * int(cnt / max_count * 40) if max_count > 0 else ""
            print(f"    {lo + i*bw:8.4f} |{bar} {cnt}")

        return mu, sigma, skew, kurt

    mx, sx, skx, kux = stats(x, "X: char_var (character diversity)")
    my, sy, sky, kuy = stats(y, "Y: narr_std (narrative volatility)")
    mz, sz, skz, kuz = stats(z, "Z: |dir_ratio - 0.575| (visual balance)")
    mr, sr, skr, kur = stats(r, "Rating (IMDB)")

    # ─── Correlation matrix ───
    print(f"\n{'=' * 70}")
    print("  CORRELATION MATRIX (all pairs)")
    print(f"{'=' * 70}")

    def pearson(a, b):
        n = len(a)
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

    pairs = [
        ('char_var', x), ('narr_std', y), ('|dev|', z), ('rating', r)
    ]
    print(f"\n  {'':>12s}", end="")
    for name, _ in pairs:
        print(f" {name:>10s}", end="")
    print()
    for n1, v1 in pairs:
        print(f"  {n1:>12s}", end="")
        for n2, v2 in pairs:
            corr, _ = pearson(v1, v2)
            print(f" {corr:+10.4f}", end="")
        print()

    # ─── Multiple regression ───
    print(f"\n{'=' * 70}")
    print("  MULTIPLE REGRESSION: rating ~ char_var + narr_std + |dev|")
    print(f"{'=' * 70}")

    # Standardize features
    n = len(complete)

    def standardize(vals):
        mu = sum(vals) / len(vals)
        sigma = (sum((v - mu)**2 for v in vals) / len(vals)) ** 0.5
        return [(v - mu) / sigma if sigma > 0 else 0 for v in vals], mu, sigma

    xs, xmu, xsig = standardize(x)
    ys, ymu, ysig = standardize(y)
    zs, zmu, zsig = standardize(z)

    # OLS via normal equations: β = (X'X)^-1 X'y
    # Design matrix: [1, xs, ys, zs]
    # For 3 features + intercept, solve 4x4 system
    X = [[1, xs[i], ys[i], zs[i]] for i in range(n)]
    k = 4  # number of params

    # X'X
    XtX = [[0]*k for _ in range(k)]
    Xty = [0]*k
    for i in range(n):
        for j in range(k):
            for l in range(k):
                XtX[j][l] += X[i][j] * X[i][l]
            Xty[j] += X[i][j] * r[i]

    # Gauss elimination to solve XtX * beta = Xty
    aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        # Pivot
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

    # Predictions and R²
    y_pred = [sum(beta[j] * X[i][j] for j in range(k)) for i in range(n)]
    ss_res = sum((r[i] - y_pred[i])**2 for i in range(n))
    ss_tot = sum((r[i] - mr)**2 for i in range(n))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k) if n > k else r_sq

    # Standard errors
    mse = ss_res / (n - k) if n > k else ss_res / n

    # Invert XtX for se (recompute)
    # Actually let's just compute t-stats from beta and residual variance
    # se(beta_j) = sqrt(mse * (X'X)^-1_jj)
    # We already have the inverse in aug (identity part)
    # Re-solve for inverse
    inv_XtX = [[0]*k for _ in range(k)]
    for target_col in range(k):
        rhs = [1.0 if i == target_col else 0.0 for i in range(k)]
        aug2 = [XtX[i][:] + [rhs[i]] for i in range(k)]
        for col in range(k):
            max_row = max(range(col, k), key=lambda r: abs(aug2[r][col]))
            aug2[col], aug2[max_row] = aug2[max_row], aug2[col]
            pivot = aug2[col][col]
            if abs(pivot) < 1e-12:
                continue
            for j in range(col, k+1):
                aug2[col][j] /= pivot
            for row in range(k):
                if row == col:
                    continue
                factor = aug2[row][col]
                for j in range(col, k+1):
                    aug2[row][j] -= factor * aug2[col][j]
        for i in range(k):
            inv_XtX[i][target_col] = aug2[i][k]

    se = [(mse * inv_XtX[j][j]) ** 0.5 if inv_XtX[j][j] > 0 else 0 for j in range(k)]
    t_stats = [beta[j] / se[j] if se[j] > 0 else 0 for j in range(k)]

    labels = ['intercept', 'char_var', 'narr_std', '|dev|']
    print(f"\n  {'Variable':>12s} {'β':>10s} {'SE':>10s} {'t':>8s} {'sig':>6s}")
    print(f"  {'-'*50}")
    for j in range(k):
        sig = "***" if abs(t_stats[j]) > 3.29 else "**" if abs(t_stats[j]) > 2.58 else "*" if abs(t_stats[j]) > 1.96 else ""
        print(f"  {labels[j]:>12s} {beta[j]:+10.4f} {se[j]:10.4f} {t_stats[j]:+8.2f} {sig:>6s}")

    print(f"\n  R² = {r_sq:.4f}")
    print(f"  Adjusted R² = {adj_r_sq:.4f}")
    print(f"  n = {n}")
    print(f"  RMSE = {mse**0.5:.3f}")

    # ─── Non-linear: add quadratic terms ───
    print(f"\n{'=' * 70}")
    print("  QUADRATIC REGRESSION: rating ~ x + y + z + x² + y² + z²")
    print(f"{'=' * 70}")

    X2 = [[1, xs[i], ys[i], zs[i], xs[i]**2, ys[i]**2, zs[i]**2] for i in range(n)]
    k2 = 7

    XtX2 = [[0]*k2 for _ in range(k2)]
    Xty2 = [0]*k2
    for i in range(n):
        for j in range(k2):
            for l in range(k2):
                XtX2[j][l] += X2[i][j] * X2[i][l]
            Xty2[j] += X2[i][j] * r[i]

    aug = [XtX2[i][:] + [Xty2[i]] for i in range(k2)]
    for col in range(k2):
        max_row = max(range(col, k2), key=lambda rr: abs(aug[rr][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        for j in range(col, k2+1):
            aug[col][j] /= pivot
        for row in range(k2):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, k2+1):
                aug[row][j] -= factor * aug[col][j]

    beta2 = [aug[i][k2] for i in range(k2)]
    y_pred2 = [sum(beta2[j] * X2[i][j] for j in range(k2)) for i in range(n)]
    ss_res2 = sum((r[i] - y_pred2[i])**2 for i in range(n))
    r_sq2 = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
    adj_r_sq2 = 1 - (1 - r_sq2) * (n - 1) / (n - k2) if n > k2 else r_sq2
    mse2 = ss_res2 / (n - k2)

    labels2 = ['intercept', 'char_var', 'narr_std', '|dev|', 'char_var²', 'narr_std²', '|dev|²']
    print(f"\n  {'Variable':>12s} {'β':>10s}")
    print(f"  {'-'*25}")
    for j in range(k2):
        print(f"  {labels2[j]:>12s} {beta2[j]:+10.4f}")

    print(f"\n  R² = {r_sq2:.4f}")
    print(f"  Adjusted R² = {adj_r_sq2:.4f}")
    print(f"  RMSE = {mse2**0.5:.3f}")

    # ─── Prediction error distribution ───
    print(f"\n{'=' * 70}")
    print("  PREDICTION ERROR ANALYSIS")
    print(f"{'=' * 70}")

    errors = [r[i] - y_pred2[i] for i in range(n)]
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / n
    within_05 = sum(1 for e in abs_errors if e <= 0.5) / n
    within_10 = sum(1 for e in abs_errors if e <= 1.0) / n
    within_15 = sum(1 for e in abs_errors if e <= 1.5) / n

    print(f"  MAE = {mae:.3f}")
    print(f"  Within ±0.5: {within_05:.1%}")
    print(f"  Within ±1.0: {within_10:.1%}")
    print(f"  Within ±1.5: {within_15:.1%}")

    # ─── Extreme predictions ───
    print(f"\n  Best predicted vs actual (top 10 closest):")
    indexed = list(range(n))
    indexed.sort(key=lambda i: abs_errors[i])
    for rank, i in enumerate(indexed[:10]):
        print(f"    pred={y_pred2[i]:.1f} actual={r[i]:.1f} err={errors[i]:+.2f}")

    print(f"\n  Worst predictions (top 10 furthest):")
    indexed.sort(key=lambda i: -abs_errors[i])
    for rank, i in enumerate(indexed[:10]):
        print(f"    pred={y_pred2[i]:.1f} actual={r[i]:.1f} err={errors[i]:+.2f}")

    # ─── Save ───
    output = {
        'n': n,
        'normality': {
            'char_var': {'mean': mx, 'std': sx, 'skew': round(skx, 3), 'kurtosis': round(kux, 3)},
            'narr_std': {'mean': my, 'std': sy, 'skew': round(sky, 3), 'kurtosis': round(kuy, 3)},
            'dev': {'mean': mz, 'std': sz, 'skew': round(skz, 3), 'kurtosis': round(kuz, 3)},
        },
        'linear_regression': {
            'r_squared': round(r_sq, 4),
            'adj_r_squared': round(adj_r_sq, 4),
            'coefficients': {labels[j]: round(beta[j], 4) for j in range(k)},
        },
        'quadratic_regression': {
            'r_squared': round(r_sq2, 4),
            'adj_r_squared': round(adj_r_sq2, 4),
            'rmse': round(mse2**0.5, 3),
            'mae': round(mae, 3),
        },
    }
    with open(f'{BASE}/screenplay/predict_rating_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: predict_rating_results.json")


if __name__ == "__main__":
    main()
