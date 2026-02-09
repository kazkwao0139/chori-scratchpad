"""
뉴스 데이터 수집 & 코사인 유사도 스펙트럼 분석

1. 공개 데이터셋(McIntire) 다운로드 — 라벨링된 Real/Fake 뉴스
2. RSS 피드로 현재 핫뉴스 수집 (BBC, NPR)
3. TF-IDF 벡터화 → 코사인 유사도 행렬
4. 그래프 라플라시안 고유값 스펙트럼 비교
5. 가설 검증: 가짜뉴스 = 이질적 확산 → 스펙트럼 차이

핵심 가설:
    진짜 뉴스: 주제별 클러스터 내에서 유사한 기사끼리 퍼짐 (동질적)
              → 코사인 유사도 행렬이 블록 대각선 구조
              → 라플라시안 고유값이 0 근처에 집중

    가짜 뉴스: 주제 무관하게 무차별 확산 (이질적)
              → 코사인 유사도가 균일하게 분산
              → 라플라시안 고유값이 넓게 분포
"""

import sys
import os
import time
import csv
import io
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import requests
import feedparser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = Path(__file__).parent / 'output'


# ============================================================
#  1. 데이터 수집
# ============================================================

def download_mcintire_dataset(max_per_class: int = 300) -> Dict[str, List[str]]:
    """McIntire fake/real 뉴스 데이터셋 다운로드 (GitHub 직접 링크).

    Returns
    -------
    {'real': [텍스트, ...], 'fake': [텍스트, ...]}
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / 'mcintire_news.json'

    if cache_path.exists():
        print("  [캐시] McIntire 데이터셋 로드...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    url = ("https://raw.githubusercontent.com/GeorgeMcIntire/"
           "fake_real_news_dataset/main/fake_and_real_news_dataset.csv")
    print(f"  [다운로드] McIntire 데이터셋...")
    print(f"    URL: {url}")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    real_texts, fake_texts = [], []

    for row in reader:
        text = row.get('text', '').strip()
        label = row.get('label', '').strip().upper()
        if not text or len(text) < 100:
            continue
        if label == 'REAL' and len(real_texts) < max_per_class:
            real_texts.append(text[:2000])  # 최대 2000자
        elif label == 'FAKE' and len(fake_texts) < max_per_class:
            fake_texts.append(text[:2000])

    result = {'real': real_texts, 'fake': fake_texts}

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"    Real: {len(real_texts)}건, Fake: {len(fake_texts)}건")
    return result


def collect_rss_news(max_articles: int = 50) -> List[Dict]:
    """BBC, NPR 등 RSS 피드에서 현재 핫뉴스 수집.

    Returns
    -------
    [{'title': ..., 'summary': ..., 'source': ..., 'link': ...}, ...]
    """
    feeds = [
        ('BBC Top', 'https://feeds.bbci.co.uk/news/rss.xml'),
        ('BBC World', 'https://feeds.bbci.co.uk/news/world/rss.xml'),
        ('BBC Tech', 'https://feeds.bbci.co.uk/news/technology/rss.xml'),
        ('NPR News', 'https://feeds.npr.org/1001/rss.xml'),
        ('NPR World', 'https://feeds.npr.org/1004/rss.xml'),
        ('Google News', 'https://news.google.com/rss'),
    ]

    articles = []
    seen = set()

    for name, url in feeds:
        try:
            print(f"    [{name}] 수집 중...")
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                link = entry.get('link', '')

                # 중복 제거 (제목 해시)
                h = hashlib.md5(title.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)

                text = f"{title}. {summary}"
                if len(text) < 50:
                    continue

                articles.append({
                    'title': title,
                    'text': text,
                    'source': name,
                    'link': link,
                })
        except Exception as e:
            print(f"    [{name}] 실패: {e}")

        if len(articles) >= max_articles:
            break

    print(f"    총 {len(articles)}건 수집")
    return articles[:max_articles]


# ============================================================
#  2. TF-IDF 벡터화 & 코사인 유사도
# ============================================================

def compute_tfidf_cosine(texts: List[str], max_features: int = 3000
                         ) -> tuple:
    """텍스트 → TF-IDF → 코사인 유사도 행렬.

    Returns
    -------
    (cos_matrix, tfidf_matrix, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    cos_mat = cosine_similarity(tfidf)
    np.fill_diagonal(cos_mat, 0.0)
    return cos_mat, tfidf, vectorizer


# ============================================================
#  3. 그래프 라플라시안 & 스펙트럼 분석
# ============================================================

def cosine_laplacian(cos_mat: np.ndarray, threshold: float = 0.0
                     ) -> np.ndarray:
    """코사인 유사도 → 그래프 라플라시안.

    threshold 이하의 유사도는 0으로 클리핑 (노이즈 제거).
    """
    A = cos_mat.copy()
    A[A < threshold] = 0.0
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    return D - A


def spectral_analysis(L: np.ndarray) -> dict:
    """라플라시안 고유값 분석."""
    eigvals = np.linalg.eigvalsh(L)
    return {
        'eigenvalues': eigvals,
        'fiedler': eigvals[1] if len(eigvals) > 1 else 0,
        'spectral_gap': eigvals[1] / (eigvals[-1] + 1e-10) if len(eigvals) > 1 else 0,
        'mean': np.mean(eigvals),
        'std': np.std(eigvals),
        'max': np.max(eigvals),
    }


# ============================================================
#  4. 시각화
# ============================================================

def plot_analysis(real_spec: dict, fake_spec: dict,
                  cos_real: np.ndarray, cos_fake: np.ndarray,
                  rss_spec: Optional[dict], cos_rss: Optional[np.ndarray],
                  n_real: int, n_fake: int, n_rss: int,
                  save_path: str):
    """종합 분석 시각화."""

    n_cols = 3 if rss_spec else 2
    fig = plt.figure(figsize=(7 * n_cols, 14))
    gs = gridspec.GridSpec(3, n_cols, hspace=0.35, wspace=0.3)

    datasets = [
        ('Real News', real_spec, cos_real, 'green', n_real),
        ('Fake News', fake_spec, cos_fake, 'red', n_fake),
    ]
    if rss_spec:
        datasets.append(('RSS Hot News', rss_spec, cos_rss, 'blue', n_rss))

    # Row 0: 코사인 유사도 히트맵
    for col, (label, spec, cos_mat, color, n) in enumerate(datasets):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(cos_mat[:min(n, 100), :min(n, 100)],
                       cmap='YlOrRd', vmin=0, vmax=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'{label} — Cosine Similarity\n({n} articles)',
                     fontweight='bold')
        ax.set_xlabel('Article Index')
        ax.set_ylabel('Article Index')

    # Row 1: 고유값 스펙트럼
    ax_spec = fig.add_subplot(gs[1, :])
    for label, spec, _, color, _ in datasets:
        eigvals = spec['eigenvalues']
        ax_spec.hist(eigvals, bins=30, alpha=0.5, color=color,
                     label=f'{label} (λ₂={spec["fiedler"]:.3f})',
                     density=True)
        ax_spec.axvline(spec['fiedler'], color=color, linestyle='--',
                        alpha=0.8, linewidth=2)

    ax_spec.set_xlabel('Eigenvalue', fontsize=12)
    ax_spec.set_ylabel('Density', fontsize=12)
    ax_spec.set_title('Laplacian Eigenvalue Spectrum Comparison',
                      fontweight='bold', fontsize=14)
    ax_spec.legend(fontsize=11)
    ax_spec.grid(True, alpha=0.3)

    # Row 2: 유사도 분포 + 통계 요약
    ax_dist = fig.add_subplot(gs[2, 0])
    for label, _, cos_mat, color, _ in datasets:
        vals = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        ax_dist.hist(vals, bins=50, alpha=0.5, color=color,
                     label=label, density=True)
    ax_dist.set_xlabel('Cosine Similarity')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('Pairwise Cosine Similarity Distribution',
                      fontweight='bold')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    # 통계 테이블
    ax_table = fig.add_subplot(gs[2, 1:])
    ax_table.axis('off')

    table_data = []
    headers = ['Metric', 'Real News', 'Fake News']
    if rss_spec:
        headers.append('RSS Hot News')

    metrics = [
        ('Articles', [str(d[4]) for d in datasets]),
        ('Fiedler λ₂', [f'{d[1]["fiedler"]:.4f}' for d in datasets]),
        ('Spectral Gap', [f'{d[1]["spectral_gap"]:.4f}' for d in datasets]),
        ('Eigenvalue Mean', [f'{d[1]["mean"]:.4f}' for d in datasets]),
        ('Eigenvalue Std', [f'{d[1]["std"]:.4f}' for d in datasets]),
        ('Eigenvalue Max', [f'{d[1]["max"]:.4f}' for d in datasets]),
        ('Mean Cosine Sim', [f'{d[2][np.triu_indices_from(d[2], k=1)].mean():.4f}'
                             for d in datasets]),
    ]

    for name, vals in metrics:
        table_data.append([name] + vals)

    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 헤더 스타일
    for j, h in enumerate(headers):
        table[0, j].set_facecolor('#2d3748')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 데이터 행 색상
    colors = ['#f0fff4', '#fff5f5']
    if rss_spec:
        colors.append('#ebf8ff')
    for i in range(len(table_data)):
        table[i + 1, 0].set_facecolor('#edf2f7')
        table[i + 1, 0].set_text_props(fontweight='bold')
        for j in range(1, len(headers)):
            table[i + 1, j].set_facecolor(colors[j - 1] if j - 1 < len(colors) else '#ffffff')

    fig.suptitle('Real vs Fake News — Cosine Similarity Spectral Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================
#  메인
# ============================================================

def main():
    print("=" * 60)
    print("  NEWS DATA COLLECTOR & COSINE SPECTRAL ANALYZER")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    SAMPLE_SIZE = 200  # 클래스당 기사 수

    # ---- 1. 데이터셋 다운로드 ----
    print("\n[1] 라벨링 데이터셋 수집 (McIntire)...")
    dataset = download_mcintire_dataset(max_per_class=SAMPLE_SIZE)
    n_real = len(dataset['real'])
    n_fake = len(dataset['fake'])
    print(f"    Real: {n_real}건, Fake: {n_fake}건")

    # ---- 2. RSS 핫뉴스 수집 ----
    print("\n[2] RSS 피드에서 현재 핫뉴스 수집...")
    rss_articles = collect_rss_news(max_articles=60)
    n_rss = len(rss_articles)
    rss_texts = [a['text'] for a in rss_articles]

    # 수집된 뉴스 샘플 출력
    print("\n    --- 수집된 핫뉴스 (상위 5건) ---")
    for i, a in enumerate(rss_articles[:5]):
        print(f"    {i+1}. [{a['source']}] {a['title'][:60]}")

    # ---- 3. TF-IDF + 코사인 유사도 ----
    print("\n[3] TF-IDF 벡터화 & 코사인 유사도 계산...")

    # Real 뉴스
    print("    [3a] Real 뉴스...")
    cos_real, _, _ = compute_tfidf_cosine(dataset['real'])
    print(f"         평균 유사도: {cos_real[np.triu_indices_from(cos_real, k=1)].mean():.4f}")

    # Fake 뉴스
    print("    [3b] Fake 뉴스...")
    cos_fake, _, _ = compute_tfidf_cosine(dataset['fake'])
    print(f"         평균 유사도: {cos_fake[np.triu_indices_from(cos_fake, k=1)].mean():.4f}")

    # RSS 핫뉴스
    cos_rss = None
    if n_rss >= 10:
        print("    [3c] RSS 핫뉴스...")
        cos_rss, _, _ = compute_tfidf_cosine(rss_texts)
        print(f"         평균 유사도: {cos_rss[np.triu_indices_from(cos_rss, k=1)].mean():.4f}")

    # ---- 4. 라플라시안 스펙트럼 분석 ----
    print("\n[4] 라플라시안 고유값 스펙트럼 분석...")

    L_real = cosine_laplacian(cos_real, threshold=0.05)
    L_fake = cosine_laplacian(cos_fake, threshold=0.05)
    spec_real = spectral_analysis(L_real)
    spec_fake = spectral_analysis(L_fake)

    print(f"    Real — Fiedler λ₂: {spec_real['fiedler']:.4f}, "
          f"Spectral gap: {spec_real['spectral_gap']:.4f}")
    print(f"    Fake — Fiedler λ₂: {spec_fake['fiedler']:.4f}, "
          f"Spectral gap: {spec_fake['spectral_gap']:.4f}")

    spec_rss = None
    if cos_rss is not None:
        L_rss = cosine_laplacian(cos_rss, threshold=0.05)
        spec_rss = spectral_analysis(L_rss)
        print(f"    RSS  — Fiedler λ₂: {spec_rss['fiedler']:.4f}, "
              f"Spectral gap: {spec_rss['spectral_gap']:.4f}")

    # ---- 5. 시각화 ----
    print("\n[5] 시각화 생성...")
    plot_analysis(
        spec_real, spec_fake,
        cos_real, cos_fake,
        spec_rss, cos_rss,
        n_real, n_fake, n_rss,
        save_path=str(OUTPUT_DIR / 'news_cosine_spectral_analysis.png')
    )

    # ---- 결론 ----
    print("\n" + "=" * 60)
    print("  ANALYSIS RESULTS")
    print("=" * 60)

    ratio = spec_fake['fiedler'] / (spec_real['fiedler'] + 1e-10)
    print(f"\n  Fiedler λ₂ 비율 (Fake/Real): {ratio:.4f}")

    if spec_real['fiedler'] < spec_fake['fiedler']:
        print("\n  ★ 가설 지지: Real 뉴스의 Fiedler 값이 더 작음")
        print("    → Real 뉴스는 주제 클러스터 내에서 더 동질적으로 분포")
        print("    → Fake 뉴스는 더 무작위적/이질적 구조")
    else:
        print("\n  ▲ 가설과 반대 결과: Real 뉴스의 Fiedler 값이 더 큼")
        print("    → 추가 분석 필요 (샘플 크기, 주제 분포 등)")

    cos_real_mean = cos_real[np.triu_indices_from(cos_real, k=1)].mean()
    cos_fake_mean = cos_fake[np.triu_indices_from(cos_fake, k=1)].mean()

    print(f"\n  평균 코사인 유사도:")
    print(f"    Real: {cos_real_mean:.4f}")
    print(f"    Fake: {cos_fake_mean:.4f}")

    if cos_real_mean > cos_fake_mean:
        print("    → Real 뉴스가 더 높은 내부 유사도 (주제 집중)")
    else:
        print("    → Fake 뉴스가 더 높은 내부 유사도")

    print(f"\n  출력: {OUTPUT_DIR}/news_cosine_spectral_analysis.png")
    print("=" * 60)

    # 데이터 저장
    summary = {
        'real_count': n_real,
        'fake_count': n_fake,
        'rss_count': n_rss,
        'real_fiedler': float(spec_real['fiedler']),
        'fake_fiedler': float(spec_fake['fiedler']),
        'rss_fiedler': float(spec_rss['fiedler']) if spec_rss else None,
        'real_mean_cosine': float(cos_real_mean),
        'fake_mean_cosine': float(cos_fake_mean),
        'fiedler_ratio': float(ratio),
    }
    summary_path = DATA_DIR / 'analysis_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
