"""
바이럴 마케팅 확산 시뮬레이터

소셜 그래프 위의 콘텐츠 확산을 코사인-유사도 가중 라플라시안으로 모델링한다.
reaction_diffusion_grid.py의 격자 라플라시안을 소셜 그래프용으로 교체한 것이 핵심.

지배 방정식:
    ∂u/∂t = L_cos @ u          (확산)
    L_cos = D - A_cos           (코사인 유사도 가중 그래프 라플라시안)
    A_cos_ij = cos_sim(feat_i, feat_j)  if edge (i,j) exists

인플루언서 선정:
    L의 최소 비자명 고유벡터(Fiedler vector) → 고유벡터 중심성

가짜뉴스 판별:
    정상 콘텐츠 — 유사도 높은 방향으로 확산 (동질적 전파)
    가짜뉴스   — 유사도 무시 무차별 확산   (이질적 전파)
    → 고유값 스펙트럼 분포 차이로 구분
"""

import sys
import os
import numpy as np
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
#  그래프 생성: Barabási-Albert 모델
# ============================================================

def barabasi_albert_graph(n: int, m: int, seed: int = 42
                          ) -> List[Tuple[int, int]]:
    """Barabási-Albert preferential attachment 그래프 생성.

    Parameters
    ----------
    n : 총 노드 수
    m : 새 노드가 연결할 기존 노드 수
    seed : 난수 시드

    Returns
    -------
    edges : (i, j) 엣지 리스트 (양방향)
    """
    rng = np.random.RandomState(seed)
    # 초기 완전 그래프 (m+1 노드)
    edges = []
    degree = np.zeros(n, dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1

    for new_node in range(m + 1, n):
        # preferential attachment: 확률 ∝ degree
        existing = np.arange(new_node)
        prob = degree[:new_node].copy()
        prob /= prob.sum()
        targets = rng.choice(existing, size=m, replace=False, p=prob)
        for t in targets:
            edges.append((new_node, t))
            degree[new_node] += 1
            degree[t] += 1

    return edges


def adjacency_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """엣지 리스트 → 인접 행렬 (대칭)."""
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


# ============================================================
#  유저 피처 벡터 & 코사인 유사도
# ============================================================

def generate_user_features(n: int, dim: int = 8, seed: int = 42
                           ) -> np.ndarray:
    """합성 유저 피처 벡터 (관심사, 활동 시간대 등).

    Returns
    -------
    features : (n, dim) 배열, 각 행이 한 유저의 피처
    """
    rng = np.random.RandomState(seed)
    # 3개 클러스터로 유저 그룹 생성
    n_clusters = 3
    centers = rng.randn(n_clusters, dim) * 2
    labels = rng.randint(0, n_clusters, size=n)
    features = centers[labels] + rng.randn(n, dim) * 0.5
    # 정규화
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    features = features / norms
    return features


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """피처 행렬 → 코사인 유사도 행렬.

    이미 정규화된 벡터이므로 단순 내적.
    [-1, 1] → [0, 1] 로 선형 변환하여 모든 엣지에 양수 가중치를 보장.
    """
    sim = features @ features.T
    # 대각선 제거 (자기 자신과의 유사도)
    np.fill_diagonal(sim, 0.0)
    # [-1, 1] → [0, 1] 선형 변환: (sim + 1) / 2
    sim = (sim + 1.0) / 2.0
    np.fill_diagonal(sim, 0.0)
    return sim


# ============================================================
#  코사인 유사도 가중 그래프 라플라시안
# ============================================================

def cosine_laplacian(adj: np.ndarray, cos_sim: np.ndarray) -> np.ndarray:
    """A_cos = adj ⊙ cos_sim,  L = D - A_cos.

    Parameters
    ----------
    adj     : (n, n) 인접 행렬 (0/1)
    cos_sim : (n, n) 코사인 유사도 행렬

    Returns
    -------
    L : (n, n) 코사인 유사도 가중 그래프 라플라시안
    """
    A_cos = adj * cos_sim  # element-wise (Hadamard)
    # 대칭 보장
    A_cos = (A_cos + A_cos.T) / 2.0
    D = np.diag(A_cos.sum(axis=1))
    return D - A_cos


def random_laplacian(adj: np.ndarray) -> np.ndarray:
    """가짜뉴스용: 유사도 무시, 인접 여부만 사용.

    L = D - A (무가중).
    """
    D = np.diag(adj.sum(axis=1))
    return D - adj


# ============================================================
#  확산 시뮬레이션: Forward Euler
# ============================================================

def diffusion_simulate(L: np.ndarray, u0: np.ndarray,
                       dt: float = 0.005, steps: int = 2000,
                       save_every: int = 20,
                       diffusion_coeff: float = 0.1
                       ) -> dict:
    """∂u/∂t = -diffusion_coeff * L @ u  (Forward Euler).

    음의 부호: 라플라시안 정의 L=D-A 이므로
    확산 방향이 '높은 곳 → 낮은 곳'이 되려면 부호 반전 필요.

    Returns
    -------
    history : {'u': [...], 't': [...]}
    """
    n = len(u0)
    u = u0.copy()
    history = {'u': [u.copy()], 't': [0.0]}

    for step in range(1, steps + 1):
        du = -diffusion_coeff * (L @ u)
        u = u + dt * du
        u = np.clip(u, 0.0, None)  # 음수 방지

        if step % save_every == 0:
            history['u'].append(u.copy())
            history['t'].append(step * dt)

    return history


# ============================================================
#  인플루언서 선정: 고유벡터 중심성
# ============================================================

def find_influencers(L: np.ndarray, adj: np.ndarray, cos_sim: np.ndarray,
                     top_k: int = 5) -> dict:
    """고유벡터 중심성 기반 인플루언서 선정.

    1) A_cos 의 주 고유벡터 → 고유벡터 중심성
    2) Fiedler vector (L의 두 번째 최소 고유벡터) → 커뮤니티 경계

    Returns
    -------
    result : {
        'eigenvector_centrality': top-k 노드 인덱스,
        'fiedler_vector': Fiedler vector,
        'eigenvalues': L의 고유값,
    }
    """
    A_cos = adj * cos_sim

    # 고유벡터 중심성: A_cos의 최대 고유값에 해당하는 고유벡터
    eigvals_A, eigvecs_A = np.linalg.eigh(A_cos)
    centrality = np.abs(eigvecs_A[:, -1])  # 최대 고유벡터

    # 라플라시안 고유값 분해
    eigvals_L, eigvecs_L = np.linalg.eigh(L)

    # Fiedler vector (두 번째 최소 고유벡터)
    fiedler = eigvecs_L[:, 1]

    # top-k 인플루언서
    top_indices = np.argsort(centrality)[-top_k:][::-1]

    return {
        'centrality': centrality,
        'top_influencers': top_indices,
        'fiedler_vector': fiedler,
        'eigenvalues_L': eigvals_L,
        'eigenvalues_A': eigvals_A,
    }


# ============================================================
#  시각화
# ============================================================

def compute_layout(n: int, edges: List[Tuple[int, int]],
                   seed: int = 42, iterations: int = 50) -> np.ndarray:
    """Fruchterman-Reingold force-directed 레이아웃 (간이 구현).

    Returns
    -------
    pos : (n, 2) 좌표
    """
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    k = 1.0 / np.sqrt(n)  # 이상적 엣지 길이

    for it in range(iterations):
        disp = np.zeros((n, 2))
        temp = max(0.1, 1.0 - it / iterations)

        # 반발력 (모든 쌍)
        for i in range(n):
            diff = pos[i] - pos
            dist = np.linalg.norm(diff, axis=1)
            dist = np.maximum(dist, 0.01)
            force = (k * k / dist)[:, None] * (diff / dist[:, None])
            force[i] = 0
            disp[i] += force.sum(axis=0)

        # 인력 (엣지 연결)
        for i, j in edges:
            diff = pos[j] - pos[i]
            dist = max(np.linalg.norm(diff), 0.01)
            f = (dist / k) * (diff / dist)
            disp[i] += f
            disp[j] -= f

        # 위치 업데이트
        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp_norm = np.maximum(disp_norm, 0.01)
        pos += (disp / disp_norm) * min(temp, 0.05)
        pos = np.clip(pos, 0, 1)

    return pos


def plot_all(n: int, edges: List[Tuple[int, int]],
             pos: np.ndarray,
             history_normal: dict, history_fake: dict,
             influencer_result: dict,
             L_normal: np.ndarray, L_fake: np.ndarray,
             out_dir: str):
    """6-panel 시각화 → 이미지 저장.

    Panel 구성:
      [0] 소셜 그래프 + 인플루언서
      [1] 정상 확산 최종 상태
      [2] 가짜뉴스 확산 최종 상태
      [3] 확산 시계열 비교
      [4] 고유값 스펙트럼 비교
      [5] 고유벡터 중심성 분포
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    centrality = influencer_result['centrality']
    top_inf = influencer_result['top_influencers']
    eigvals_normal = np.linalg.eigvalsh(L_normal)
    eigvals_fake = np.linalg.eigvalsh(L_fake)

    # ---- Panel 0: 소셜 그래프 ----
    ax0 = fig.add_subplot(gs[0, 0])
    for i, j in edges:
        ax0.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                 'gray', alpha=0.15, linewidth=0.5)
    scatter = ax0.scatter(pos[:, 0], pos[:, 1], c=centrality,
                          cmap='YlOrRd', s=40, zorder=5, edgecolors='white',
                          linewidth=0.5)
    ax0.scatter(pos[top_inf, 0], pos[top_inf, 1], s=150,
                facecolors='none', edgecolors='red', linewidth=2, zorder=6)
    for idx in top_inf:
        ax0.annotate(f'#{idx}', pos[idx], fontsize=7, fontweight='bold',
                     color='red', ha='center', va='bottom',
                     xytext=(0, 8), textcoords='offset points')
    plt.colorbar(scatter, ax=ax0, label='Eigenvector Centrality')
    ax0.set_title('Social Graph + Influencers', fontweight='bold')
    ax0.set_xticks([])
    ax0.set_yticks([])

    # ---- Panel 1 & 2: 중간 시점 확산 비교 (차이가 가장 큰 시점) ----
    # 표준편차 차이가 최대인 프레임 찾기
    n_frames = min(len(history_normal['u']), len(history_fake['u']))
    best_frame = 0
    best_diff = 0
    for fi in range(n_frames):
        diff = abs(np.std(history_normal['u'][fi]) - np.std(history_fake['u'][fi]))
        if diff > best_diff:
            best_diff = diff
            best_frame = fi
    # 범위 통일
    vmax_mid = max(np.max(history_normal['u'][best_frame]),
                   np.max(history_fake['u'][best_frame]), 0.01)

    ax1 = fig.add_subplot(gs[0, 1])
    u_mid_normal = history_normal['u'][best_frame]
    t_mid = history_normal['t'][best_frame]
    scatter1 = ax1.scatter(pos[:, 0], pos[:, 1], c=u_mid_normal,
                           cmap='Greens', s=50, edgecolors='white',
                           linewidth=0.5, vmin=0, vmax=vmax_mid)
    plt.colorbar(scatter1, ax=ax1, label='Infection Level')
    ax1.set_title(f'Normal Diffusion (t={t_mid:.2f})', fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    u_mid_fake = history_fake['u'][best_frame]
    scatter2 = ax2.scatter(pos[:, 0], pos[:, 1], c=u_mid_fake,
                           cmap='Reds', s=50, edgecolors='white',
                           linewidth=0.5, vmin=0, vmax=vmax_mid)
    plt.colorbar(scatter2, ax=ax2, label='Infection Level')
    ax2.set_title(f'Fake News Diffusion (t={t_mid:.2f})', fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # ---- Panel 3: 확산 불균일도 시계열 비교 ----
    ax3 = fig.add_subplot(gs[1, 0])
    std_normal = [np.std(u) for u in history_normal['u']]
    std_fake = [np.std(u) for u in history_fake['u']]
    ax3.plot(history_normal['t'], std_normal, 'g-', linewidth=2,
             label='Normal (std)')
    ax3.plot(history_fake['t'], std_fake, 'r--', linewidth=2,
             label='Fake News (std)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Std Dev (Spread Inhomogeneity)')
    ax3.set_title('Diffusion Pattern Divergence', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: 고유값 스펙트럼 비교 ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(eigvals_normal, bins=25, alpha=0.6, color='green',
             label='Normal (cos-weighted)', density=True)
    ax4.hist(eigvals_fake, bins=25, alpha=0.6, color='red',
             label='Fake (unweighted)', density=True)
    ax4.axvline(eigvals_normal[1], color='green', linestyle='--', alpha=0.8,
                label=f'Fiedler λ₂={eigvals_normal[1]:.3f}')
    ax4.axvline(eigvals_fake[1], color='red', linestyle='--', alpha=0.8,
                label=f'Fiedler λ₂={eigvals_fake[1]:.3f}')
    ax4.set_xlabel('Eigenvalue')
    ax4.set_ylabel('Density')
    ax4.set_title('Laplacian Eigenvalue Spectrum', fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ---- Panel 5: 고유벡터 중심성 분포 ----
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_centrality = np.sort(centrality)[::-1]
    ax5.bar(range(n), sorted_centrality, color='coral', alpha=0.7)
    ax5.axhline(np.mean(centrality), color='navy', linestyle='--',
                label=f'Mean = {np.mean(centrality):.3f}')
    ax5.set_xlabel('User (sorted by centrality)')
    ax5.set_ylabel('Eigenvector Centrality')
    ax5.set_title('Centrality Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig.suptitle('Viral Marketing Simulator — Cosine-Similarity Laplacian Diffusion',
                 fontsize=14, fontweight='bold', y=0.98)

    path = os.path.join(out_dir, 'viral_marketing_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {path}")
    return path


def plot_snapshots(n: int, pos: np.ndarray,
                   history: dict, label: str,
                   out_dir: str):
    """확산 스냅샷 (초기 / 중간 / 최종)."""
    frames = [0, len(history['u']) // 2, -1]
    titles = ['Initial', 'Mid', 'Final']

    # 전체 프레임의 최대값으로 vmax 통일
    global_max = max(np.max(history['u'][f]) for f in frames)
    global_max = max(global_max, 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, frame, title in zip(axes, frames, titles):
        u = history['u'][frame]
        t = history['t'][frame]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=u,
                        cmap='YlOrRd', s=50, edgecolors='white',
                        linewidth=0.5, vmin=0, vmax=global_max)
        ax.set_title(f'{title} (t={t:.3f})', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax)

    fig.suptitle(f'{label} — Diffusion Snapshots', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f'snapshots_{label.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {path}")


# ============================================================
#  메인
# ============================================================

def main():
    print("=" * 55)
    print("  VIRAL MARKETING SIMULATOR")
    print("  Cosine-Similarity Laplacian Diffusion on Social Graph")
    print("=" * 55)

    # ---- 파라미터 ----
    N = 80          # 유저 수
    M = 3           # BA 모델: 새 노드당 연결 수
    FEAT_DIM = 8    # 피처 벡터 차원
    DT = 0.002      # 시간 간격
    STEPS = 800     # 시뮬레이션 스텝
    DIFF_COEFF = 0.15  # 확산 계수
    TOP_K = 5       # 인플루언서 수
    SEED = 42

    # ---- 1. 그래프 생성 ----
    print("\n[1] Barabási-Albert 그래프 생성...")
    edges = barabasi_albert_graph(N, M, seed=SEED)
    adj = adjacency_matrix(N, edges)
    print(f"    노드: {N},  엣지: {len(edges)},  평균 차수: {adj.sum()/N:.1f}")

    # ---- 2. 유저 피처 + 코사인 유사도 ----
    print("[2] 유저 피처 벡터 생성 및 코사인 유사도 계산...")
    features = generate_user_features(N, FEAT_DIM, seed=SEED)
    cos_sim = cosine_similarity_matrix(features)
    print(f"    피처 차원: {FEAT_DIM},  평균 유사도: {cos_sim[adj > 0].mean():.3f}")

    # ---- 3. 라플라시안 구성 ----
    print("[3] 코사인 유사도 가중 라플라시안 구성...")
    L_cos = cosine_laplacian(adj, cos_sim)
    L_rand = random_laplacian(adj)
    print(f"    L_cos 크기: {L_cos.shape}")

    # ---- 4. 인플루언서 선정 ----
    print("[4] 고유벡터 중심성 기반 인플루언서 선정...")
    inf_result = find_influencers(L_cos, adj, cos_sim, top_k=TOP_K)
    print(f"    Top-{TOP_K} 인플루언서: {inf_result['top_influencers']}")
    for i, idx in enumerate(inf_result['top_influencers']):
        print(f"      #{idx}: centrality={inf_result['centrality'][idx]:.4f}, "
              f"degree={int(adj[idx].sum())}")

    # ---- 5. 확산 시뮬레이션 ----
    print("[5] 확산 시뮬레이션...")

    # 시드 유저: top 인플루언서 상위 2명
    seed_users = inf_result['top_influencers'][:2]
    u0 = np.zeros(N)
    for su in seed_users:
        u0[su] = 1.0
    seed_user = seed_users[0]

    # 정상 콘텐츠: 코사인 유사도 라플라시안 사용
    print(f"    시드 유저: {seed_users}")
    print("    [5a] 정상 콘텐츠 확산 (코사인 유사도 기반)...")
    history_normal = diffusion_simulate(L_cos, u0, dt=DT, steps=STEPS,
                                        save_every=10,
                                        diffusion_coeff=DIFF_COEFF)

    # 가짜뉴스: 무가중 라플라시안 사용
    print("    [5b] 가짜뉴스 확산 (유사도 무시)...")
    history_fake = diffusion_simulate(L_rand, u0, dt=DT, steps=STEPS,
                                      save_every=10,
                                      diffusion_coeff=DIFF_COEFF)

    total_normal = np.sum(history_normal['u'][-1])
    total_fake = np.sum(history_fake['u'][-1])
    print(f"    최종 감염 총량 — 정상: {total_normal:.3f}, 가짜: {total_fake:.3f}")

    # ---- 6. 고유값 분석 ----
    print("[6] 고유값 스펙트럼 분석...")
    eigvals_cos = np.linalg.eigvalsh(L_cos)
    eigvals_rand = np.linalg.eigvalsh(L_rand)
    print(f"    정상 Fiedler 고유값 (λ₂): {eigvals_cos[1]:.4f}")
    print(f"    가짜 Fiedler 고유값 (λ₂): {eigvals_rand[1]:.4f}")
    print(f"    스펙트럼 갭 비율: {eigvals_cos[1]/eigvals_rand[1]:.4f}")

    # ---- 7. 시각화 ----
    print("[7] 시각화 생성...")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    pos = compute_layout(N, edges, seed=SEED)

    plot_all(N, edges, pos,
             history_normal, history_fake,
             inf_result, L_cos, L_rand, out_dir)

    plot_snapshots(N, pos, history_normal, 'Normal Content', out_dir)
    plot_snapshots(N, pos, history_fake, 'Fake News', out_dir)

    # ---- 요약 ----
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  그래프: BA({N}, {M}),  엣지: {len(edges)}")
    print(f"  시드 유저: #{seed_user} (centrality={inf_result['centrality'][seed_user]:.4f})")
    print(f"  정상 확산 총량: {total_normal:.3f}")
    print(f"  가짜뉴스 확산 총량: {total_fake:.3f}")
    print(f"  Fiedler λ₂ — 정상: {eigvals_cos[1]:.4f}, 가짜: {eigvals_rand[1]:.4f}")
    print(f"  출력: {out_dir}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
