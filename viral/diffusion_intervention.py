"""
확산 패턴 정밀 분석 & 차단 전략 시뮬레이터

검증 목표:
    1. Real vs Fake 확산이 "어떻게" 다른지 정량화
    2. 확산 초기에 구분 가능한 시점은 언제인지
    3. 어떤 노드를 차단하면 가짜뉴스 확산을 멈출 수 있는지
    4. 최소 비용으로 최대 차단 효과를 내는 전략은?

차단 전략:
    A. 중심성 기반 — 고유벡터 중심성 top-k 노드 제거 (예방접종)
    B. 유사도 기반 — 저유사도 엣지 차단 (가짜뉴스 경로 차단)
    C. 조기 탐지 — 감염 패턴 모니터링 → 이상 감지 시 허브 차단
    D. 랜덤 차단 — 비교 기준선
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 시뮬레이터 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viral_marketing_sim import (
    barabasi_albert_graph, adjacency_matrix,
    generate_user_features, cosine_similarity_matrix,
    cosine_laplacian, random_laplacian,
    find_influencers, compute_layout,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# ============================================================
#  정밀 확산 시뮬레이션 (경로 추적 포함)
# ============================================================

def diffusion_with_tracking(L: np.ndarray, u0: np.ndarray,
                            adj: np.ndarray, cos_sim: np.ndarray,
                            dt: float = 0.002, steps: int = 1500,
                            diffusion_coeff: float = 0.15,
                            infection_threshold: float = 0.01
                            ) -> dict:
    """확산 시뮬레이션 + 감염 경로/시점 추적.

    Returns
    -------
    {
        'u_history': [u0, u1, ...],
        't': [0, dt, 2*dt, ...],
        'infection_time': 각 노드가 처음 감염된 시점,
        'infection_order': 감염 순서,
        'edge_sim_used': 감염 전파에 사용된 엣지의 유사도 분포,
        'infected_count': 시간별 감염 노드 수,
        'std_history': 시간별 표준편차,
        'max_history': 시간별 최대 감염값,
        'cluster_coeff': 시간별 감염 노드의 클러스터링 계수,
    }
    """
    n = len(u0)
    u = u0.copy()

    infection_time = np.full(n, np.inf)
    infection_order = []
    edge_sims = []

    # 초기 감염 노드 기록
    for i in range(n):
        if u[i] > infection_threshold:
            infection_time[i] = 0.0
            infection_order.append(i)

    history = {
        'u_history': [u.copy()],
        't': [0.0],
        'infected_count': [np.sum(u > infection_threshold)],
        'std_history': [np.std(u)],
        'max_history': [np.max(u)],
        'cluster_coeff': [],
    }

    save_every = max(1, steps // 200)

    for step in range(1, steps + 1):
        u_prev = u.copy()
        du = -diffusion_coeff * (L @ u)
        u = u + dt * du
        u = np.clip(u, 0.0, None)

        t = step * dt

        # 새로 감염된 노드 추적
        for i in range(n):
            if u[i] > infection_threshold and infection_time[i] == np.inf:
                infection_time[i] = t
                infection_order.append(i)

                # 이 노드를 감염시킨 엣지의 유사도 수집
                neighbors = np.where(adj[i] > 0)[0]
                infected_neighbors = [j for j in neighbors
                                      if infection_time[j] < t]
                if infected_neighbors:
                    sims = [cos_sim[i, j] for j in infected_neighbors]
                    edge_sims.extend(sims)

        if step % save_every == 0:
            history['u_history'].append(u.copy())
            history['t'].append(t)
            history['infected_count'].append(int(np.sum(u > infection_threshold)))
            history['std_history'].append(float(np.std(u)))
            history['max_history'].append(float(np.max(u)))

    history['infection_time'] = infection_time
    history['infection_order'] = infection_order
    history['edge_sim_used'] = edge_sims

    return history


# ============================================================
#  차단 전략
# ============================================================

def apply_node_removal(adj: np.ndarray, cos_sim: np.ndarray,
                       nodes_to_remove: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """노드 제거 (행/열을 0으로 만듦)."""
    adj_mod = adj.copy()
    cos_mod = cos_sim.copy()
    for node in nodes_to_remove:
        adj_mod[node, :] = 0
        adj_mod[:, node] = 0
        cos_mod[node, :] = 0
        cos_mod[:, node] = 0
    return adj_mod, cos_mod


def apply_edge_removal(adj: np.ndarray, cos_sim: np.ndarray,
                       threshold: float) -> np.ndarray:
    """저유사도 엣지 제거: cos_sim < threshold 인 엣지 차단."""
    adj_mod = adj.copy()
    mask = cos_sim < threshold
    adj_mod[mask] = 0
    return adj_mod


def intervention_simulation(n: int, edges: List[Tuple[int, int]],
                            adj: np.ndarray, cos_sim: np.ndarray,
                            features: np.ndarray,
                            seed_nodes: List[int],
                            dt: float, steps: int, diff_coeff: float
                            ) -> dict:
    """여러 차단 전략의 효과 비교 시뮬레이션.

    전략:
        0. baseline_normal  — 정상 확산 (코사인 가중)
        1. baseline_fake    — 가짜뉴스 확산 (무가중)
        2. no_intervention  — 가짜뉴스 확산, 차단 없음
        3. hub_removal      — 중심성 top-5 노드 제거 후 가짜뉴스 확산
        4. edge_filter      — 저유사도 엣지 제거 후 가짜뉴스 확산
        5. random_removal   — 랜덤 5개 노드 제거 (기준선)
        6. early_detect     — 초기 20% 감염 시점에서 허브 차단
    """
    results = {}

    # 초기 조건
    u0 = np.zeros(n)
    for s in seed_nodes:
        u0[s] = 1.0

    # --- 0. 정상 확산 (코사인 가중) ---
    L_cos = cosine_laplacian(adj, cos_sim)
    results['normal'] = diffusion_with_tracking(
        L_cos, u0, adj, cos_sim, dt=dt, steps=steps,
        diffusion_coeff=diff_coeff)

    # --- 1. 가짜뉴스 (무가중, 차단 없음) ---
    L_rand = random_laplacian(adj)
    results['fake_no_intervention'] = diffusion_with_tracking(
        L_rand, u0, adj, cos_sim, dt=dt, steps=steps,
        diffusion_coeff=diff_coeff)

    # --- 2. 허브 제거 (top-5 중심성 노드) ---
    inf_result = find_influencers(L_cos, adj, cos_sim, top_k=5)
    hub_nodes = list(inf_result['top_influencers'])
    adj_no_hub, cos_no_hub = apply_node_removal(adj, cos_sim, hub_nodes)
    L_no_hub = random_laplacian(adj_no_hub)

    # 시드 노드가 제거된 허브에 포함되면 대체
    u0_hub = np.zeros(n)
    remaining_seeds = [s for s in seed_nodes if s not in hub_nodes]
    if not remaining_seeds:
        # 허브가 아닌 노드 중 degree 높은 것
        degrees = adj_no_hub.sum(axis=1)
        remaining_seeds = [np.argmax(degrees)]
    for s in remaining_seeds:
        u0_hub[s] = 1.0

    results['hub_removal'] = diffusion_with_tracking(
        L_no_hub, u0_hub, adj_no_hub, cos_no_hub, dt=dt, steps=steps,
        diffusion_coeff=diff_coeff)

    # --- 3. 저유사도 엣지 차단 (cos_sim < median) ---
    edge_sims = cos_sim[adj > 0]
    threshold = np.median(edge_sims) if len(edge_sims) > 0 else 0.5
    adj_filtered = apply_edge_removal(adj, cos_sim, threshold)
    L_filtered = random_laplacian(adj_filtered)

    results['edge_filter'] = diffusion_with_tracking(
        L_filtered, u0, adj_filtered, cos_sim, dt=dt, steps=steps,
        diffusion_coeff=diff_coeff)

    # --- 4. 랜덤 노드 제거 (기준선, 10회 평균) ---
    rng = np.random.RandomState(42)
    random_counts = []
    for trial in range(10):
        random_nodes = rng.choice(n, size=5, replace=False).tolist()
        adj_rand, cos_rand = apply_node_removal(adj, cos_sim, random_nodes)
        L_rand_rem = random_laplacian(adj_rand)
        u0_rand = u0.copy()
        for rn in random_nodes:
            u0_rand[rn] = 0
        res = diffusion_with_tracking(
            L_rand_rem, u0_rand, adj_rand, cos_rand, dt=dt, steps=steps,
            diffusion_coeff=diff_coeff)
        random_counts.append(res['infected_count'])

    # 평균
    max_len = max(len(c) for c in random_counts)
    avg_counts = []
    for i in range(max_len):
        vals = [c[i] for c in random_counts if i < len(c)]
        avg_counts.append(np.mean(vals))
    results['random_removal_counts'] = avg_counts
    results['random_removal_t'] = results['fake_no_intervention']['t'][:max_len]

    # --- 5. 조기 탐지 + 차단 ---
    # 가짜뉴스 확산 → 감염 20%에 도달하면 허브 차단
    results['early_detection'] = early_detection_sim(
        n, adj, cos_sim, u0, hub_nodes, dt, steps, diff_coeff)

    # 메타 데이터
    results['hub_nodes'] = hub_nodes
    results['edge_threshold'] = threshold
    results['influencer_result'] = inf_result

    return results


def early_detection_sim(n, adj, cos_sim, u0, hub_nodes,
                        dt, steps, diff_coeff,
                        trigger_ratio=0.15):
    """조기 탐지 시뮬레이션: 감염률이 trigger_ratio에 도달하면 허브 차단."""
    L = random_laplacian(adj)
    u = u0.copy()
    threshold = 0.01
    triggered = False
    trigger_time = None

    history = {
        'u_history': [u.copy()],
        't': [0.0],
        'infected_count': [int(np.sum(u > threshold))],
        'trigger_time': None,
    }

    save_every = max(1, steps // 200)

    for step in range(1, steps + 1):
        du = -diff_coeff * (L @ u)
        u = u + dt * du
        u = np.clip(u, 0.0, None)
        t = step * dt

        # 트리거 조건: 감염 노드가 전체의 trigger_ratio 이상
        infected_ratio = np.sum(u > threshold) / n
        if not triggered and infected_ratio >= trigger_ratio:
            triggered = True
            trigger_time = t
            history['trigger_time'] = t
            # 허브 노드 차단 (즉시)
            for node in hub_nodes:
                u[node] = 0
            # 라플라시안 재구성
            adj_mod, cos_mod = apply_node_removal(adj, cos_sim, hub_nodes)
            L = random_laplacian(adj_mod)

        if step % save_every == 0:
            history['u_history'].append(u.copy())
            history['t'].append(t)
            history['infected_count'].append(int(np.sum(u > threshold)))

    return history


# ============================================================
#  조기 구분 분석
# ============================================================

def early_detection_analysis(normal_hist: dict, fake_hist: dict,
                             n: int) -> dict:
    """확산 초기에 Real vs Fake를 구분할 수 있는 시점 분석.

    지표: 감염 노드의 코사인 유사도 분산
    가설: Fake는 초기부터 유사도 분산이 높을 것
    """
    results = {
        'time': [],
        'normal_edge_sim_mean': [],
        'fake_edge_sim_mean': [],
        'divergence_point': None,
    }

    min_len = min(len(normal_hist['t']), len(fake_hist['t']))
    for i in range(min_len):
        t = normal_hist['t'][i]
        u_n = normal_hist['u_history'][i]
        u_f = fake_hist['u_history'][i]

        # 감염된 노드의 평균값 차이
        n_std = np.std(u_n)
        f_std = np.std(u_f)

        results['time'].append(t)
        results['normal_edge_sim_mean'].append(n_std)
        results['fake_edge_sim_mean'].append(f_std)

        # 처음으로 10% 이상 차이나는 시점
        if results['divergence_point'] is None and abs(n_std - f_std) > 0.005:
            results['divergence_point'] = t

    return results


# ============================================================
#  시각화
# ============================================================

def plot_comprehensive_analysis(results: dict, n: int, pos: np.ndarray,
                                edges: List[Tuple[int, int]],
                                cos_sim: np.ndarray,
                                early_det: dict,
                                save_path: str):
    """8-panel 종합 분석 시각화."""

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # ---- Panel 0: 감염 곡선 비교 (전략별) ----
    ax0 = fig.add_subplot(gs[0, 0])

    strategies = [
        ('normal', 'Normal Content', 'green', '-'),
        ('fake_no_intervention', 'Fake (No Action)', 'red', '-'),
        ('hub_removal', 'Fake + Hub Removal', 'blue', '--'),
        ('edge_filter', 'Fake + Edge Filter', 'purple', '--'),
        ('early_detection', 'Fake + Early Detect', 'orange', '-.'),
    ]

    for key, label, color, style in strategies:
        if key in results:
            h = results[key]
            ax0.plot(h['t'][:len(h['infected_count'])],
                     h['infected_count'], color=color, linestyle=style,
                     linewidth=2, label=label)

    # 랜덤 제거 평균
    if 'random_removal_counts' in results:
        t_rand = results['random_removal_t']
        c_rand = results['random_removal_counts']
        min_len = min(len(t_rand), len(c_rand))
        ax0.plot(t_rand[:min_len], c_rand[:min_len],
                 color='gray', linestyle=':', linewidth=2,
                 label='Fake + Random Removal')

    # 조기 탐지 시점 마커
    if results.get('early_detection', {}).get('trigger_time'):
        tt = results['early_detection']['trigger_time']
        ax0.axvline(tt, color='orange', linestyle=':', alpha=0.6)
        ax0.annotate(f'Trigger\nt={tt:.2f}', xy=(tt, ax0.get_ylim()[1] * 0.5),
                     fontsize=8, color='orange', fontweight='bold')

    ax0.set_xlabel('Time')
    ax0.set_ylabel('Infected Nodes')
    ax0.set_title('Intervention Strategy Comparison', fontweight='bold')
    ax0.legend(fontsize=7, loc='lower right')
    ax0.grid(True, alpha=0.3)

    # ---- Panel 1: 확산 속도 (감염률 미분) ----
    ax1 = fig.add_subplot(gs[0, 1])
    for key, label, color, style in strategies[:3]:
        if key in results:
            h = results[key]
            counts = np.array(h['infected_count'], dtype=float)
            times = np.array(h['t'][:len(counts)])
            if len(counts) > 1:
                rate = np.diff(counts) / np.diff(times)
                # 스무딩
                window = min(5, len(rate))
                if window > 1:
                    kernel = np.ones(window) / window
                    rate = np.convolve(rate, kernel, mode='same')
                ax1.plot(times[1:], rate, color=color, linestyle=style,
                         linewidth=2, label=label)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Infection Rate (d/dt)')
    ax1.set_title('Spread Velocity', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: 조기 구분 시점 ----
    ax2 = fig.add_subplot(gs[0, 2])
    if early_det['time']:
        ax2.plot(early_det['time'], early_det['normal_edge_sim_mean'],
                 'g-', linewidth=2, label='Normal (std)')
        ax2.plot(early_det['time'], early_det['fake_edge_sim_mean'],
                 'r--', linewidth=2, label='Fake (std)')
        if early_det['divergence_point']:
            ax2.axvline(early_det['divergence_point'], color='navy',
                        linestyle=':', linewidth=2)
            ax2.annotate(f'Divergence\nt={early_det["divergence_point"]:.3f}',
                         xy=(early_det['divergence_point'], ax2.get_ylim()[0]),
                         fontsize=9, color='navy', fontweight='bold',
                         xytext=(10, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='navy'))
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Infection Std Dev')
    ax2.set_title('Early Detection Window', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: 감염 전파 엣지 유사도 분포 ----
    ax3 = fig.add_subplot(gs[1, 0])
    normal_sims = results['normal'].get('edge_sim_used', [])
    fake_sims = results['fake_no_intervention'].get('edge_sim_used', [])

    if normal_sims:
        ax3.hist(normal_sims, bins=30, alpha=0.6, color='green',
                 label=f'Normal (mean={np.mean(normal_sims):.3f})',
                 density=True)
    if fake_sims:
        ax3.hist(fake_sims, bins=30, alpha=0.6, color='red',
                 label=f'Fake (mean={np.mean(fake_sims):.3f})',
                 density=True)
    ax3.set_xlabel('Edge Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Transmission Edge Similarity', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: 차단 효과 비교 (최종 감염 수) ----
    ax4 = fig.add_subplot(gs[1, 1])
    strategy_names = []
    final_counts = []
    colors_bar = []

    bar_strategies = [
        ('normal', 'Normal\n(baseline)', '#22c55e'),
        ('fake_no_intervention', 'Fake\n(no action)', '#ef4444'),
        ('hub_removal', 'Hub\nRemoval', '#3b82f6'),
        ('edge_filter', 'Edge\nFilter', '#a855f7'),
        ('early_detection', 'Early\nDetect', '#f97316'),
    ]

    for key, label, color in bar_strategies:
        if key in results:
            h = results[key]
            strategy_names.append(label)
            final_counts.append(h['infected_count'][-1])
            colors_bar.append(color)

    if 'random_removal_counts' in results:
        strategy_names.append('Random\nRemoval')
        final_counts.append(results['random_removal_counts'][-1])
        colors_bar.append('#6b7280')

    bars = ax4.bar(strategy_names, final_counts, color=colors_bar, alpha=0.8,
                   edgecolor='white', linewidth=1.5)

    # 감소율 표시
    if len(final_counts) > 1:
        baseline = final_counts[1]  # fake no intervention
        for i, (bar, count) in enumerate(zip(bars, final_counts)):
            if i > 1:  # intervention strategies
                reduction = (1 - count / baseline) * 100
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'-{reduction:.0f}%', ha='center', va='bottom',
                         fontsize=9, fontweight='bold', color=colors_bar[i])

    ax4.set_ylabel('Final Infected Nodes')
    ax4.set_title('Intervention Effectiveness', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # ---- Panel 5: 감염 순서 vs 중심성 ----
    ax5 = fig.add_subplot(gs[1, 2])
    inf_result = results.get('influencer_result')
    if inf_result:
        centrality = inf_result['centrality']

        for key, label, color in [('normal', 'Normal', 'green'),
                                   ('fake_no_intervention', 'Fake', 'red')]:
            h = results[key]
            inf_time = h['infection_time']
            mask = inf_time < np.inf
            if np.any(mask):
                ax5.scatter(centrality[mask], inf_time[mask],
                            c=color, alpha=0.5, s=30, label=label)

        ax5.set_xlabel('Eigenvector Centrality')
        ax5.set_ylabel('Infection Time')
        ax5.set_title('Centrality vs Infection Time', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # ---- Panel 6: 그래프 + 차단 노드 표시 ----
    ax6 = fig.add_subplot(gs[2, 0])
    for i, j in edges:
        ax6.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                 'gray', alpha=0.1, linewidth=0.5)

    u_final = results['fake_no_intervention']['u_history'][-1]
    scatter6 = ax6.scatter(pos[:, 0], pos[:, 1], c=u_final,
                           cmap='Reds', s=40, edgecolors='white',
                           linewidth=0.5, zorder=5)

    hub_nodes = results.get('hub_nodes', [])
    if hub_nodes:
        ax6.scatter(pos[hub_nodes, 0], pos[hub_nodes, 1], s=200,
                    facecolors='none', edgecolors='blue', linewidth=3,
                    zorder=6, label='Hub (to block)')
    ax6.set_title('Fake Spread + Critical Nodes', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.set_xticks([])
    ax6.set_yticks([])

    # ---- Panel 7: 허브 차단 후 그래프 ----
    ax7 = fig.add_subplot(gs[2, 1])
    for i, j in edges:
        if i in hub_nodes or j in hub_nodes:
            continue
        ax7.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                 'gray', alpha=0.1, linewidth=0.5)

    u_hub = results['hub_removal']['u_history'][-1]
    ax7.scatter(pos[:, 0], pos[:, 1], c=u_hub,
                cmap='Reds', s=40, edgecolors='white',
                linewidth=0.5, zorder=5)
    ax7.scatter(pos[hub_nodes, 0], pos[hub_nodes, 1], s=200,
                marker='x', c='blue', linewidth=3, zorder=6,
                label='Blocked')
    ax7.set_title('After Hub Removal', fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.set_xticks([])
    ax7.set_yticks([])

    # ---- Panel 8: 요약 통계 테이블 ----
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    fake_final = results['fake_no_intervention']['infected_count'][-1]

    table_data = [
        ['Strategy', 'Final Infected', 'Reduction', 'Cost'],
        ['No Action', str(fake_final), '—', '—'],
    ]

    for key, label, cost in [
        ('hub_removal', 'Hub Removal (5)', '5 nodes'),
        ('edge_filter', 'Edge Filter', f'{results.get("edge_threshold", 0):.2f} threshold'),
        ('early_detection', 'Early Detection', '5 nodes (delayed)'),
    ]:
        if key in results:
            final = results[key]['infected_count'][-1]
            red = f'-{(1 - final / fake_final) * 100:.1f}%'
            table_data.append([label, str(final), red, cost])

    if 'random_removal_counts' in results:
        final_rand = int(results['random_removal_counts'][-1])
        red_rand = f'-{(1 - final_rand / fake_final) * 100:.1f}%'
        table_data.append(['Random (5 avg)', str(final_rand), red_rand, '5 nodes'])

    table = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # 헤더 스타일
    for j in range(4):
        table[0, j].set_facecolor('#1e293b')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        table[i, 0].set_text_props(fontweight='bold')

    fig.suptitle('Fake News Diffusion — Intervention Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================
#  메인
# ============================================================

def main():
    print("=" * 60)
    print("  DIFFUSION INTERVENTION SIMULATOR")
    print("  Real vs Fake Detection & Blocking Strategies")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 파라미터 ----
    N = 100
    M = 3
    FEAT_DIM = 8
    DT = 0.002
    STEPS = 1500
    DIFF_COEFF = 0.15
    SEED = 42

    # ---- 1. 그래프 생성 ----
    print("\n[1] 그래프 생성...")
    edges = barabasi_albert_graph(N, M, seed=SEED)
    adj = adjacency_matrix(N, edges)
    features = generate_user_features(N, FEAT_DIM, seed=SEED)
    cos_sim = cosine_similarity_matrix(features)
    print(f"    노드: {N}, 엣지: {len(edges)}")

    # ---- 2. 인플루언서 분석 ----
    print("[2] 인플루언서 분석...")
    L_cos = cosine_laplacian(adj, cos_sim)
    inf_result = find_influencers(L_cos, adj, cos_sim, top_k=5)
    seed_nodes = list(inf_result['top_influencers'][:2])
    print(f"    시드 유저: {seed_nodes}")
    print(f"    Top-5 허브: {list(inf_result['top_influencers'])}")

    # ---- 3. 차단 전략 시뮬레이션 ----
    print("[3] 차단 전략 시뮬레이션 (6가지 전략)...")
    results = intervention_simulation(
        N, edges, adj, cos_sim, features,
        seed_nodes, DT, STEPS, DIFF_COEFF)

    # ---- 4. 조기 구분 분석 ----
    print("[4] 조기 구분 시점 분석...")
    early_det = early_detection_analysis(
        results['normal'], results['fake_no_intervention'], N)
    if early_det['divergence_point']:
        print(f"    패턴 분기 시점: t={early_det['divergence_point']:.3f}")
    else:
        print("    분기 시점 미검출")

    # ---- 5. 감염 경로 유사도 분석 ----
    print("[5] 감염 경로 유사도 분석...")
    normal_sims = results['normal']['edge_sim_used']
    fake_sims = results['fake_no_intervention']['edge_sim_used']
    if normal_sims:
        print(f"    Normal — 전파 엣지 평균 유사도: {np.mean(normal_sims):.4f}")
    if fake_sims:
        print(f"    Fake   — 전파 엣지 평균 유사도: {np.mean(fake_sims):.4f}")

    # ---- 6. 시각화 ----
    print("[6] 시각화...")
    pos = compute_layout(N, edges, seed=SEED)
    plot_comprehensive_analysis(
        results, N, pos, edges, cos_sim, early_det,
        save_path=os.path.join(OUTPUT_DIR, 'intervention_analysis.png'))

    # ---- 결과 요약 ----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    fake_final = results['fake_no_intervention']['infected_count'][-1]
    print(f"\n  [확산 패턴 차이]")
    if normal_sims and fake_sims:
        print(f"    전파 엣지 유사도 — Normal: {np.mean(normal_sims):.4f}, "
              f"Fake: {np.mean(fake_sims):.4f}")
    if early_det['divergence_point']:
        total_time = STEPS * DT
        pct = early_det['divergence_point'] / total_time * 100
        print(f"    조기 구분 가능 시점: t={early_det['divergence_point']:.3f} "
              f"(전체의 {pct:.1f}%)")

    print(f"\n  [차단 전략 효과] (baseline: {fake_final} nodes infected)")
    for key, label in [('hub_removal', '허브 제거'),
                        ('edge_filter', '엣지 필터'),
                        ('early_detection', '조기 탐지')]:
        if key in results:
            final = results[key]['infected_count'][-1]
            red = (1 - final / fake_final) * 100
            print(f"    {label}: {final} nodes ({red:+.1f}%)")

    if 'random_removal_counts' in results:
        rand_final = int(results['random_removal_counts'][-1])
        red = (1 - rand_final / fake_final) * 100
        print(f"    랜덤 제거: {rand_final} nodes ({red:+.1f}%) ← 기준선")

    print(f"\n  출력: {OUTPUT_DIR}/intervention_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
