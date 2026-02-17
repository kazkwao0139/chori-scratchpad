"""기생충 — 전체 캐릭터 서프라이즈 타임라인 (하나의 그래프)
zlib + LLM을 겹쳐서, 영화 어디서 누가 터지는지 한 눈에."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# ── 한글 폰트 ──
for name in ['Malgun Gothic', 'NanumGothic']:
    if [f for f in fm.fontManager.ttflist if name in f.name]:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──
with open('D:/game-portfolio-main/parasite_llm_analysis.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

with open('D:/game-portfolio-main/parasite_anomalies.json', 'r', encoding='utf-8') as f:
    zlib_anomalies = json.load(f)

with open('D:/game-portfolio-main/parasite_llm_anomalies.json', 'r', encoding='utf-8') as f:
    llm_anomalies = json.load(f)

MAIN_CHARS = ['기우', '기택', '연교', '충숙', '동익', '기정', '문광', '근세']
CHAR_COLORS = {
    '기우': '#E74C3C', '기택': '#C0392B', '충숙': '#E67E22', '기정': '#F39C12',
    '연교': '#3498DB', '동익': '#2980B9', '문광': '#8E44AD', '근세': '#9B59B6',
}

# ══════════════════════════════════════
# 그래프 1: 전체 캐릭터 LLM 퍼플렉시티 오버레이
# ══════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8))

# 각 캐릭터의 LLM 타임라인을 하나에 겹침
for char in MAIN_CHARS:
    if char not in llm_data:
        continue
    r = llm_data[char]
    pos = r['timeline']['positions']
    bpc = r['timeline']['bpc']
    color = CHAR_COLORS.get(char, 'gray')

    # z-score 정규화 (캐릭터간 스케일 맞추기)
    bpc_arr = np.array(bpc)
    mean_b = np.mean(bpc_arr)
    std_b = np.std(bpc_arr)
    if std_b > 0:
        z_bpc = (bpc_arr - mean_b) / std_b
    else:
        z_bpc = np.zeros_like(bpc_arr)

    ax.plot(pos, z_bpc, color=color, linewidth=2, alpha=0.7, label=char)

    # 피크 (z > 1.5) 표시
    for i, (p, z) in enumerate(zip(pos, z_bpc)):
        if z > 1.5:
            ax.scatter([p], [z], color=color, s=100, zorder=10,
                       edgecolors='white', linewidth=1.5)
            ax.annotate(char, (p, z), xytext=(3, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=color)

# 영화 구조 주석
plot_points = [
    (0.15, '기생 시작\n(기우 과외)'),
    (0.30, '가족 침투\n완료'),
    (0.50, '폭풍 전야\n(파티 계획)'),
    (0.65, '지하실\n발각'),
    (0.75, '폭우/\n반지하 침수'),
    (0.85, '가든 파티\n학살'),
    (0.95, '에필로그'),
]
for x, label in plot_points:
    ax.axvline(x=x, color='#BDC3C7', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(x, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -2.5, label,
            ha='center', va='top', fontsize=8, color='#7F8C8D',
            style='italic')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=12, fontweight='bold')
ax.set_ylabel('서프라이즈 z-score →', fontsize=12, fontweight='bold')
ax.set_title('기생충 — 누가, 언제 터지는가?\n(캐릭터별 LLM 퍼플렉시티 z-score 오버레이)',
             fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, ncol=4)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_surprise_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_surprise_overlay.png")


# ══════════════════════════════════════
# 그래프 2: 합산 서프라이즈 (영화 전체의 긴장도)
# ══════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# 공통 시간 그리드에 보간
grid = np.linspace(0.05, 0.95, 200)
all_z_curves = []

for char in MAIN_CHARS:
    if char not in llm_data:
        continue
    r = llm_data[char]
    pos = np.array(r['timeline']['positions'])
    bpc = np.array(r['timeline']['bpc'])

    # z-score
    mean_b = np.mean(bpc)
    std_b = np.std(bpc)
    if std_b > 0:
        z_bpc = (bpc - mean_b) / std_b
    else:
        continue

    # 보간
    z_interp = np.interp(grid, pos, z_bpc)
    all_z_curves.append(z_interp)

all_z = np.array(all_z_curves)

# 상단: 합산 서프라이즈 (= 영화의 총 긴장도)
total_surprise = np.mean(all_z, axis=0)
# 최대 기여 캐릭터 추적
max_char_idx = np.argmax(all_z, axis=0)
char_list = [c for c in MAIN_CHARS if c in llm_data]

ax1.fill_between(grid, total_surprise, alpha=0.3, color='#E74C3C')
ax1.plot(grid, total_surprise, color='#E74C3C', linewidth=2.5)
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# 피크에 누가 터졌는지 표시
from scipy.signal import find_peaks
peaks, properties = find_peaks(total_surprise, height=0.3, distance=10, prominence=0.2)
for pi in peaks:
    dominant_char = char_list[max_char_idx[pi]]
    color = CHAR_COLORS.get(dominant_char, 'gray')
    ax1.scatter([grid[pi]], [total_surprise[pi]], color=color, s=120, zorder=10,
                edgecolors='white', linewidth=2)
    ax1.annotate(f'{dominant_char}\n({int(grid[pi]*100)}%)',
                 (grid[pi], total_surprise[pi]),
                 xytext=(0, 15), textcoords='offset points',
                 fontsize=9, fontweight='bold', color=color, ha='center')

# 영화 구조 주석
for x, label in plot_points:
    ax1.axvline(x=x, color='#BDC3C7', linestyle=':', linewidth=1, alpha=0.6)
    ax1.text(x, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 1.5,
             label, ha='center', va='top', fontsize=8, color='#7F8C8D', style='italic')

ax1.set_ylabel('평균 서프라이즈 →', fontsize=12, fontweight='bold')
ax1.set_title('기생충 — 영화의 긴장도 곡선\n(전 캐릭터 LLM 서프라이즈 합산, 피크 = 주도 캐릭터 표시)',
              fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.15)

# 하단: 스택 영역 (누가 얼마나 기여하는지)
# 양수 부분만 스택
z_positive = np.clip(all_z, 0, None)

ax2.stackplot(grid, z_positive,
              labels=char_list,
              colors=[CHAR_COLORS.get(c, 'gray') for c in char_list],
              alpha=0.7)

for x, label in plot_points:
    ax2.axvline(x=x, color='#BDC3C7', linestyle=':', linewidth=1, alpha=0.6)

ax2.set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=12, fontweight='bold')
ax2.set_ylabel('서프라이즈 기여도 →', fontsize=12, fontweight='bold')
ax2.set_title('캐릭터별 서프라이즈 기여도 (스택)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, ncol=4)
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_surprise_total.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_surprise_total.png")


# ══════════════════════════════════════
# 그래프 3: zlib vs LLM 괴리 타임라인 (Character Depth)
# ══════════════════════════════════════

# zlib 타임라인도 필요 — 재계산
import zlib as zlib_mod

with open('D:/game-portfolio-main/parasite_dialogue.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

WINDOW = 15
STRIDE = 5

def text_entropy(text: str) -> float:
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib_mod.compress(raw, 9)) / len(raw)

fig, axes = plt.subplots(len(MAIN_CHARS), 1, figsize=(18, 3 * len(MAIN_CHARS)),
                         sharex=True)
fig.suptitle('기생충 — 표면(zlib) vs 심층(LLM) 괴리\n'
             '두 곡선이 벌어질수록 = "겉과 속이 다른" 순간',
             fontsize=14, fontweight='bold', y=0.995)

depth_scores = {}

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW or char not in llm_data:
        ax.text(0.5, 0.5, f'{char}: 데이터 부족', transform=ax.transAxes, ha='center')
        ax.set_ylabel(char, fontsize=11, fontweight='bold',
                      color=CHAR_COLORS.get(char, 'gray'))
        continue

    # zlib 타임라인
    zlib_pos = []
    zlib_ent = []
    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        e = text_entropy(chunk)
        pos = (i + WINDOW / 2) / len(lines)
        zlib_pos.append(pos)
        zlib_ent.append(e)

    # LLM 타임라인
    llm_pos = llm_data[char]['timeline']['positions']
    llm_bpc = llm_data[char]['timeline']['bpc']

    # 둘 다 z-score 정규화
    zlib_z = (np.array(zlib_ent) - np.mean(zlib_ent)) / (np.std(zlib_ent) + 1e-10)
    llm_z = (np.array(llm_bpc) - np.mean(llm_bpc)) / (np.std(llm_bpc) + 1e-10)

    # 공통 그리드에 보간
    common_grid = np.linspace(max(min(zlib_pos), min(llm_pos)),
                              min(max(zlib_pos), max(llm_pos)), 100)
    zlib_interp = np.interp(common_grid, zlib_pos, zlib_z)
    llm_interp = np.interp(common_grid, llm_pos, llm_z)

    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(common_grid, zlib_interp, color=color, linewidth=2, alpha=0.6,
            linestyle='--', label='zlib (표면)')
    ax.plot(common_grid, llm_interp, color=color, linewidth=2, alpha=0.9,
            label='LLM (심층)')

    # 괴리 영역 채우기
    gap = np.abs(llm_interp - zlib_interp)
    ax.fill_between(common_grid, zlib_interp, llm_interp,
                    alpha=0.2, color=color)

    # 괴리가 큰 지점 표시
    gap_peaks = np.where(gap > 1.5)[0]
    for gi in gap_peaks[::5]:  # 너무 많으면 간격 띄움
        ax.annotate('◆', (common_grid[gi], llm_interp[gi]),
                    fontsize=8, color='red', fontweight='bold', ha='center')

    # 상관계수 = Character Depth
    corr = np.corrcoef(zlib_interp, llm_interp)[0, 1]
    depth_scores[char] = round(1 - corr, 4)  # 낮은 상관 = 높은 depth

    ax.text(0.98, 0.95, f'r = {corr:.3f}\nDepth = {1-corr:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color,
                  rotation=0, labelpad=40)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 1)
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.15)

axes[-1].set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=11)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_depth_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_depth_gap.png")

# Character Depth 랭킹
print("\n" + "=" * 50)
print("Character Depth (1 - r) 랭킹")
print("  r = zlib-LLM 타임라인 상관계수")
print("  높을수록 = 겉과 속이 다른 캐릭터")
print("=" * 50)
for char, depth in sorted(depth_scores.items(), key=lambda x: -x[1]):
    bar = '█' * int(depth * 30)
    print(f"  {char:>4}  {depth:.4f}  {bar}")
