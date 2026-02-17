"""기생충 캐릭터 일관성 — 엔트로피σ vs 코사인σ 산점도"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 설정 ──
font_candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Gulim']
font_found = None
for name in font_candidates:
    matches = [f for f in fm.fontManager.ttflist if name in f.name]
    if matches:
        font_found = name
        break

if font_found:
    plt.rcParams['font.family'] = font_found
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──
with open('D:/game-portfolio-main/parasite_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

entropy = data['entropy']
cosine = data['cosine']

chars = sorted(set(entropy.keys()) & set(cosine.keys()),
               key=lambda c: -entropy[c]['total_lines'])

# ── 데이터 추출 ──
names = []
e_std = []
c_std = []
sizes = []
colors = []

# 가족 색상 구분
kim_family = {'기우', '기택', '충숙', '기정'}
park_family = {'연교', '동익', '다혜', '다송'}

for char in chars:
    names.append(char)
    e_std.append(entropy[char]['entropy_std'])
    c_std.append(cosine[char]['cosine_std'])
    sizes.append(entropy[char]['total_lines'])

    if char in kim_family:
        colors.append('#E74C3C')   # 빨강 — 김씨 가족
    elif char in park_family:
        colors.append('#3498DB')   # 파랑 — 박씨 가족
    else:
        colors.append('#95A5A6')   # 회색 — 기타

e_std = np.array(e_std)
c_std = np.array(c_std)
sizes = np.array(sizes)

# ── 그래프 ──
fig, ax = plt.subplots(figsize=(12, 8))

# 사분면 배경
e_thresh = 0.015
c_thresh = 0.02

ax.axvline(x=e_thresh, color='#BDC3C7', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=c_thresh, color='#BDC3C7', linestyle='--', linewidth=1, alpha=0.7)

# 사분면 라벨
ax.text(0.003, 0.16, '★ 아크 있는 일관된 캐릭터\n(말투 일관 + 내용 변화)',
        fontsize=9, color='#27AE60', alpha=0.8, style='italic')
ax.text(0.04, 0.16, '△ 불안정\n(말투+내용 둘다 변동)',
        fontsize=9, color='#E67E22', alpha=0.8, style='italic')
ax.text(0.003, 0.005, '● 안정적\n(변화 적음)',
        fontsize=9, color='#7F8C8D', alpha=0.8, style='italic')
ax.text(0.04, 0.005, '○ 평이\n(말투만 변동)',
        fontsize=9, color='#7F8C8D', alpha=0.8, style='italic')

# 산점도 — 버블 크기 = 대사 수
scatter = ax.scatter(e_std, c_std,
                     s=sizes * 1.5,   # 대사 수에 비례
                     c=colors,
                     alpha=0.7,
                     edgecolors='white',
                     linewidth=1.5,
                     zorder=5)

# 캐릭터 이름 라벨
for i, name in enumerate(names):
    offset_x = 0.001
    offset_y = 0.004
    # 겹침 방지 조정
    if name == '기택':
        offset_y = -0.008
    elif name == '문광':
        offset_x = 0.002
    elif name == '기정':
        offset_y = -0.008

    ax.annotate(name,
                (e_std[i], c_std[i]),
                xytext=(e_std[i] + offset_x, c_std[i] + offset_y),
                fontsize=11, fontweight='bold',
                color=colors[i],
                zorder=10)

# 축/제목
ax.set_xlabel('엔트로피 σ (말투 변동) →', fontsize=12, fontweight='bold')
ax.set_ylabel('코사인 유사도 σ (내용 변동) →', fontsize=12, fontweight='bold')
ax.set_title('기생충 — 캐릭터 일관성 분석\n(버블 크기 = 대사 수)', fontsize=15, fontweight='bold')

# 범례
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
           markersize=12, label='김씨 가족 (기택/충숙/기우/기정)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
           markersize=12, label='박씨 가족 (동익/연교/다혜/다송)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#95A5A6',
           markersize=12, label='기타 (문광/근세/민혁/가게사장)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.set_xlim(-0.005, max(e_std) + 0.02)
ax.set_ylim(-0.005, max(c_std) + 0.03)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_character_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장 완료: parasite_character_analysis.png")
