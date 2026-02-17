"""기생충 캐릭터별 엔트로피/코사인 유사도 시간축 분석
— 어디서 캐릭터가 예측 불가능한 대사를 뱉는가?"""

import json
import zlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer

# ── 한글 폰트 ──
for name in ['Malgun Gothic', 'NanumGothic']:
    if [f for f in fm.fontManager.ttflist if name in f.name]:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 ──
with open('D:/game-portfolio-main/parasite_dialogue.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

MAIN_CHARS = ['기우', '기택', '연교', '충숙', '동익', '기정', '문광', '근세']
WINDOW = 15       # 슬라이딩 윈도우 크기 (대사 수)
STRIDE = 5        # 스트라이드
CHAR_COLORS = {
    '기우': '#E74C3C', '기택': '#C0392B', '충숙': '#E67E22', '기정': '#F39C12',
    '연교': '#3498DB', '동익': '#2980B9', '다혜': '#1ABC9C', '다송': '#16A085',
    '문광': '#8E44AD', '근세': '#9B59B6',
}


def text_entropy(text: str) -> float:
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sliding_entropy(lines, window=WINDOW, stride=STRIDE):
    """슬라이딩 윈도우 엔트로피"""
    positions = []
    entropies = []
    for i in range(0, len(lines) - window + 1, stride):
        chunk = '\n'.join(lines[i:i + window])
        e = text_entropy(chunk)
        # 위치를 0~1로 정규화 (영화 진행도)
        pos = (i + window / 2) / len(lines)
        positions.append(pos)
        entropies.append(e)
    return positions, entropies


def sliding_cosine(lines, model, window=WINDOW, stride=STRIDE):
    """슬라이딩 윈도우 코사인 유사도 (전체 평균 임베딩 대비)"""
    # 전체 대사의 평균 임베딩
    all_text = '\n'.join(lines)
    all_emb = model.encode([all_text])[0]

    positions = []
    similarities = []
    for i in range(0, len(lines) - window + 1, stride):
        chunk = '\n'.join(lines[i:i + window])
        chunk_emb = model.encode([chunk])[0]
        sim = cosine_sim(chunk_emb, all_emb)
        pos = (i + window / 2) / len(lines)
        positions.append(pos)
        similarities.append(sim)
    return positions, similarities


def find_anomalies(positions, values, n=3):
    """상위 n개 이상치(피크) 찾기"""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    # z-score 기준
    z_scores = np.abs(values - mean) / std if std > 0 else np.zeros_like(values)
    top_idx = np.argsort(z_scores)[-n:]
    return [(positions[i], values[i], z_scores[i]) for i in top_idx]


def get_dialogue_at_position(lines, pos, window=WINDOW):
    """특정 위치의 대사 반환"""
    idx = int(pos * len(lines) - window / 2)
    idx = max(0, min(idx, len(lines) - window))
    return lines[idx:idx + window]


print("모델 로딩...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ══════════════════════════════════════
# 그래프 1: 엔트로피 타임라인
# ══════════════════════════════════════
fig, axes = plt.subplots(len(MAIN_CHARS), 1, figsize=(14, 3 * len(MAIN_CHARS)),
                         sharex=True)
fig.suptitle('기생충 — 캐릭터별 엔트로피 변화 (영화 진행 순서)\n'
             '↑ 높을수록 예측 불가능한 말투', fontsize=14, fontweight='bold', y=0.995)

anomaly_report = {}

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW:
        ax.text(0.5, 0.5, f'{char}: 대사 부족', transform=ax.transAxes, ha='center')
        ax.set_ylabel(char, fontsize=11, fontweight='bold', color=CHAR_COLORS.get(char, 'gray'))
        continue

    pos, ent = sliding_entropy(lines)
    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(pos, ent, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, ent, alpha=0.15, color=color)

    # 평균선
    mean_e = np.mean(ent)
    ax.axhline(y=mean_e, color=color, linestyle='--', alpha=0.4, linewidth=1)

    # 이상치 표시
    anomalies = find_anomalies(pos, ent, n=2)
    for a_pos, a_val, a_z in anomalies:
        ax.scatter([a_pos], [a_val], color='red', s=80, zorder=10,
                   edgecolors='white', linewidth=1.5)
        ax.annotate(f'z={a_z:.1f}', (a_pos, a_val),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold')

    # 이상치 대사 저장
    char_anomalies = []
    for a_pos, a_val, a_z in anomalies:
        sample_lines = get_dialogue_at_position(lines, a_pos)
        char_anomalies.append({
            'position': round(a_pos, 2),
            'entropy': round(a_val, 4),
            'z_score': round(a_z, 2),
            'sample': sample_lines[:5]
        })
    anomaly_report[char] = char_anomalies

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color, rotation=0, labelpad=40)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=11)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_entropy_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_entropy_timeline.png")

# ══════════════════════════════════════
# 그래프 2: 코사인 유사도 타임라인
# ══════════════════════════════════════
fig, axes = plt.subplots(len(MAIN_CHARS), 1, figsize=(14, 3 * len(MAIN_CHARS)),
                         sharex=True)
fig.suptitle('기생충 — 캐릭터별 코사인 유사도 변화 (전체 평균 대비)\n'
             '↓ 낮을수록 평소와 다른 내용의 대사', fontsize=14, fontweight='bold', y=0.995)

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW:
        ax.text(0.5, 0.5, f'{char}: 대사 부족', transform=ax.transAxes, ha='center')
        ax.set_ylabel(char, fontsize=11, fontweight='bold', color=CHAR_COLORS.get(char, 'gray'))
        continue

    pos, sims = sliding_cosine(lines, model)
    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(pos, sims, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, sims, alpha=0.15, color=color)

    mean_s = np.mean(sims)
    ax.axhline(y=mean_s, color=color, linestyle='--', alpha=0.4, linewidth=1)

    # 최저점 (가장 "다른" 대사) 표시
    min_idx = np.argmin(sims)
    ax.scatter([pos[min_idx]], [sims[min_idx]], color='red', s=80, zorder=10,
               edgecolors='white', linewidth=1.5)
    ax.annotate('★ 전환점', (pos[min_idx], sims[min_idx]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=9, color='red', fontweight='bold')

    # 전환점 대사 저장
    sample = get_dialogue_at_position(lines, pos[min_idx])
    if char in anomaly_report:
        anomaly_report[char].append({
            'type': 'cosine_min',
            'position': round(pos[min_idx], 2),
            'cosine': round(sims[min_idx], 4),
            'sample': sample[:5]
        })

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color, rotation=0, labelpad=40)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=11)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_cosine_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_cosine_timeline.png")

# ══════════════════════════════════════
# 이상치 대사 리포트
# ══════════════════════════════════════
print("\n" + "=" * 60)
print("캐릭터별 이상 대사 (예측 불가능한 순간)")
print("=" * 60)
for char in MAIN_CHARS:
    if char not in anomaly_report:
        continue
    print(f"\n【{char}】")
    for a in anomaly_report[char]:
        pos_pct = int(a['position'] * 100)
        if 'entropy' in a:
            print(f"  📍 영화 {pos_pct}% 지점 | 엔트로피 이상치 (z={a.get('z_score', '?')})")
        else:
            print(f"  📍 영화 {pos_pct}% 지점 | 코사인 최저 (평소와 가장 다른 대사)")
        for line in a['sample']:
            print(f"     \"{line}\"")

# JSON 저장
with open('D:/game-portfolio-main/parasite_anomalies.json', 'w', encoding='utf-8') as f:
    json.dump(anomaly_report, f, ensure_ascii=False, indent=2)
print("\n결과 저장: parasite_anomalies.json")
