"""ルルーシュ 페르소나 분리 실험
대사를 윈도우로 쪼개고, 문자 빈도 벡터로 클러스터링.
각 클러스터 = 하나의 페르소나.
k=2,3,4,5에서 실루엣 스코어로 최적 k를 찾는다.
"""

import json
import zlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ── 폰트 ──
for name in ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Malgun Gothic']:
    if [f for f in fm.fontManager.ttflist if name in f.name]:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 ──
with open('D:/game-portfolio-main/code_geass_dialogue_ja.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

with open('D:/game-portfolio-main/code_geass_dialogue_ja_detail.json', 'r', encoding='utf-8') as f:
    detail = json.load(f)

lelouch_lines = dialogue['ルルーシュ']
lelouch_detail = detail['ルルーシュ']

WINDOW = 12
STRIDE = 4

# ══════════════════════════════════════
# 1. 윈도우별 특징 벡터 추출
# ══════════════════════════════════════
def text_entropy(text: str) -> float:
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def char_freq_vector(text: str, vocab: list) -> np.ndarray:
    """문자 빈도 벡터 (vocab 순서)"""
    counter = Counter(c for c in text if c.strip())
    total = sum(counter.values())
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counter.get(ch, 0) / total for ch in vocab])


# 전체 어휘 구축 (상위 200자)
all_text = '\n'.join(lelouch_lines)
char_counts = Counter(c for c in all_text if c.strip())
vocab = [ch for ch, _ in char_counts.most_common(200)]

print(f"ルルーシュ 총 {len(lelouch_lines)}줄")
print(f"어휘 크기: {len(vocab)}자 (상위 200)")

# 윈도우 특징 추출
windows = []
window_meta = []  # 각 윈도우의 메타 정보

for i in range(0, len(lelouch_lines) - WINDOW + 1, STRIDE):
    chunk_lines = lelouch_lines[i:i + WINDOW]
    chunk_text = '\n'.join(chunk_lines)

    # 특징: zlib 엔트로피 + 문자 빈도
    ent = text_entropy(chunk_text)
    freq = char_freq_vector(chunk_text, vocab)

    # 추가 특징
    avg_len = np.mean([len(l) for l in chunk_lines])
    question_rate = sum(1 for l in chunk_lines if '？' in l or '?' in l) / WINDOW
    exclaim_rate = sum(1 for l in chunk_lines if '！' in l or '!' in l) / WINDOW
    ellipsis_rate = sum(1 for l in chunk_lines if '…' in l or '―' in l) / WINDOW

    feature = np.concatenate([
        [ent, avg_len, question_rate, exclaim_rate, ellipsis_rate],
        freq
    ])
    windows.append(feature)

    # 메타
    pos = (i + WINDOW / 2) / len(lelouch_lines)
    meta_items = lelouch_detail[i:i + WINDOW]
    seasons = [m['season'] for m in meta_items]
    episodes = [m['episode'] for m in meta_items]
    window_meta.append({
        'pos': pos,
        'start_idx': i,
        'season': seasons[0],
        'episode': episodes[0],
        'sample': chunk_lines[:3],
    })

X = np.array(windows)
print(f"윈도우 수: {len(X)}")

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ══════════════════════════════════════
# 2. 최적 k 탐색
# ══════════════════════════════════════
print("\n" + "=" * 60)
print("실루엣 스코어로 최적 k 탐색")
print("=" * 60)

k_range = range(2, 8)
silhouette_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    inertias.append(km.inertia_)
    print(f"  k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n최적 k = {best_k} (silhouette = {max(silhouette_scores):.4f})")


# ══════════════════════════════════════
# 3. 최적 k로 클러스터링 + 분석
# ══════════════════════════════════════
for analysis_k in [best_k, 3]:  # best_k와 3(가설) 둘 다 분석
    print(f"\n{'='*60}")
    print(f"k={analysis_k} 클러스터링 분석")
    print(f"{'='*60}")

    km = KMeans(n_clusters=analysis_k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)

    for c in range(analysis_k):
        mask = labels == c
        cluster_windows = [window_meta[i] for i in range(len(labels)) if labels[i] == c]
        cluster_X = X[mask]

        # 클러스터 내부 통계
        ent_values = cluster_X[:, 0]  # 첫 번째 특징 = entropy
        avg_len_values = cluster_X[:, 1]
        q_rate = cluster_X[:, 2]
        ex_rate = cluster_X[:, 3]
        el_rate = cluster_X[:, 4]

        # 시즌/에피소드 분포
        season_counts = Counter(w['season'] for w in cluster_windows)
        ep_counts = Counter(f"{w['season']}_E{w['episode']:02d}" for w in cluster_windows)
        top_eps = ep_counts.most_common(5)

        print(f"\n  Cluster {c} ({sum(mask)}개 윈도우, {sum(mask)/len(labels)*100:.1f}%)")
        print(f"    zlib 평균: {np.mean(ent_values):.4f} ± {np.std(ent_values):.4f}")
        print(f"    문장길이:  {np.mean(avg_len_values):.1f}자 ± {np.std(avg_len_values):.1f}")
        print(f"    의문문율:  {np.mean(q_rate):.2f}")
        print(f"    감탄문율:  {np.mean(ex_rate):.2f}")
        print(f"    여운(…―):  {np.mean(el_rate):.2f}")
        print(f"    시즌분포:  {dict(season_counts)}")
        print(f"    주요 에피소드: {top_eps}")
        print(f"    샘플:")
        for w in cluster_windows[:3]:
            for line in w['sample']:
                print(f"      「{line}」")
            print()


# ══════════════════════════════════════
# 4. 그래프
# ══════════════════════════════════════

# 4-1. 실루엣 스코어
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(k_range), silhouette_scores, 'o-', color='#8B0000', linewidth=2, markersize=8)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('k (number of personas)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax1.set_title('How many personas does Lelouch have?', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.2)
for i, (k, s) in enumerate(zip(k_range, silhouette_scores)):
    ax1.annotate(f'{s:.3f}', (k, s), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')

ax2.plot(list(k_range), inertias, 'o-', color='#2F4F4F', linewidth=2, markersize=8)
ax2.set_xlabel('k', fontsize=12, fontweight='bold')
ax2.set_ylabel('Inertia (within-cluster sum)', fontsize=12, fontweight='bold')
ax2.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_persona_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: code_geass_persona_k.png")


# 4-2. 타임라인: 클러스터 색상으로 를르슈 대사 진행
CLUSTER_COLORS = ['#8B0000', '#1E90FF', '#32CD32', '#FF4500', '#9B59B6', '#E67E22', '#2F4F4F']

for analysis_k in [best_k, 3]:
    km = KMeans(n_clusters=analysis_k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(18, 5))
    positions = [w['pos'] for w in window_meta]
    entropies = X[:, 0]

    for c in range(analysis_k):
        mask = labels == c
        c_pos = [positions[i] for i in range(len(labels)) if mask[i]]
        c_ent = [entropies[i] for i in range(len(labels)) if mask[i]]
        ax.scatter(c_pos, c_ent, c=CLUSTER_COLORS[c], s=20, alpha=0.6,
                   label=f'Persona {c}')

    # 시즌 경계
    s1_count = sum(1 for item in lelouch_detail if item['season'] == 'S1')
    boundary = s1_count / len(lelouch_lines)
    ax.axvline(x=boundary, color='red', linestyle='-', linewidth=2, alpha=0.4)
    ax.text(boundary, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else max(entropies),
            ' S1|R2', fontsize=11, color='red', alpha=0.7, va='top', fontweight='bold')

    ax.set_xlabel('Story progression', fontsize=12, fontweight='bold')
    ax.set_ylabel('zlib entropy', fontsize=12, fontweight='bold')
    ax.set_title(f'Lelouch Persona Timeline (k={analysis_k})\n'
                 f'Each color = a distinct speech pattern cluster',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'D:/game-portfolio-main/code_geass_persona_timeline_k{analysis_k}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: code_geass_persona_timeline_k{analysis_k}.png")


# 4-3. 비교: 다른 캐릭터도 같은 분석
print("\n" + "=" * 60)
print("다른 캐릭터 vs ルルーシュ: 최적 k 비교")
print("=" * 60)

compare_chars = ['ルルーシュ', 'スザク', 'C.C.', 'カレン', 'シュナイゼル', 'コーネリア']
char_best_k = {}

for char in compare_chars:
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW * 3:
        continue

    # 윈도우 특징
    wins = []
    all_char_text = '\n'.join(lines)
    c_counts = Counter(c for c in all_char_text if c.strip())
    c_vocab = [ch for ch, _ in c_counts.most_common(200)]

    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        ent = text_entropy(chunk)
        freq = char_freq_vector(chunk, c_vocab)
        avg_l = np.mean([len(l) for l in lines[i:i+WINDOW]])
        q = sum(1 for l in lines[i:i+WINDOW] if '？' in l or '?' in l) / WINDOW
        ex = sum(1 for l in lines[i:i+WINDOW] if '！' in l or '!' in l) / WINDOW
        el = sum(1 for l in lines[i:i+WINDOW] if '…' in l or '―' in l) / WINDOW
        wins.append(np.concatenate([[ent, avg_l, q, ex, el], freq]))

    if len(wins) < 10:
        continue

    Xc = StandardScaler().fit_transform(np.array(wins))

    best_sil = -1
    best_ck = 2
    sils = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lab = km.fit_predict(Xc)
        s = silhouette_score(Xc, lab)
        sils.append(s)
        if s > best_sil:
            best_sil = s
            best_ck = k

    char_best_k[char] = (best_ck, best_sil, sils)
    print(f"  {char:>12}: best k={best_ck} (sil={best_sil:.4f})  "
          f"[{', '.join(f'k{k}={s:.3f}' for k, s in zip(range(2,8), sils))}]")

# 비교 차트
fig, ax = plt.subplots(figsize=(14, 6))
x = list(range(2, 8))
for char in compare_chars:
    if char not in char_best_k:
        continue
    _, _, sils = char_best_k[char]
    color = {
        'ルルーシュ': '#8B0000', 'スザク': '#1E90FF', 'C.C.': '#32CD32',
        'カレン': '#FF4500', 'シュナイゼル': '#2F4F4F', 'コーネリア': '#4B0082',
    }.get(char, 'gray')
    ax.plot(x, sils, 'o-', color=color, linewidth=2, markersize=6,
            label=f'{char} (best k={char_best_k[char][0]})', alpha=0.8)

ax.set_xlabel('k (number of personas)', fontsize=13, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
ax.set_title('How many personas does each character have?\n'
             'Higher silhouette = cleaner separation',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
ax.set_xticks(x)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_persona_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: code_geass_persona_comparison.png")
