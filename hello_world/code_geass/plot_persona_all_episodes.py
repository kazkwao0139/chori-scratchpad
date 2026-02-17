"""ルルーシュ 페르소나 분포 — 전 50화 차트"""

import json
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for name in ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Malgun Gothic']:
    if [f for f in fm.fontManager.ttflist if name in f.name]:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

with open('D:/game-portfolio-main/code_geass_llm_persona_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

features = results['features']
meta = results['meta']

feature_keys = ['bits_per_char', 'bits_per_token', 'bits_std', 'max_surprise',
                'median_surprise', 'low_surprise_ratio', 'high_surprise_ratio',
                'zlib_entropy', 'avg_len', 'question_rate', 'exclaim_rate', 'ellipsis_rate']

X = np.array([[f[k] for k in feature_keys] for f in features])
X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = km.fit_predict(X_scaled)

# 전 에피소드 집계
episodes = []
for season in ['S1', 'R2']:
    for ep in range(1, 26):
        idx_list = [i for i, m in enumerate(meta)
                    if m['season'] == season and m['episode'] == ep]
        if not idx_list:
            continue
        dist = Counter(labels[i] for i in idx_list)
        total = len(idx_list)
        bpc_values = [features[i]['bits_per_char'] for i in idx_list]
        episodes.append({
            'season': season, 'ep': ep, 'total': total,
            'zero_pct': dist.get(0, 0) / total * 100,
            'emotion_pct': dist.get(1, 0) / total * 100,
            'student_pct': dist.get(2, 0) / total * 100,
            'bpc_mean': np.mean(bpc_values),
            'bpc_std': np.std(bpc_values),
        })

# 텍스트 출력
print(f"{'EP':>7} {'Win':>4} {'Zero%':>6} {'Emot%':>6} {'Stud%':>6} "
      f"{'BPC':>6} {'Dominant':>10}  Bar")
print('-' * 80)
for e in episodes:
    label = f"{e['season']} E{e['ep']:02d}"
    pcts = {'Zero': e['zero_pct'], 'Emotion': e['emotion_pct'], 'Student': e['student_pct']}
    dominant = max(pcts, key=pcts.get)
    z = int(e['zero_pct'] / 5)
    em = int(e['emotion_pct'] / 5)
    st = int(e['student_pct'] / 5)
    bar = '\033[91m' + '█' * z + '\033[94m' + '░' * em + '\033[92m' + '·' * st + '\033[0m'
    print(f"{label:>7} {e['total']:>4} {e['zero_pct']:>5.0f}% "
          f"{e['emotion_pct']:>5.0f}% {e['student_pct']:>5.0f}% "
          f"{e['bpc_mean']:>5.2f} {dominant:>10}  {bar}")

# ═══════════════════════════════
# 스택 바 차트
# ═══════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12),
                                gridspec_kw={'height_ratios': [3, 1]})

x_labels = [f"{e['season']}\nE{e['ep']:02d}" for e in episodes]
x = np.arange(len(episodes))

zero_pcts = [e['zero_pct'] for e in episodes]
emotion_pcts = [e['emotion_pct'] for e in episodes]
student_pcts = [e['student_pct'] for e in episodes]

ax1.bar(x, zero_pcts, color='#8B0000', alpha=0.85,
        label='Persona 0: Zero (command/speech)')
ax1.bar(x, emotion_pcts, bottom=zero_pcts, color='#1E90FF', alpha=0.85,
        label='Persona 1: Emotional (volatile/unpredictable)')
bottom2 = [z + e for z, e in zip(zero_pcts, emotion_pcts)]
ax1.bar(x, student_pcts, bottom=bottom2, color='#32CD32', alpha=0.85,
        label='Persona 2: Student (casual/everyday)')

# S1|R2 경계
s1_count = sum(1 for e in episodes if e['season'] == 'S1')
ax1.axvline(x=s1_count - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.6)
ax1.text(s1_count - 0.5, 103, ' S1 | R2 ', ha='center', fontsize=13,
         color='red', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 주요 에피소드 주석
annotations = {
    ('S1', 1): 'New Demon\nBorn',
    ('S1', 4): 'His Name\nis Zero',
    ('S1', 8): 'Black\nKnights',
    ('S1', 11): 'Narita',
    ('S1', 14): 'Geass vs\nGeass',
    ('S1', 16): 'Nunnally\nHostage',
    ('S1', 22): 'Bloodstained\nEuphy ★',
    ('S1', 25): 'Zero',
    ('R2', 1): 'Demon\nAwakens',
    ('R2', 8): '1 Million\nMiracles',
    ('R2', 13): 'Shirley\nDies',
    ('R2', 19): 'Betrayal',
    ('R2', 21): 'Ragnarok',
    ('R2', 25): 'Re; ★',
}
for i, e in enumerate(episodes):
    key = (e['season'], e['ep'])
    if key in annotations:
        ax1.annotate(annotations[key], (i, 105), ha='center', va='bottom',
                     fontsize=7.5, fontweight='bold', color='#333')

ax1.set_ylabel('Persona Distribution (%)', fontsize=14, fontweight='bold')
ax1.set_title('Code Geass — Which face does Lelouch wear?\n'
              'LLM-based persona clustering across all 50 episodes',
              fontsize=17, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, ncol=3)
ax1.set_xlim(-0.5, len(episodes) - 0.5)
ax1.set_ylim(0, 120)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=6.5, rotation=0)
ax1.grid(axis='y', alpha=0.15)

# 하단: BPC 변동 (dominant persona 색상)
bpc_means = [e['bpc_mean'] for e in episodes]
bpc_stds = [e['bpc_std'] for e in episodes]

dominant_colors = []
for e in episodes:
    pcts = [e['zero_pct'], e['emotion_pct'], e['student_pct']]
    dominant_colors.append(['#8B0000', '#1E90FF', '#32CD32'][np.argmax(pcts)])

ax2.bar(x, bpc_means, color=dominant_colors, alpha=0.7, width=0.7)
ax2.errorbar(x, bpc_means, yerr=bpc_stds, fmt='none', ecolor='gray',
             alpha=0.4, capsize=2)
ax2.axvline(x=s1_count - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.6)
ax2.axhline(y=np.mean(bpc_means), color='gray', linestyle='--', alpha=0.3)

ax2.set_xlabel('Episode', fontsize=14, fontweight='bold')
ax2.set_ylabel('LLM bits/char', fontsize=14, fontweight='bold')
ax2.set_title('LLM Perplexity per Episode (bar color = dominant persona)',
              fontsize=12, fontweight='bold')
ax2.set_xlim(-0.5, len(episodes) - 0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontsize=6.5, rotation=0)
ax2.grid(axis='y', alpha=0.15)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_persona_all_episodes.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: code_geass_persona_all_episodes.png')
