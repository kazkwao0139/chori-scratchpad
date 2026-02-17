"""ルルーシュ 페르소나 분리 — LLM 퍼플렉시티 기반
Qwen2.5-3B로 각 대사 윈도우의 bits/char를 계산하고,
LLM 특징으로 클러스터링하여 페르소나를 분리한다.
"""

import json
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
# 1. LLM 로드
# ══════════════════════════════════════
print("=" * 60)
print("Qwen2.5-3B 로딩...")
print("=" * 60)

MODEL_NAME = "Qwen/Qwen2.5-3B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()
print("모델 로드 완료")


def compute_window_features(text: str) -> dict:
    """텍스트의 LLM 기반 특징 추출
    - bits_per_char: 평균 bits/char
    - bits_std: 토큰별 bits 표준편차 (놀라움의 변동)
    - max_surprise: 최대 놀라움 (가장 예측 불가능한 토큰)
    - low_surprise_ratio: 낮은 놀라움 비율 (뻔한 토큰 비율)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] < 3:
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 각 토큰의 log prob
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # nats → bits
    token_bits = -token_log_probs[0].float().cpu().numpy() / np.log(2)

    # bits/char
    text_bytes = len(text.encode('utf-8'))
    total_bits = float(token_bits.sum())
    bits_per_char = total_bits / max(len(text), 1)

    return {
        'bits_per_char': bits_per_char,
        'bits_per_token': float(np.mean(token_bits)),
        'bits_std': float(np.std(token_bits)),
        'max_surprise': float(np.max(token_bits)),
        'median_surprise': float(np.median(token_bits)),
        'low_surprise_ratio': float(np.mean(token_bits < 2.0)),
        'high_surprise_ratio': float(np.mean(token_bits > 8.0)),
    }


# ══════════════════════════════════════
# 2. 윈도우별 LLM 특징 추출
# ══════════════════════════════════════
print(f"\n{len(lelouch_lines)}줄, 윈도우 크기={WINDOW}, 스트라이드={STRIDE}")

windows_features = []
window_meta = []
window_texts = []

total_windows = (len(lelouch_lines) - WINDOW) // STRIDE + 1
print(f"총 {total_windows}개 윈도우 처리 중...")

for idx, i in enumerate(range(0, len(lelouch_lines) - WINDOW + 1, STRIDE)):
    chunk_lines = lelouch_lines[i:i + WINDOW]
    chunk_text = '\n'.join(chunk_lines)

    # LLM 특징
    llm_feat = compute_window_features(chunk_text)
    if llm_feat is None:
        continue

    # zlib 특징도 추가
    raw = chunk_text.encode('utf-8')
    zlib_ent = len(zlib.compress(raw, 9)) / max(len(raw), 1)

    # 문체 특징
    avg_len = np.mean([len(l) for l in chunk_lines])
    q_rate = sum(1 for l in chunk_lines if '？' in l or '?' in l) / WINDOW
    ex_rate = sum(1 for l in chunk_lines if '！' in l or '!' in l) / WINDOW
    el_rate = sum(1 for l in chunk_lines if '…' in l or '―' in l) / WINDOW

    feature = {
        **llm_feat,
        'zlib_entropy': zlib_ent,
        'avg_len': avg_len,
        'question_rate': q_rate,
        'exclaim_rate': ex_rate,
        'ellipsis_rate': el_rate,
    }
    windows_features.append(feature)

    pos = (i + WINDOW / 2) / len(lelouch_lines)
    meta_items = lelouch_detail[i:i + WINDOW]
    window_meta.append({
        'pos': pos,
        'season': meta_items[0]['season'],
        'episode': meta_items[0]['episode'],
        'sample': chunk_lines[:3],
    })
    window_texts.append(chunk_text)

    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{total_windows} "
              f"(bpc={llm_feat['bits_per_char']:.3f}, zlib={zlib_ent:.3f})")

print(f"\n처리 완료: {len(windows_features)}개 윈도우")

# ══════════════════════════════════════
# 3. 클러스터링
# ══════════════════════════════════════
# 특징 행렬 구성
feature_keys = ['bits_per_char', 'bits_per_token', 'bits_std', 'max_surprise',
                'median_surprise', 'low_surprise_ratio', 'high_surprise_ratio',
                'zlib_entropy', 'avg_len', 'question_rate', 'exclaim_rate', 'ellipsis_rate']

X = np.array([[f[k] for k in feature_keys] for f in windows_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("LLM 특징 기반 클러스터링")
print("=" * 60)

k_range = range(2, 8)
silhouette_scores = []
best_labels = {}

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    best_labels[k] = labels
    print(f"  k={k}: silhouette={sil:.4f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n최적 k = {best_k} (silhouette = {max(silhouette_scores):.4f})")

# ══════════════════════════════════════
# 4. k=3 상세 분석 (가설: 제로/학생/사적)
# ══════════════════════════════════════
for analysis_k in [best_k, 3]:
    labels = best_labels[analysis_k]
    print(f"\n{'='*60}")
    print(f"k={analysis_k} 클러스터 상세 분석")
    print(f"{'='*60}")

    for c in range(analysis_k):
        mask = labels == c
        cluster_feat = [windows_features[i] for i in range(len(labels)) if mask[i]]
        cluster_meta = [window_meta[i] for i in range(len(labels)) if mask[i]]

        bpc = [f['bits_per_char'] for f in cluster_feat]
        bpt = [f['bits_per_token'] for f in cluster_feat]
        bstd = [f['bits_std'] for f in cluster_feat]
        zlib_e = [f['zlib_entropy'] for f in cluster_feat]
        q_r = [f['question_rate'] for f in cluster_feat]
        ex_r = [f['exclaim_rate'] for f in cluster_feat]
        el_r = [f['ellipsis_rate'] for f in cluster_feat]
        avg_l = [f['avg_len'] for f in cluster_feat]

        season_dist = Counter(m['season'] for m in cluster_meta)
        ep_dist = Counter(f"{m['season']}_E{m['episode']:02d}" for m in cluster_meta)
        top_eps = ep_dist.most_common(5)

        print(f"\n  Cluster {c} ({sum(mask)}개, {sum(mask)/len(labels)*100:.1f}%)")
        print(f"    LLM bits/char: {np.mean(bpc):.3f} ± {np.std(bpc):.3f}")
        print(f"    LLM bits/tok:  {np.mean(bpt):.3f} ± {np.std(bpt):.3f}")
        print(f"    LLM bits σ:    {np.mean(bstd):.3f} ± {np.std(bstd):.3f}")
        print(f"    zlib entropy:  {np.mean(zlib_e):.4f} ± {np.std(zlib_e):.4f}")
        print(f"    문장길이:      {np.mean(avg_l):.1f}자")
        print(f"    의문문율:      {np.mean(q_r):.2f}")
        print(f"    감탄문율:      {np.mean(ex_r):.2f}")
        print(f"    여운율(…―):    {np.mean(el_r):.2f}")
        print(f"    시즌: {dict(season_dist)}")
        print(f"    주요 EP: {top_eps}")
        print(f"    샘플:")
        for m in cluster_meta[:4]:
            print(f"      [{m['season']}_E{m['episode']:02d}] " +
                  ' / '.join(f'「{s}」' for s in m['sample']))


# ══════════════════════════════════════
# 5. 그래프
# ══════════════════════════════════════
COLORS = ['#8B0000', '#1E90FF', '#32CD32', '#FF4500', '#9B59B6', '#E67E22', '#2F4F4F']

# 5-1. 실루엣 비교 (zlib vs LLM)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(k_range), silhouette_scores, 'o-', color='#8B0000', linewidth=2.5,
        markersize=10, label='LLM features', zorder=10)

# zlib 기준도 (이전 결과)
zlib_sils = [0.028, 0.013, 0.020, 0.010, 0.008, 0.010]  # 이전 분석 결과
ax.plot(list(k_range), zlib_sils, 's--', color='#999', linewidth=2,
        markersize=8, label='zlib features (char freq)', alpha=0.7)

ax.axvline(x=best_k, color='red', linestyle=':', alpha=0.4)
ax.set_xlabel('k (number of personas)', fontsize=13, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
ax.set_title('Lelouch Persona Count: LLM vs zlib features\n'
             'Higher = cleaner persona separation',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.2)
for k, s in zip(k_range, silhouette_scores):
    ax.annotate(f'{s:.3f}', (k, s), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold', color='#8B0000')
plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_llm_vs_zlib_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: code_geass_llm_vs_zlib_k.png")

# 5-2. PCA 2D 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for analysis_k in [best_k, 3]:
    labels = best_labels[analysis_k]
    fig, ax = plt.subplots(figsize=(12, 8))

    for c in range(analysis_k):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[c], s=25, alpha=0.5, label=f'Persona {c}')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Lelouch Personas in LLM Feature Space (k={analysis_k})\n'
                 f'PCA projection of Qwen2.5-3B perplexity features',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f'D:/game-portfolio-main/code_geass_llm_persona_pca_k{analysis_k}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: code_geass_llm_persona_pca_k{analysis_k}.png")

# 5-3. 타임라인
for analysis_k in [best_k, 3]:
    labels = best_labels[analysis_k]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    positions = [m['pos'] for m in window_meta]
    bpc_values = [f['bits_per_char'] for f in windows_features]

    for c in range(analysis_k):
        mask = labels == c
        c_pos = [positions[i] for i in range(len(labels)) if mask[i]]
        c_bpc = [bpc_values[i] for i in range(len(labels)) if mask[i]]
        ax1.scatter(c_pos, c_bpc, c=COLORS[c], s=20, alpha=0.5,
                    label=f'Persona {c}')

    # S1|R2 경계
    s1_count = sum(1 for item in lelouch_detail if item['season'] == 'S1')
    boundary = s1_count / len(lelouch_lines)
    ax1.axvline(x=boundary, color='red', linewidth=2, alpha=0.4)
    ax1.text(boundary, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else max(bpc_values),
             ' S1|R2', fontsize=11, color='red', fontweight='bold', va='top')

    ax1.set_ylabel('LLM bits/char', fontsize=12, fontweight='bold')
    ax1.set_title(f'Lelouch Persona Timeline — LLM Perplexity (k={analysis_k})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.15)

    # 하단: 페르소나 비율 변화
    # 이동 평균으로 각 페르소나의 지배율
    grid = np.linspace(0.02, 0.98, 200)
    bandwidth = 0.05
    for c in range(analysis_k):
        mask = labels == c
        c_pos = np.array([positions[i] for i in range(len(labels)) if mask[i]])
        ratio = np.array([np.sum(np.abs(c_pos - g) < bandwidth) for g in grid])
        all_count = np.array([np.sum(np.abs(np.array(positions) - g) < bandwidth) for g in grid])
        ratio = ratio / np.maximum(all_count, 1)
        ax2.plot(grid, ratio, color=COLORS[c], linewidth=2, alpha=0.7,
                 label=f'Persona {c}')
        ax2.fill_between(grid, ratio, alpha=0.1, color=COLORS[c])

    ax2.axvline(x=boundary, color='red', linewidth=2, alpha=0.4)
    ax2.set_xlabel('Story progression', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Persona ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Persona dominance over time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'D:/game-portfolio-main/code_geass_llm_persona_timeline_k{analysis_k}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: code_geass_llm_persona_timeline_k{analysis_k}.png")

# 결과 저장
results = {
    'silhouette_scores': {k: s for k, s in zip(k_range, silhouette_scores)},
    'best_k': best_k,
    'windows_count': len(windows_features),
    'features': windows_features,
    'meta': window_meta,
}
with open('D:/game-portfolio-main/code_geass_llm_persona_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("\nSaved: code_geass_llm_persona_results.json")
print("\n완료!")
