"""
Bruce Wayne 페르소나 분리 — LLM 퍼플렉시티 기반
BATMAN + WAYNE 대사를 합치고 Qwen2.5-3B로 클러스터링.
클러스터가 각본의 BATMAN/WAYNE 라벨과 자연스럽게 일치하는지 검증.

동일 파이프라인: analyze_lelouch_llm_persona.py
"""

import json
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = "D:/game-portfolio-main/SCRATCHPAD/hello_world/dark_knight"

# ── 데이터 로드 ──
with open(f'{BASE}/dark_knight_dialogue.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

with open(f'{BASE}/dark_knight_dialogue_detail.json', 'r', encoding='utf-8') as f:
    detail = json.load(f)

# BATMAN + WAYNE 합침 (순서 보존: detail의 line_num 기준)
batman_detail = [d for d in detail if d['character'] == 'BATMAN']
wayne_detail = [d for d in detail if d['character'] == 'WAYNE']

# line_num 기준 정렬 (각본 순서 유지)
combined = sorted(batman_detail + wayne_detail, key=lambda d: d['line_num'])
combined_lines = [d['line'] for d in combined]
combined_labels_true = [d['character'] for d in combined]  # ground truth

print(f"BATMAN: {len(batman_detail)} lines")
print(f"WAYNE:  {len(wayne_detail)} lines")
print(f"Total:  {len(combined)} lines")

# ══════════════════════════════════════
# 1. LLM 로드
# ══════════════════════════════════════
print("\n" + "=" * 60)
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
    MODEL_NAME, dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.eval()
print("모델 로드 완료")


def compute_window_features(text: str) -> dict:
    """LLM 퍼플렉시티 기반 특징 추출"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] < 3:
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    token_bits = -token_log_probs[0].float().cpu().numpy() / np.log(2)

    bits_per_char = float(token_bits.sum()) / max(len(text), 1)

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
# 2. 윈도우별 특징 추출
# ══════════════════════════════════════
WINDOW = 8   # 영어 대사가 일본어보다 길어서 윈도우 좀 줄임
STRIDE = 3

print(f"\n{len(combined_lines)}줄, 윈도우={WINDOW}, 스트라이드={STRIDE}")

windows_features = []
window_meta = []
window_ground_truth = []  # 각 윈도우의 다수 라벨

total_windows = (len(combined_lines) - WINDOW) // STRIDE + 1
print(f"총 {total_windows}개 윈도우 처리 중...")

for idx, i in enumerate(range(0, len(combined_lines) - WINDOW + 1, STRIDE)):
    chunk_lines = combined_lines[i:i + WINDOW]
    chunk_text = '\n'.join(chunk_lines)

    # LLM 특징
    llm_feat = compute_window_features(chunk_text)
    if llm_feat is None:
        continue

    # zlib 특징
    raw = chunk_text.encode('utf-8')
    zlib_ent = len(zlib.compress(raw, 9)) / max(len(raw), 1)

    # 문체 특징 (영어)
    avg_len = np.mean([len(l) for l in chunk_lines])
    q_rate = sum(1 for l in chunk_lines if '?' in l) / WINDOW
    ex_rate = sum(1 for l in chunk_lines if '!' in l) / WINDOW
    el_rate = sum(1 for l in chunk_lines if '...' in l or '—' in l or '- ' in l) / WINDOW

    feature = {
        **llm_feat,
        'zlib_entropy': zlib_ent,
        'avg_len': avg_len,
        'question_rate': q_rate,
        'exclaim_rate': ex_rate,
        'ellipsis_rate': el_rate,
    }
    windows_features.append(feature)

    # Ground truth: 윈도우 내 다수 라벨
    labels_in_window = combined_labels_true[i:i + WINDOW]
    majority = Counter(labels_in_window).most_common(1)[0][0]
    batman_pct = sum(1 for l in labels_in_window if l == 'BATMAN') / WINDOW

    pos = (i + WINDOW / 2) / len(combined_lines)
    window_meta.append({
        'pos': pos,
        'majority_label': majority,
        'batman_pct': batman_pct,
        'sample': chunk_lines[:3],
    })
    window_ground_truth.append(1 if majority == 'BATMAN' else 0)

    if (idx + 1) % 20 == 0:
        print(f"  {idx+1}/{total_windows} "
              f"(bpc={llm_feat['bits_per_char']:.3f}, label={majority})")

print(f"\n처리 완료: {len(windows_features)}개 윈도우")

# ══════════════════════════════════════
# 3. 클러스터링
# ══════════════════════════════════════
feature_keys = ['bits_per_char', 'bits_per_token', 'bits_std', 'max_surprise',
                'median_surprise', 'low_surprise_ratio', 'high_surprise_ratio',
                'zlib_entropy', 'avg_len', 'question_rate', 'exclaim_rate', 'ellipsis_rate']

X = np.array([[f[k] for k in feature_keys] for f in windows_features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("LLM 특징 기반 Bruce Wayne 클러스터링")
print("=" * 60)

gt = np.array(window_ground_truth)

k_range = range(2, 7)
silhouette_scores = []
ari_scores = []
best_labels = {}

for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    best_labels[k] = labels

    # k=2일 때 ARI (Adjusted Rand Index)로 ground truth와 비교
    if k == 2:
        ari = adjusted_rand_score(gt, labels)
        ari_scores.append(ari)
        print(f"  k={k}: silhouette={sil:.4f}, ARI vs ground truth={ari:.4f}")
    else:
        print(f"  k={k}: silhouette={sil:.4f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n최적 k = {best_k} (silhouette = {max(silhouette_scores):.4f})")

# ══════════════════════════════════════
# 4. k=2 상세 분석: BATMAN vs WAYNE 일치?
# ══════════════════════════════════════
print("\n" + "=" * 60)
print("k=2 클러스터 vs 각본 라벨 비교")
print("=" * 60)

labels_k2 = best_labels[2]

for c in range(2):
    mask = labels_k2 == c
    cluster_meta = [window_meta[i] for i in range(len(labels_k2)) if mask[i]]
    cluster_feat = [windows_features[i] for i in range(len(labels_k2)) if mask[i]]
    cluster_gt = [window_ground_truth[i] for i in range(len(labels_k2)) if mask[i]]

    batman_count = sum(cluster_gt)
    wayne_count = sum(mask) - batman_count

    bpc = [f['bits_per_char'] for f in cluster_feat]
    avg_l = [f['avg_len'] for f in cluster_feat]
    q_r = [f['question_rate'] for f in cluster_feat]
    ex_r = [f['exclaim_rate'] for f in cluster_feat]
    el_r = [f['ellipsis_rate'] for f in cluster_feat]
    zlib_e = [f['zlib_entropy'] for f in cluster_feat]

    print(f"\n  Cluster {c} ({sum(mask)}개, {sum(mask)/len(labels_k2)*100:.1f}%)")
    print(f"    BATMAN: {batman_count} ({batman_count/sum(mask)*100:.0f}%)")
    print(f"    WAYNE:  {wayne_count} ({wayne_count/sum(mask)*100:.0f}%)")
    print(f"    LLM bits/char: {np.mean(bpc):.3f} +/- {np.std(bpc):.3f}")
    print(f"    zlib entropy:  {np.mean(zlib_e):.4f}")
    print(f"    문장길이:      {np.mean(avg_l):.1f}")
    print(f"    의문문율:      {np.mean(q_r):.2f}")
    print(f"    감탄문율:      {np.mean(ex_r):.2f}")
    print(f"    여운율(..—):   {np.mean(el_r):.2f}")
    print(f"    샘플:")
    for m in cluster_meta[:5]:
        print(f"      [{m['majority_label']:>7}] " +
              ' / '.join(f'"{s[:40]}"' for s in m['sample']))

# Confusion matrix
cm = confusion_matrix(gt, labels_k2)
print(f"\n  Confusion Matrix (row=ground truth, col=cluster):")
print(f"          C0    C1")
print(f"  WAYNE  {cm[0][0]:>4}  {cm[0][1]:>4}")
print(f"  BATMAN {cm[1][0]:>4}  {cm[1][1]:>4}")

# 어느 매핑이 더 나은지
match_a = cm[0][0] + cm[1][1]
match_b = cm[0][1] + cm[1][0]
best_match = max(match_a, match_b)
accuracy = best_match / len(labels_k2)
print(f"\n  Best alignment accuracy: {accuracy:.1%} ({best_match}/{len(labels_k2)})")
print(f"  (random baseline: 50%)")

# ══════════════════════════════════════
# 5. k=3도 분석 (를르슈처럼 3 페르소나?)
# ══════════════════════════════════════
if 3 in best_labels:
    labels_k3 = best_labels[3]
    print(f"\n{'='*60}")
    print(f"k=3 — Bruce Wayne에도 3 페르소나가 있는가?")
    print(f"{'='*60}")

    for c in range(3):
        mask = labels_k3 == c
        cluster_meta = [window_meta[i] for i in range(len(labels_k3)) if mask[i]]
        cluster_feat = [windows_features[i] for i in range(len(labels_k3)) if mask[i]]
        cluster_gt = [window_ground_truth[i] for i in range(len(labels_k3)) if mask[i]]

        batman_count = sum(cluster_gt)
        wayne_count = sum(mask) - batman_count

        bpc = [f['bits_per_char'] for f in cluster_feat]
        avg_l = [f['avg_len'] for f in cluster_feat]
        q_r = [f['question_rate'] for f in cluster_feat]
        ex_r = [f['exclaim_rate'] for f in cluster_feat]

        print(f"\n  Cluster {c} ({sum(mask)}개, {sum(mask)/len(labels_k3)*100:.1f}%)")
        print(f"    BATMAN/WAYNE: {batman_count}/{wayne_count}")
        print(f"    LLM bpc: {np.mean(bpc):.3f}, 문장길이: {np.mean(avg_l):.1f}")
        print(f"    의문문율: {np.mean(q_r):.2f}, 감탄문율: {np.mean(ex_r):.2f}")
        for m in cluster_meta[:3]:
            print(f"      [{m['majority_label']:>7}] " +
                  ' / '.join(f'"{s[:35]}"' for s in m['sample']))


# ══════════════════════════════════════
# 6. 그래프
# ══════════════════════════════════════
COLORS = ['#1C1C1C', '#4169E1', '#FF4500']  # Batman-dark, Wayne-blue, Extra-orange

# 6-1. PCA 시각화 (k=2, ground truth 대비)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 왼쪽: 클러스터 라벨
labels_k2 = best_labels[2]
for c in range(2):
    mask = labels_k2 == c
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=COLORS[c], s=30, alpha=0.6, label=f'Cluster {c}')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax1.set_title('LLM Clustering (k=2)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.15)

# 오른쪽: Ground truth 라벨
for label, color, name in [(1, '#1C1C1C', 'BATMAN'), (0, '#4169E1', 'WAYNE')]:
    mask = gt == label
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, s=30, alpha=0.6, label=name)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax2.set_title('Script Labels (Ground Truth)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.15)

fig.suptitle(f'Bruce Wayne Persona — LLM Clustering vs Script Labels\n'
             f'Silhouette={silhouette_scores[0]:.3f}, Alignment={accuracy:.1%}',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}/bruce_persona_pca_k2.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: bruce_persona_pca_k2.png")

# 6-2. 타임라인: 클러스터 vs Ground truth
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

positions = [m['pos'] for m in window_meta]
bpc_values = [f['bits_per_char'] for f in windows_features]

# 상단: BPC 타임라인, 색=클러스터
for c in range(2):
    mask = labels_k2 == c
    c_pos = [positions[i] for i in range(len(labels_k2)) if mask[i]]
    c_bpc = [bpc_values[i] for i in range(len(labels_k2)) if mask[i]]
    ax1.scatter(c_pos, c_bpc, c=COLORS[c], s=20, alpha=0.6, label=f'Cluster {c}')
ax1.set_ylabel('LLM bits/char', fontsize=12, fontweight='bold')
ax1.set_title('Bruce Wayne — LLM Clustering Timeline', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.15)

# 하단: BPC 타임라인, 색=Ground truth
for label, color, name in [(1, '#1C1C1C', 'BATMAN'), (0, '#4169E1', 'WAYNE')]:
    mask = gt == label
    g_pos = [positions[i] for i in range(len(gt)) if mask[i]]
    g_bpc = [bpc_values[i] for i in range(len(gt)) if mask[i]]
    ax2.scatter(g_pos, g_bpc, c=color, s=20, alpha=0.6, label=name)
ax2.set_ylabel('LLM bits/char', fontsize=12, fontweight='bold')
ax2.set_xlabel('Script progression', fontsize=12, fontweight='bold')
ax2.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(f'{BASE}/bruce_persona_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bruce_persona_timeline.png")

# 6-3. Silhouette 비교
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(k_range), silhouette_scores, 'o-', color='#1C1C1C', linewidth=2.5,
        markersize=10)
for k, s in zip(k_range, silhouette_scores):
    ax.annotate(f'{s:.3f}', (k, s), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('k (number of personas)', fontsize=13, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
ax.set_title('Bruce Wayne — Optimal Persona Count\n'
             'How many personas does Bruce have?',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{BASE}/bruce_persona_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bruce_persona_silhouette.png")

# 결과 저장
results = {
    'silhouette_scores': {str(k): s for k, s in zip(k_range, silhouette_scores)},
    'best_k': best_k,
    'alignment_accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'windows_count': len(windows_features),
    'batman_windows': int(gt.sum()),
    'wayne_windows': int(len(gt) - gt.sum()),
}
with open(f'{BASE}/bruce_persona_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved: bruce_persona_results.json")
print("\n완료!")
