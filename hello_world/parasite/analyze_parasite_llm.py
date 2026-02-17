"""기생충 캐릭터별 LLM 퍼플렉시티 분석
— zlib 대신 실제 language model의 per-token log-probability로 entropy 측정

RTX 4070 Ti 12GB 기준, Qwen2.5-3B (fp16 ~6GB)
"""

import json
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
WINDOW = 15
STRIDE = 5
CHAR_COLORS = {
    '기우': '#E74C3C', '기택': '#C0392B', '충숙': '#E67E22', '기정': '#F39C12',
    '연교': '#3498DB', '동익': '#2980B9', '다혜': '#1ABC9C', '다송': '#16A085',
    '문광': '#8E44AD', '근세': '#9B59B6',
}

# ── 모델 로딩 ──
MODEL_NAME = "Qwen/Qwen2.5-3B"
print(f"모델 로딩: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print(f"모델 로딩 완료 (GPU: {torch.cuda.get_device_name(0)})\n")


# ── 퍼플렉시티 계산 ──
def compute_bits_per_char(text: str, max_length: int = 4096) -> dict:
    """텍스트의 LLM 기반 bits/char 계산"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    n_tokens = input_ids.shape[1]
    n_chars = len(text)

    if n_tokens < 2:
        return {'bits_per_token': 0, 'bits_per_char': 0, 'n_tokens': n_tokens}

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]          # (1, T-1, V)
        targets = input_ids[:, 1:]                   # (1, T-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # nats → bits
        token_bits = (-token_log_probs / math.log(2)).squeeze(0)   # (T-1,)
        avg_bits_per_token = token_bits.mean().item()
        total_bits = token_bits.sum().item()
        bits_per_char = total_bits / n_chars if n_chars > 0 else 0

    return {
        'bits_per_token': avg_bits_per_token,
        'bits_per_char': bits_per_char,
        'n_tokens': n_tokens,
        'n_chars': n_chars,
        'token_bits': token_bits.cpu().numpy().tolist()
    }


def sliding_window_bpc(lines, window=WINDOW, stride=STRIDE):
    """슬라이딩 윈도우 LLM bits/char"""
    positions = []
    bpc_values = []
    for i in range(0, len(lines) - window + 1, stride):
        chunk = '\n'.join(lines[i:i + window])
        r = compute_bits_per_char(chunk)
        pos = (i + window / 2) / len(lines)
        positions.append(pos)
        bpc_values.append(r['bits_per_char'])
    return positions, bpc_values


def find_anomalies(positions, values, n=2):
    """z-score 기반 이상치 탐지"""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    z_scores = np.abs(values - mean) / std
    top_idx = np.argsort(z_scores)[-n:]
    return [(positions[i], values[i], z_scores[i]) for i in top_idx]


def get_dialogue_at_position(lines, pos, window=WINDOW):
    """특정 위치의 대사 반환"""
    idx = int(pos * len(lines) - window / 2)
    idx = max(0, min(idx, len(lines) - window))
    return lines[idx:idx + window]


# ══════════════════════════════════════
# 분석 실행
# ══════════════════════════════════════
print("=" * 65)
print(f"캐릭터별 LLM 퍼플렉시티 분석 ({MODEL_NAME})")
print("=" * 65)

results = {}

for char in MAIN_CHARS:
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW:
        print(f"  {char}: 대사 부족 ({len(lines)}줄), 스킵")
        continue

    print(f"  {char} ({len(lines)}줄)... ", end="", flush=True)

    # 전체 bits/char
    full_text = '\n'.join(lines)
    full_r = compute_bits_per_char(full_text)

    # 슬라이딩 윈도우
    positions, bpc_vals = sliding_window_bpc(lines)

    results[char] = {
        'total_lines': len(lines),
        'bits_per_token': round(full_r['bits_per_token'], 4),
        'bits_per_char': round(full_r['bits_per_char'], 4),
        'n_tokens': full_r['n_tokens'],
        'bpc_mean': round(float(np.mean(bpc_vals)), 4),
        'bpc_std': round(float(np.std(bpc_vals)), 4),
        'timeline': {
            'positions': [round(p, 4) for p in positions],
            'bpc': [round(p, 4) for p in bpc_vals]
        }
    }

    print(f"bits/char={full_r['bits_per_char']:.3f}, "
          f"mean={np.mean(bpc_vals):.3f}, σ={np.std(bpc_vals):.4f}")

# ── JSON 저장 ──
with open('D:/game-portfolio-main/parasite_llm_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장: parasite_llm_analysis.json")


# ══════════════════════════════════════
# 그래프 1: LLM 퍼플렉시티 타임라인
# ══════════════════════════════════════
fig, axes = plt.subplots(len(MAIN_CHARS), 1, figsize=(14, 3 * len(MAIN_CHARS)),
                         sharex=True)
fig.suptitle(f'기생충 — 캐릭터별 LLM 퍼플렉시티 변화 ({MODEL_NAME})\n'
             '↑ 높을수록 LLM이 예측하지 못한 대사', fontsize=14, fontweight='bold', y=0.995)

anomaly_report = {}

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    if char not in results:
        ax.text(0.5, 0.5, f'{char}: 대사 부족', transform=ax.transAxes, ha='center')
        ax.set_ylabel(char, fontsize=11, fontweight='bold', color=CHAR_COLORS.get(char, 'gray'))
        continue

    r = results[char]
    pos = r['timeline']['positions']
    bpc = r['timeline']['bpc']
    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(pos, bpc, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, bpc, alpha=0.15, color=color)

    mean_b = np.mean(bpc)
    ax.axhline(y=mean_b, color=color, linestyle='--', alpha=0.4, linewidth=1)

    # 이상치
    anomalies = find_anomalies(pos, bpc, n=2)
    for a_pos, a_val, a_z in anomalies:
        ax.scatter([a_pos], [a_val], color='red', s=80, zorder=10,
                   edgecolors='white', linewidth=1.5)
        ax.annotate(f'z={a_z:.1f}', (a_pos, a_val),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold')

    # 이상치 대사 저장
    lines = dialogue.get(char, [])
    char_anomalies = []
    for a_pos, a_val, a_z in anomalies:
        sample = get_dialogue_at_position(lines, a_pos)
        char_anomalies.append({
            'position': round(a_pos, 2),
            'bpc': round(a_val, 4),
            'z_score': round(a_z, 2),
            'sample': sample[:5]
        })
    anomaly_report[char] = char_anomalies

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color, rotation=0, labelpad=40)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('영화 진행도 (0 = 시작, 1 = 끝) →', fontsize=11)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_llm_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_llm_timeline.png")


# ══════════════════════════════════════
# 그래프 2: zlib vs LLM 비교
# ══════════════════════════════════════
with open('D:/game-portfolio-main/parasite_analysis.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

chars_both = [c for c in MAIN_CHARS if c in results and c in old_data['entropy']]

# ── Left: σ 비교 산점도 ──
ax = axes[0]
zlib_std = [old_data['entropy'][c]['entropy_std'] for c in chars_both]
llm_std = [results[c]['bpc_std'] for c in chars_both]

for i, c in enumerate(chars_both):
    color = CHAR_COLORS.get(c, 'gray')
    ax.scatter(zlib_std[i], llm_std[i], c=color, s=180,
               edgecolors='white', linewidth=1.5, zorder=5)
    ax.annotate(c, (zlib_std[i], llm_std[i]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=11, fontweight='bold', color=color)

# 상관계수
corr = np.corrcoef(zlib_std, llm_std)[0, 1]
ax.set_xlabel('zlib 엔트로피 σ (표면 패턴)', fontsize=12, fontweight='bold')
ax.set_ylabel('LLM 퍼플렉시티 σ (심층 이해)', fontsize=12, fontweight='bold')
ax.set_title(f'표면 vs 심층 변동성 비교\nr = {corr:.3f}', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)

# ── Right: bits/char 랭킹 ──
ax = axes[1]
sorted_chars = sorted(chars_both, key=lambda c: results[c]['bits_per_char'])
bpc = [results[c]['bits_per_char'] for c in sorted_chars]
bar_colors = [CHAR_COLORS.get(c, 'gray') for c in sorted_chars]

bars = ax.barh(range(len(sorted_chars)), bpc, color=bar_colors, alpha=0.8,
               edgecolor='white', linewidth=1.5)
ax.set_yticks(range(len(sorted_chars)))
ax.set_yticklabels(sorted_chars, fontsize=11, fontweight='bold')
ax.set_xlabel('bits/char (LLM)', fontsize=12, fontweight='bold')
ax.set_title('캐릭터별 LLM bits/char 순위\n← 예측 가능 | 예측 불가능 →',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='x')

for i, b in enumerate(bars):
    ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2,
            f'{bpc[i]:.3f}', va='center', fontsize=10, fontweight='bold')

plt.suptitle(f'기생충 — zlib vs LLM 비교 ({MODEL_NAME})',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('D:/game-portfolio-main/parasite_llm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: parasite_llm_comparison.png")


# ══════════════════════════════════════
# 콘솔 리포트
# ══════════════════════════════════════
print("\n" + "=" * 70)
print(f"캐릭터별 LLM 퍼플렉시티 ({MODEL_NAME})")
print("=" * 70)
print(f"{'캐릭터':>6}  {'대사':>5}  {'tokens':>6}  {'bits/tok':>8}  {'bits/chr':>8}  {'σ':>8}")
print("-" * 70)
for char in MAIN_CHARS:
    if char not in results:
        continue
    r = results[char]
    print(f"{char:>6}  {r['total_lines']:>5}  {r['n_tokens']:>6}  "
          f"{r['bits_per_token']:>8.4f}  {r['bits_per_char']:>8.4f}  {r['bpc_std']:>8.4f}")

print("\n" + "=" * 70)
print("이상치 대사 (LLM이 가장 예측 못한 순간)")
print("=" * 70)
for char in MAIN_CHARS:
    if char not in anomaly_report:
        continue
    print(f"\n【{char}】")
    for a in anomaly_report[char]:
        pct = int(a['position'] * 100)
        print(f"  @ 영화 {pct}% | bits/char={a['bpc']:.3f} | z={a['z_score']}")
        for line in a['sample']:
            print(f"     \"{line}\"")

# 이상치 JSON 저장
with open('D:/game-portfolio-main/parasite_llm_anomalies.json', 'w', encoding='utf-8') as f:
    json.dump(anomaly_report, f, ensure_ascii=False, indent=2)
print("\n결과 저장: parasite_llm_anomalies.json")
