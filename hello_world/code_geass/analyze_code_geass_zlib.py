"""Code Geass — zlib 엔트로피 + cosine similarity 분석
Netflix 일본어 자막 기반, 기생충과 동일한 파이프라인.

추가: ゼロ/ルルーシュ 분리 분석 (페르소나 비교)
"""

import json
import zlib
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 한글/일본어 폰트 ──
for name in ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Malgun Gothic']:
    if [f for f in fm.fontManager.ttflist if name in f.name]:
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──
with open('D:/game-portfolio-main/code_geass_dialogue_ja.json', 'r', encoding='utf-8') as f:
    dialogue = json.load(f)

with open('D:/game-portfolio-main/code_geass_dialogue_ja_detail.json', 'r', encoding='utf-8') as f:
    detail = json.load(f)

# ══════════════════════════════════════
# 메인 캐릭터 선정
# ══════════════════════════════════════
MAIN_CHARS = ['ルルーシュ', 'スザク', 'C.C.', 'カレン', 'シャーリー',
              'コーネリア', 'ユーフェミア', 'ナナリー', 'ロロ', 'シュナイゼル']

CHAR_COLORS = {
    'ルルーシュ': '#8B0000',  # dark red (주인공)
    'スザク':    '#1E90FF',  # dodger blue (라이벌)
    'C.C.':     '#32CD32',  # lime green
    'カレン':    '#FF4500',  # orange red
    'シャーリー': '#FF69B4',  # hot pink
    'コーネリア': '#4B0082',  # indigo
    'ユーフェミア':'#FF1493', # deep pink
    'ナナリー':  '#DDA0DD',  # plum
    'ロロ':     '#808000',  # olive
    'シュナイゼル':'#2F4F4F', # dark slate gray
}

# ── 분석 함수들 ──
WINDOW = 15
STRIDE = 5

def text_entropy(text: str) -> float:
    """zlib 압축률 = 표면 엔트로피"""
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def char_frequency(text: str) -> dict:
    """문자 빈도 벡터"""
    freq = {}
    for ch in text:
        if ch.strip():
            freq[ch] = freq.get(ch, 0) + 1
    return freq


def cosine_sim(v1: dict, v2: dict) -> float:
    """두 빈도 벡터의 코사인 유사도"""
    all_keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in all_keys)
    m1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    m2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


# ══════════════════════════════════════
# 1. 캐릭터별 기본 통계
# ══════════════════════════════════════
print("=" * 70)
print("Code Geass — zlib 엔트로피 분석")
print("=" * 70)

results = {}

for char in MAIN_CHARS:
    lines = dialogue.get(char, [])
    if len(lines) < WINDOW:
        print(f"  {char}: 대사 부족 ({len(lines)}줄)")
        continue

    full_text = '\n'.join(lines)
    total_entropy = text_entropy(full_text)

    # 슬라이딩 윈도우 엔트로피
    entropies = []
    positions = []
    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        e = text_entropy(chunk)
        pos = (i + WINDOW / 2) / len(lines)
        entropies.append(e)
        positions.append(pos)

    # 슬라이딩 윈도우 코사인 (이전 윈도우와 비교)
    cosines = []
    cos_positions = []
    prev_vec = None
    for i in range(0, len(lines) - WINDOW + 1, STRIDE):
        chunk = '\n'.join(lines[i:i + WINDOW])
        vec = char_frequency(chunk)
        if prev_vec is not None:
            sim = cosine_sim(prev_vec, vec)
            cosines.append(sim)
            cos_positions.append((i + WINDOW / 2) / len(lines))
        prev_vec = vec

    ent_arr = np.array(entropies)
    cos_arr = np.array(cosines) if cosines else np.array([0])

    results[char] = {
        'total_lines': len(lines),
        'total_entropy': round(total_entropy, 6),
        'entropy_mean': round(float(np.mean(ent_arr)), 6),
        'entropy_std': round(float(np.std(ent_arr)), 6),
        'cosine_mean': round(float(np.mean(cos_arr)), 6),
        'cosine_std': round(float(np.std(cos_arr)), 6),
        'timeline': {
            'positions': [round(p, 4) for p in positions],
            'entropy': [round(float(e), 6) for e in entropies],
        },
        'cosine_timeline': {
            'positions': [round(p, 4) for p in cos_positions],
            'cosine': [round(float(c), 6) for c in cosines],
        },
    }

    print(f"\n  {char} ({len(lines)}줄)")
    print(f"    전체 zlib: {total_entropy:.4f}")
    print(f"    윈도우 평균: {np.mean(ent_arr):.4f} ± {np.std(ent_arr):.4f}")
    print(f"    코사인 평균: {np.mean(cos_arr):.4f} ± {np.std(cos_arr):.4f}")


# ══════════════════════════════════════
# 2. ルルーシュ 페르소나 분석
#    제로 모드 vs 일상 모드를 에피소드 맥락으로 구분
# ══════════════════════════════════════
print("\n" + "=" * 70)
print("ルルーシュ 페르소나 분석: 전반 vs 후반 엔트로피 변동")
print("=" * 70)

lelouch_detail = detail.get('ルルーシュ', [])
if lelouch_detail:
    # 에피소드별 엔트로피 계산
    ep_entropies = {}
    for item in lelouch_detail:
        key = f"{item['season']}_E{item['episode']:02d}"
        if key not in ep_entropies:
            ep_entropies[key] = []
        ep_entropies[key].append(item['text'])

    ep_stats = []
    for ep_key in sorted(ep_entropies.keys()):
        lines = ep_entropies[ep_key]
        if len(lines) >= 5:
            text = '\n'.join(lines)
            e = text_entropy(text)
            ep_stats.append((ep_key, len(lines), e))
            print(f"  {ep_key}: {len(lines):>4}줄, zlib={e:.4f}")

    # 전반(S1) vs 후반(R2) 비교
    s1_ent = [e for k, n, e in ep_stats if k.startswith('S1')]
    r2_ent = [e for k, n, e in ep_stats if k.startswith('R2')]
    if s1_ent and r2_ent:
        print(f"\n  S1 평균: {np.mean(s1_ent):.4f} ± {np.std(s1_ent):.4f}")
        print(f"  R2 평균: {np.mean(r2_ent):.4f} ± {np.std(r2_ent):.4f}")
        print(f"  차이: {abs(np.mean(s1_ent) - np.mean(r2_ent)):.4f}")


# ══════════════════════════════════════
# 3. 그래프: 엔트로피 타임라인 (전 캐릭터)
# ══════════════════════════════════════
fig, axes = plt.subplots(len(MAIN_CHARS), 1,
                         figsize=(18, 3 * len(MAIN_CHARS)), sharex=True)
fig.suptitle('Code Geass — zlib entropy timeline\n'
             'Character voice consistency across 50 episodes',
             fontsize=16, fontweight='bold', y=0.995)

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    if char not in results:
        ax.text(0.5, 0.5, f'{char}: not enough data', transform=ax.transAxes, ha='center')
        continue

    r = results[char]
    pos = r['timeline']['positions']
    ent = r['timeline']['entropy']
    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(pos, ent, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, ent, alpha=0.15, color=color)

    # 평균선
    mean_e = r['entropy_mean']
    ax.axhline(y=mean_e, color=color, linestyle='--', linewidth=1, alpha=0.5)

    # 시즌 경계
    # ルルーシュ의 경우 S1 대사가 약 절반 지점
    total = r['total_lines']
    s1_count = sum(1 for item in detail.get(char, []) if item['season'] == 'S1')
    if s1_count > 0 and s1_count < total:
        boundary = s1_count / total
        ax.axvline(x=boundary, color='red', linestyle='-', linewidth=2, alpha=0.4)
        ax.text(boundary, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else max(ent),
                ' S1|R2', fontsize=9, color='red', alpha=0.6, va='top')

    # 통계 표시
    ax.text(0.98, 0.95,
            f'mean={r["entropy_mean"]:.4f}\nσ={r["entropy_std"]:.4f}\nlines={total}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color,
                  rotation=0, labelpad=50)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.15)

axes[-1].set_xlabel('Story progression (0=start, 1=end)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_zlib_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: code_geass_zlib_timeline.png")


# ══════════════════════════════════════
# 4. 그래프: 코사인 유사도 타임라인
# ══════════════════════════════════════
fig, axes = plt.subplots(len(MAIN_CHARS), 1,
                         figsize=(18, 3 * len(MAIN_CHARS)), sharex=True)
fig.suptitle('Code Geass — Cosine similarity timeline\n'
             'How much does the character vocabulary shift?',
             fontsize=16, fontweight='bold', y=0.995)

for idx, char in enumerate(MAIN_CHARS):
    ax = axes[idx]
    if char not in results or not results[char]['cosine_timeline']['positions']:
        ax.text(0.5, 0.5, f'{char}: not enough data', transform=ax.transAxes, ha='center')
        continue

    r = results[char]
    pos = r['cosine_timeline']['positions']
    cos = r['cosine_timeline']['cosine']
    color = CHAR_COLORS.get(char, 'gray')

    ax.plot(pos, cos, color=color, linewidth=2, alpha=0.8)
    ax.fill_between(pos, cos, alpha=0.15, color=color)

    mean_c = r['cosine_mean']
    ax.axhline(y=mean_c, color=color, linestyle='--', linewidth=1, alpha=0.5)

    # 시즌 경계
    total = r['total_lines']
    s1_count = sum(1 for item in detail.get(char, []) if item['season'] == 'S1')
    if s1_count > 0 and s1_count < total:
        boundary = s1_count / total
        ax.axvline(x=boundary, color='red', linestyle='-', linewidth=2, alpha=0.4)

    ax.text(0.98, 0.05,
            f'mean={r["cosine_mean"]:.4f}\nσ={r["cosine_std"]:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel(char, fontsize=11, fontweight='bold', color=color,
                  rotation=0, labelpad=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.15)

axes[-1].set_xlabel('Story progression (0=start, 1=end)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_cosine_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: code_geass_cosine_timeline.png")


# ══════════════════════════════════════
# 5. 비교 차트: σ 스캐터
# ══════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

for char in MAIN_CHARS:
    if char not in results:
        continue
    r = results[char]
    color = CHAR_COLORS.get(char, 'gray')
    ax.scatter(r['entropy_std'], r['cosine_std'],
               s=r['total_lines'] / 3, color=color, alpha=0.7,
               edgecolors='white', linewidth=2, zorder=10)
    ax.annotate(char, (r['entropy_std'], r['cosine_std']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=11, fontweight='bold', color=color)

ax.set_xlabel('Entropy σ (voice instability)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cosine σ (vocabulary shift)', fontsize=13, fontweight='bold')
ax.set_title('Code Geass — Character Voice Fingerprint\n'
             'Size = dialogue count, Position = instability',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('D:/game-portfolio-main/code_geass_voice_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: code_geass_voice_scatter.png")


# ══════════════════════════════════════
# 6. 결과 저장
# ══════════════════════════════════════
with open('D:/game-portfolio-main/code_geass_zlib_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved: code_geass_zlib_results.json")

# 요약 테이블
print("\n" + "=" * 70)
print(f"{'Character':>14} {'Lines':>6} {'Entropy':>8} {'σ_ent':>8} {'Cosine':>8} {'σ_cos':>8}")
print("-" * 70)
for char in MAIN_CHARS:
    if char not in results:
        continue
    r = results[char]
    print(f"{char:>14} {r['total_lines']:>6} {r['entropy_mean']:>8.4f} {r['entropy_std']:>8.4f} "
          f"{r['cosine_mean']:>8.4f} {r['cosine_std']:>8.4f}")
