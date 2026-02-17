"""
기생충 캐릭터 일관성 분석
- 엔트로피: 말투/어휘 패턴의 일관성 (표면)
- 코사인 유사도: 의미/태도/성격의 일관성 (의미)

잘 쓴 캐릭터 = 엔트로피 분산 낮음 (말투 일관) + 코사인 분산 높음 (내용 변화 = 캐릭터 아크)
"""

import json
import zlib
import math
import numpy as np
from collections import defaultdict

# ── 설정 ──
DIALOGUE_PATH = "D:/game-portfolio-main/parasite_dialogue.json"
MIN_LINES = 20          # 최소 대사 수 (너무 적으면 통계적 의미 없음)
N_CHUNKS = 4            # 캐릭터별 대사를 몇 청크로 나눌지
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_dialogue(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ══════════════════════════════════════════
# 1. 엔트로피 분석 (compression-based)
# ══════════════════════════════════════════

def text_entropy(text: str) -> float:
    """zlib 압축률 기반 엔트로피 추정. 높을수록 예측 불가능."""
    raw = text.encode('utf-8')
    compressed = zlib.compress(raw, level=9)
    if len(raw) == 0:
        return 0.0
    return len(compressed) / len(raw)


def entropy_analysis(dialogue: dict, n_chunks: int = N_CHUNKS) -> dict:
    """캐릭터별 청크 엔트로피 분산 계산"""
    results = {}

    for char, lines in dialogue.items():
        if len(lines) < MIN_LINES:
            continue

        # 대사를 n_chunks로 등분
        chunk_size = len(lines) // n_chunks
        if chunk_size == 0:
            continue

        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else len(lines)
            chunk_text = '\n'.join(lines[start:end])
            chunks.append(chunk_text)

        # 각 청크의 엔트로피
        entropies = [text_entropy(c) for c in chunks]
        mean_e = np.mean(entropies)
        var_e = np.var(entropies)
        std_e = np.std(entropies)

        results[char] = {
            'mean_entropy': round(float(mean_e), 4),
            'entropy_var': round(float(var_e), 6),
            'entropy_std': round(float(std_e), 4),
            'chunk_entropies': [round(e, 4) for e in entropies],
            'total_lines': len(lines),
        }

    return results


# ══════════════════════════════════════════
# 2. 코사인 유사도 분석 (embedding-based)
# ══════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_analysis(dialogue: dict, n_chunks: int = N_CHUNKS) -> dict:
    """캐릭터별 청크 간 코사인 유사도 계산"""
    from sentence_transformers import SentenceTransformer

    print(f"모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    results = {}

    for char, lines in dialogue.items():
        if len(lines) < MIN_LINES:
            continue

        chunk_size = len(lines) // n_chunks
        if chunk_size == 0:
            continue

        # 청크별 텍스트 생성
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else len(lines)
            chunk_text = '\n'.join(lines[start:end])
            chunks.append(chunk_text)

        # 임베딩
        embeddings = model.encode(chunks)

        # 모든 청크 쌍의 코사인 유사도
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        mean_sim = np.mean(similarities)
        var_sim = np.var(similarities)
        std_sim = np.std(similarities)

        results[char] = {
            'mean_cosine': round(float(mean_sim), 4),
            'cosine_var': round(float(var_sim), 6),
            'cosine_std': round(float(std_sim), 4),
            'pairwise_sims': [round(s, 4) for s in similarities],
            'total_lines': len(lines),
        }

    return results


# ══════════════════════════════════════════
# 3. 종합 분석
# ══════════════════════════════════════════

def combined_report(entropy_res: dict, cosine_res: dict):
    """엔트로피 + 코사인 종합 리포트"""
    print("\n" + "=" * 70)
    print("기생충 — 캐릭터 일관성 분석")
    print("=" * 70)

    # 헤더
    print(f"\n{'캐릭터':<8} {'대사':>4} │ {'엔트로피μ':>8} {'엔트로피σ':>8} │ {'코사인μ':>7} {'코사인σ':>7} │ 판정")
    print("─" * 70)

    chars = sorted(
        set(entropy_res.keys()) & set(cosine_res.keys()),
        key=lambda c: -entropy_res[c]['total_lines']
    )

    for char in chars:
        e = entropy_res[char]
        c = cosine_res[char]

        # 판정 로직
        e_consistent = e['entropy_std'] < 0.015  # 엔트로피 분산 낮음 = 말투 일관
        c_varying = c['cosine_std'] > 0.02       # 코사인 분산 높음 = 내용 변화

        if e_consistent and c_varying:
            verdict = "★ 아크 있는 일관된 캐릭터"
        elif e_consistent and not c_varying:
            verdict = "● 안정적 (변화 적음)"
        elif not e_consistent and c_varying:
            verdict = "△ 불안정 (말투+내용 둘다 변동)"
        else:
            verdict = "○ 평이"

        print(f"{char:<8} {e['total_lines']:>4} │ "
              f"{e['mean_entropy']:>8.4f} {e['entropy_std']:>8.4f} │ "
              f"{c['mean_cosine']:>7.4f} {c['cosine_std']:>7.4f} │ {verdict}")

    # 해석 가이드
    print("\n─── 해석 가이드 ───")
    print("엔트로피σ 낮음: 말투/어휘 패턴이 일관됨 (캐릭터 voice 유지)")
    print("코사인σ  높음: 대사 내용이 장면마다 변함 (캐릭터 arc 존재)")
    print("★ = 말투는 유지하면서 내용이 변화 → 잘 쓰인 캐릭터의 신호")


if __name__ == '__main__':
    dialogue = load_dialogue(DIALOGUE_PATH)

    print(">>> 1/2: 엔트로피 분석 (로컬)")
    entropy_res = entropy_analysis(dialogue)

    print(">>> 2/2: 코사인 유사도 분석 (로컬 임베딩)")
    cosine_res = cosine_analysis(dialogue)

    combined_report(entropy_res, cosine_res)

    # 결과 저장
    out = {
        'entropy': entropy_res,
        'cosine': cosine_res,
    }
    with open('D:/game-portfolio-main/parasite_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: parasite_analysis.json")
