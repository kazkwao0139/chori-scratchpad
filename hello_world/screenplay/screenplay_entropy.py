"""
Screenplay Entropy Analyzer
아카데미 각본상 수상작 vs 비수상작의 캐릭터별 entropy 분산 비교.

가설: 수상작의 캐릭터간 entropy 분산 > 비수상작
근거: hello_world 발견 10 — 좋은 서사 = 저엔트로피 backbone + 고엔트로피 perturbation
"""

import re
import math
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ─── Bigram Entropy ─────────────────────────────────

def bigram_entropy(text: str) -> float:
    """텍스트의 bigram conditional entropy (bits/char)."""
    text = text.lower()
    if len(text) < 2:
        return 0.0

    bigrams = Counter()
    unigrams = Counter()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        bigrams[(a, b)] += 1
        unigrams[a] += 1

    H = 0.0
    total = sum(bigrams.values())
    for (a, b), count in bigrams.items():
        p_ab = count / total
        p_b_given_a = count / unigrams[a]
        H -= p_ab * math.log2(p_b_given_a)
    return H


# ─── IMSDB Fetcher & Parser ─────────────────────────

def fetch_script(url: str) -> str:
    """IMSDB에서 대본 HTML을 가져온다."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (educational research)'
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode('utf-8', errors='replace')


def extract_text(html: str) -> str:
    """HTML에서 대본 텍스트만 추출."""
    m = re.search(r'<td class="scrtext">(.*?)</td>', html, re.DOTALL)
    if not m:
        m = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not m:
        return ""
    text = m.group(1)
    text = re.sub(r'<b>', '\n<b>', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text


def parse_dialogue(text: str) -> Dict[str, str]:
    """대본에서 캐릭터별 대사를 분리한다.

    규칙: 전부 대문자인 줄 = 캐릭터명, 그 아래 들여쓰기된 줄 = 대사.
    """
    lines = text.split('\n')
    characters = defaultdict(list)
    current_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_char = None
            continue

        # 캐릭터명: 전부 대문자, 2~30자, 괄호 제거
        clean = re.sub(r'\(.*?\)', '', stripped).strip()
        if (clean.isupper() and 2 <= len(clean) <= 30
                and not clean.startswith('INT') and not clean.startswith('EXT')
                and not clean.startswith('CUT') and not clean.startswith('FADE')
                and not clean.startswith('CLOSE') and not clean.startswith('ANGLE')
                and not clean.startswith('THE ')
                and re.match(r'^[A-Z][A-Z\s\.\'-]+$', clean)):
            current_char = clean
            continue

        if current_char and len(stripped) > 1:
            # 지문(action line)이 아닌 대사만
            if not stripped.isupper():
                characters[current_char].append(stripped)

    return {char: ' '.join(lines) for char, lines in characters.items()}


# ─── Analysis ────────────────────────────────────────

@dataclass
class CharEntropy:
    name: str
    entropy: float
    char_count: int


@dataclass
class ScriptAnalysis:
    title: str
    category: str  # 'winner' or 'non-winner'
    characters: List[CharEntropy]
    entropy_mean: float
    entropy_variance: float
    entropy_std: float


def analyze_script(title: str, url: str, category: str,
                   min_chars: int = 500) -> ScriptAnalysis:
    """한 대본의 캐릭터별 entropy 분석."""
    print(f"\n  [{category.upper()}] {title}")
    print(f"  fetching... ", end='', flush=True)

    html = fetch_script(url)
    text = extract_text(html)
    dialogue = parse_dialogue(text)

    print(f"{len(dialogue)} characters found")

    results = []
    for char, lines in dialogue.items():
        if len(lines) < min_chars:
            continue
        H = bigram_entropy(lines)
        results.append(CharEntropy(name=char, entropy=H, char_count=len(lines)))

    results.sort(key=lambda x: x.entropy, reverse=True)

    if not results:
        return ScriptAnalysis(title, category, [], 0, 0, 0)

    entropies = [r.entropy for r in results]
    mean = sum(entropies) / len(entropies)
    var = sum((e - mean) ** 2 for e in entropies) / len(entropies)

    # 상위 캐릭터 출력
    for i, r in enumerate(results[:8]):
        bar = '#' * int(r.entropy * 10)
        print(f"    {i+1:2d}  {r.name:<20s}  {r.entropy:.4f} b/c  "
              f"{r.char_count:>6,d} chars  {bar}")

    print(f"  ──────────────────────────────────")
    print(f"  mean: {mean:.4f}  var: {var:.6f}  std: {var**0.5:.4f}  "
          f"({len(results)} characters)")

    return ScriptAnalysis(
        title=title,
        category=category,
        characters=results,
        entropy_mean=mean,
        entropy_variance=var,
        entropy_std=var ** 0.5,
    )


# ─── Main ────────────────────────────────────────────

# 같은 연도 수상작 vs 후보작(비수상) — 공정한 비교
SCRIPTS_BY_YEAR = {
    2004: {
        'winner': ("Eternal Sunshine", "https://imsdb.com/scripts/Eternal-Sunshine-of-the-Spotless-Mind.html"),
        'nominees': [
            ("The Aviator", "https://imsdb.com/scripts/Aviator,-The.html"),
            ("Hotel Rwanda", "https://imsdb.com/scripts/Hotel-Rwanda.html"),
            ("The Incredibles", "https://imsdb.com/scripts/Incredibles,-The.html"),
        ],
    },
    2007: {
        'winner': ("Juno", "https://imsdb.com/scripts/Juno.html"),
        'nominees': [
            ("Michael Clayton", "https://imsdb.com/scripts/Michael-Clayton.html"),
            ("Ratatouille", "https://imsdb.com/scripts/Ratatouille.html"),
            ("Lars and the Real Girl", "https://imsdb.com/scripts/Lars-and-the-Real-Girl.html"),
        ],
    },
    2009: {
        'winner': ("The Hurt Locker", "https://imsdb.com/scripts/Hurt-Locker,-The.html"),
        'nominees': [
            ("Inglourious Basterds", "https://imsdb.com/scripts/Inglourious-Basterds.html"),
            ("Up", "https://imsdb.com/scripts/Up.html"),
        ],
    },
    2010: {
        'winner': ("The King's Speech", "https://imsdb.com/scripts/King's-Speech,-The.html"),
        'nominees': [
            ("Black Swan", "https://imsdb.com/scripts/Black-Swan.html"),
            ("Inception", "https://imsdb.com/scripts/Inception.html"),
            ("The Fighter", "https://imsdb.com/scripts/Fighter,-The.html"),
            ("The Kids Are All Right", "https://imsdb.com/scripts/Kids-Are-All-Right,-The.html"),
        ],
    },
}


def main():
    print("=" * 60)
    print("  SCREENPLAY ENTROPY ANALYZER")
    print("  같은 연도 수상작 vs 후보작 — 캐릭터 entropy 분산 비교")
    print("=" * 60)

    winner_results = []
    nominee_results = []

    for year, data in SCRIPTS_BY_YEAR.items():
        print(f"\n{'─' * 60}")
        print(f"  {year} Academy Awards")
        print(f"{'─' * 60}")

        # 수상작
        title, url = data['winner']
        try:
            result = analyze_script(title, url, 'winner')
            if result.characters:
                winner_results.append((year, result))
        except Exception as e:
            print(f"  ERROR: {title} — {e}")

        # 후보작
        for title, url in data['nominees']:
            try:
                result = analyze_script(title, url, 'nominee')
                if result.characters:
                    nominee_results.append((year, result))
            except Exception as e:
                print(f"  ERROR: {title} — {e}")

    # ─── 연도별 비교 ─────────────────────────────
    print("\n\n" + "=" * 60)
    print("  YEAR-BY-YEAR COMPARISON")
    print("=" * 60)

    winner_wins = 0
    total_years = 0

    for year in sorted(SCRIPTS_BY_YEAR.keys()):
        w = [r for y, r in winner_results if y == year]
        n = [r for y, r in nominee_results if y == year]
        if not w or not n:
            continue

        total_years += 1
        w_var = w[0].entropy_variance
        n_var = sum(r.entropy_variance for r in n) / len(n)

        won = w_var > n_var
        if won:
            winner_wins += 1

        mark = "✓" if won else "✗"
        print(f"\n  {year}  {mark}")
        print(f"    WINNER:   {w[0].title:<25s}  var={w_var:.6f}")
        for r in n:
            print(f"    nominee:  {r.title:<25s}  var={r.entropy_variance:.6f}")
        print(f"    winner/avg_nominee = {w_var / (n_var + 1e-10):.2f}x")

    # ─── 최종 결과 ─────────────────────────────
    print("\n\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    all_w_var = [r.entropy_variance for _, r in winner_results]
    all_n_var = [r.entropy_variance for _, r in nominee_results]

    if all_w_var and all_n_var:
        mean_w = sum(all_w_var) / len(all_w_var)
        mean_n = sum(all_n_var) / len(all_n_var)

        print(f"\n  수상작 평균 분산:    {mean_w:.6f}  (n={len(all_w_var)})")
        print(f"  후보작 평균 분산:    {mean_n:.6f}  (n={len(all_n_var)})")
        print(f"  비율:               {mean_w / (mean_n + 1e-10):.2f}x")
        print(f"  연도별 승률:        {winner_wins}/{total_years}")

        print(f"\n  {'─' * 50}")
        if mean_w > mean_n:
            print(f"  결론: 수상작의 캐릭터간 entropy 분산이 {mean_w/mean_n:.2f}배 높다.")
            print(f"  연도별 {winner_wins}/{total_years} 일관성.")
            print(f"  → 발견 10 지지: 좋은 서사는 entropy spike가 크다.")
        else:
            print(f"  결론: 후보작의 분산이 더 높거나 같다.")
            print(f"  → 발견 10 기각 또는 추가 검증 필요.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
