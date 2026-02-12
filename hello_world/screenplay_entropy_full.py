"""
Screenplay Entropy Analyzer — Full Historical Analysis
아카데미 각본상 수상작 vs 후보작의 캐릭터별 entropy 분산 비교.
가능한 한 많은 연도를 IMSDB에서 가져와 분석한다.

발견 10: 좋은 서사 = 저엔트로피 backbone + 고엔트로피 perturbation
발견 11 검증: 수상작의 캐릭터간 entropy 분산 > 후보작
"""

import re
import math
import sys
import time
import json
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ─── Bigram Entropy ─────────────────────────────────

def bigram_entropy(text: str) -> float:
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


# ─── IMSDB ──────────────────────────────────────────

def fetch_script(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (educational research)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def extract_text(html: str) -> str:
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
    lines = text.split('\n')
    characters = defaultdict(list)
    current_char = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_char = None
            continue
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
            if not stripped.isupper():
                characters[current_char].append(stripped)
    return {char: ' '.join(lines) for char, lines in characters.items()}


def analyze_one(title: str, url: str, min_chars: int = 500) -> Optional[dict]:
    html = fetch_script(url)
    if not html:
        return None
    text = extract_text(html)
    if not text or len(text) < 500:
        return None
    dialogue = parse_dialogue(text)
    results = []
    for char, lines in dialogue.items():
        if len(lines) < min_chars:
            continue
        H = bigram_entropy(lines)
        results.append({'name': char, 'entropy': H, 'chars': len(lines)})
    if len(results) < 3:  # 캐릭터 3명 미만이면 분산 의미없음
        return None
    entropies = [r['entropy'] for r in results]
    mean = sum(entropies) / len(entropies)
    var = sum((e - mean) ** 2 for e in entropies) / len(entropies)
    return {
        'title': title,
        'n_characters': len(results),
        'entropy_mean': round(mean, 6),
        'entropy_var': round(var, 6),
        'entropy_std': round(var ** 0.5, 6),
        'top_characters': sorted(results, key=lambda x: -x['entropy'])[:5],
    }


# ─── Dataset ────────────────────────────────────────
# (year, winner_title, winner_url, [(nominee_title, nominee_url), ...])

def imsdb(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"

DATASET = [
    # 1940s-1950s
    (1950, "Sunset Boulevard", imsdb("Sunset-Boulevard"),
     [("All About Eve", imsdb("All-About-Eve"))]),

    (1953, "On the Waterfront", imsdb("On-the-Waterfront"),
     [("The Barefoot Contessa", imsdb("Barefoot-Contessa,-The"))]),

    # 1960s
    (1960, "The Apartment", imsdb("Apartment,-The"),
     [("North by Northwest", imsdb("North-by-Northwest"))]),

    (1967, "In the Heat of the Night", imsdb("In-the-Heat-of-the-Night"),
     [("Bonnie and Clyde", imsdb("Bonnie-and-Clyde")),
      ("The Graduate", imsdb("Graduate,-The"))]),

    (1969, "Butch Cassidy and the Sundance Kid", imsdb("Butch-Cassidy-and-the-Sundance-Kid"),
     [("Easy Rider", imsdb("Easy-Rider"))]),

    # 1970s
    (1972, "The Godfather", imsdb("Godfather"),
     [("Cabaret", imsdb("Cabaret"))]),

    (1974, "Chinatown", imsdb("Chinatown"),
     [("The Conversation", imsdb("Conversation,-The"))]),

    (1976, "Network", imsdb("Network"),
     [("Rocky", imsdb("Rocky")),
      ("Taxi Driver", imsdb("Taxi-Driver"))]),

    (1977, "Annie Hall", imsdb("Annie-Hall"),
     [("Star Wars", imsdb("Star-Wars-A-New-Hope"))]),

    (1979, "Kramer vs. Kramer", imsdb("Kramer-vs-Kramer"),
     [("Apocalypse Now", imsdb("Apocalypse-Now"))]),

    # 1980s
    (1980, "Ordinary People", imsdb("Ordinary-People"),
     [("Raging Bull", imsdb("Raging-Bull"))]),

    (1982, "Gandhi", imsdb("Gandhi"),
     [("E.T. the Extra-Terrestrial", imsdb("E-T--the-Extra-Terrestrial")),
      ("Tootsie", imsdb("Tootsie"))]),

    (1984, "Amadeus", imsdb("Amadeus"),
     [("The Killing Fields", imsdb("Killing-Fields,-The"))]),

    (1986, "Platoon", imsdb("Platoon"),
     [("Top Gun", imsdb("Top-Gun"))]),

    (1988, "Rain Man", imsdb("Rain-Man"),
     [("Die Hard", imsdb("Die-Hard"))]),

    (1989, "Dead Poets Society", imsdb("Dead-Poets-Society"),
     [("When Harry Met Sally", imsdb("When-Harry-Met-Sally"))]),

    # 1990s
    (1990, "Dances with Wolves", imsdb("Dances-with-Wolves"),
     [("Goodfellas", imsdb("Goodfellas"))]),

    (1991, "The Silence of the Lambs", imsdb("Silence-of-the-Lambs,-The"),
     [("Thelma and Louise", imsdb("Thelma-and-Louise")),
      ("JFK", imsdb("JFK"))]),

    (1992, "Unforgiven", imsdb("Unforgiven"),
     [("A Few Good Men", imsdb("A-Few-Good-Men")),
      ("Scent of a Woman", imsdb("Scent-of-a-Woman"))]),

    (1993, "Schindler's List", imsdb("Schindler's-List"),
     [("The Fugitive", imsdb("Fugitive,-The")),
      ("The Piano", imsdb("Piano,-The"))]),

    (1994, "Forrest Gump", imsdb("Forrest-Gump"),
     [("Pulp Fiction", imsdb("Pulp-Fiction")),
      ("The Shawshank Redemption", imsdb("Shawshank-Redemption,-The"))]),

    (1995, "Braveheart", imsdb("Braveheart"),
     [("Sense and Sensibility", imsdb("Sense-and-Sensibility")),
      ("Usual Suspects", imsdb("Usual-Suspects,-The"))]),

    (1996, "The English Patient", imsdb("English-Patient,-The"),
     [("Fargo", imsdb("Fargo")),
      ("Jerry Maguire", imsdb("Jerry-Maguire"))]),

    (1997, "Good Will Hunting", imsdb("Good-Will-Hunting"),
     [("As Good as It Gets", imsdb("As-Good-As-It-Gets")),
      ("Titanic", imsdb("Titanic"))]),

    (1998, "Shakespeare in Love", imsdb("Shakespeare-in-Love"),
     [("Saving Private Ryan", imsdb("Saving-Private-Ryan")),
      ("The Truman Show", imsdb("Truman-Show,-The"))]),

    (1999, "American Beauty", imsdb("American-Beauty"),
     [("The Sixth Sense", imsdb("Sixth-Sense,-The")),
      ("The Green Mile", imsdb("Green-Mile,-The"))]),

    # 2000s
    (2000, "Almost Famous", imsdb("Almost-Famous"),
     [("Gladiator", imsdb("Gladiator")),
      ("Erin Brockovich", imsdb("Erin-Brockovich"))]),

    (2001, "A Beautiful Mind", imsdb("Beautiful-Mind,-A"),
     [("Gosford Park", imsdb("Gosford-Park")),
      ("Moulin Rouge", imsdb("Moulin-Rouge"))]),

    (2002, "The Pianist", imsdb("Pianist,-The"),
     [("Gangs of New York", imsdb("Gangs-of-New-York")),
      ("Minority Report", imsdb("Minority-Report"))]),

    (2003, "Lost in Translation", imsdb("Lost-in-Translation"),
     [("Finding Nemo", imsdb("Finding-Nemo")),
      ("Master and Commander", imsdb("Master-and-Commander"))]),

    (2004, "Eternal Sunshine", imsdb("Eternal-Sunshine-of-the-Spotless-Mind"),
     [("Hotel Rwanda", imsdb("Hotel-Rwanda")),
      ("The Aviator", imsdb("Aviator,-The"))]),

    (2005, "Crash", imsdb("Crash"),
     [("Brokeback Mountain", imsdb("Brokeback-Mountain")),
      ("Good Night and Good Luck", imsdb("Good-Night,-and-Good-Luck"))]),

    (2006, "The Departed", imsdb("Departed,-The"),
     [("Little Miss Sunshine", imsdb("Little-Miss-Sunshine")),
      ("Babel", imsdb("Babel"))]),

    (2007, "Juno", imsdb("Juno"),
     [("Michael Clayton", imsdb("Michael-Clayton")),
      ("Ratatouille", imsdb("Ratatouille"))]),

    (2008, "Slumdog Millionaire", imsdb("Slumdog-Millionaire"),
     [("The Dark Knight", imsdb("Dark-Knight,-The")),
      ("Milk", imsdb("Milk"))]),

    (2009, "The Hurt Locker", imsdb("Hurt-Locker,-The"),
     [("Inglourious Basterds", imsdb("Inglourious-Basterds")),
      ("Up", imsdb("Up"))]),

    (2010, "The King's Speech", imsdb("King's-Speech,-The"),
     [("Black Swan", imsdb("Black-Swan")),
      ("Inception", imsdb("Inception")),
      ("The Fighter", imsdb("Fighter,-The")),
      ("The Kids Are All Right", imsdb("Kids-Are-All-Right,-The"))]),

    (2011, "The Artist", imsdb("Artist,-The"),
     [("The Descendants", imsdb("Descendants,-The")),
      ("Moneyball", imsdb("Moneyball")),
      ("Drive", imsdb("Drive"))]),

    (2012, "Argo", imsdb("Argo"),
     [("Django Unchained", imsdb("Django-Unchained")),
      ("Silver Linings Playbook", imsdb("Silver-Linings-Playbook")),
      ("Lincoln", imsdb("Lincoln"))]),

    (2013, "12 Years a Slave", imsdb("12-Years-a-Slave"),
     [("The Wolf of Wall Street", imsdb("Wolf-of-Wall-Street,-The")),
      ("Her", imsdb("Her"))]),

    (2014, "Birdman", imsdb("Birdman"),
     [("Whiplash", imsdb("Whiplash")),
      ("The Grand Budapest Hotel", imsdb("Grand-Budapest-Hotel,-The")),
      ("Interstellar", imsdb("Interstellar"))]),

    (2015, "Spotlight", imsdb("Spotlight"),
     [("The Big Short", imsdb("Big-Short,-The")),
      ("The Revenant", imsdb("Revenant,-The")),
      ("Ex Machina", imsdb("Ex-Machina"))]),
]


# ─── Main ────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SCREENPLAY ENTROPY — FULL HISTORICAL ANALYSIS")
    print("  Academy Awards ~1950-2015 | IMSDB scripts")
    print("  가설: 수상작의 캐릭터간 entropy 분산 > 후보작")
    print("=" * 70)

    results = []  # (year, is_winner, analysis_dict)
    success = 0
    fail = 0

    for year, w_title, w_url, nominees in DATASET:
        print(f"\n── {year} ──")

        # Winner
        print(f"  ★ {w_title}... ", end='', flush=True)
        a = analyze_one(w_title, w_url)
        if a:
            results.append((year, True, a))
            print(f"✓ {a['n_characters']} chars, var={a['entropy_var']:.6f}")
            success += 1
        else:
            print("✗ skip")
            fail += 1

        # Nominees
        for n_title, n_url in nominees:
            print(f"    {n_title}... ", end='', flush=True)
            a = analyze_one(n_title, n_url)
            if a:
                results.append((year, False, a))
                print(f"✓ {a['n_characters']} chars, var={a['entropy_var']:.6f}")
                success += 1
            else:
                print("✗ skip")
                fail += 1

        time.sleep(0.3)  # be polite

    # ─── Analysis ────────────────────────────────

    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS — {success} scripts analyzed, {fail} skipped")
    print(f"{'=' * 70}")

    # 연도별 비교
    years_with_both = set()
    for year, is_w, _ in results:
        if is_w:
            nom_exists = any(y == year and not w for y, w, _ in results)
            if nom_exists:
                years_with_both.add(year)

    winner_wins = 0
    winner_total = 0
    all_w_var = []
    all_n_var = []

    print(f"\n  {'Year':<6} {'Winner':<30} {'W var':>10} {'Nom avg var':>12} {'Ratio':>8} {'W>N':>4}")
    print(f"  {'─'*6} {'─'*30} {'─'*10} {'─'*12} {'─'*8} {'─'*4}")

    for year in sorted(years_with_both):
        w = [(y, a) for y, is_w, a in results if y == year and is_w]
        n = [(y, a) for y, is_w, a in results if y == year and not is_w]
        if not w or not n:
            continue

        winner_total += 1
        w_var = w[0][1]['entropy_var']
        n_avg = sum(a['entropy_var'] for _, a in n) / len(n)

        won = w_var > n_avg
        if won:
            winner_wins += 1

        all_w_var.append(w_var)
        all_n_var.append(n_avg)

        mark = "✓" if won else "✗"
        ratio = w_var / (n_avg + 1e-10)
        print(f"  {year:<6} {w[0][1]['title']:<30} {w_var:>10.6f} {n_avg:>12.6f} {ratio:>7.2f}x {mark:>4}")

    # 최종
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")

    if all_w_var and all_n_var:
        mean_w = sum(all_w_var) / len(all_w_var)
        mean_n = sum(all_n_var) / len(all_n_var)
        win_rate = winner_wins / winner_total * 100

        print(f"\n  분석 연도:          {winner_total}")
        print(f"  수상작 평균 분산:    {mean_w:.6f}")
        print(f"  후보작 평균 분산:    {mean_n:.6f}")
        print(f"  비율:               {mean_w / (mean_n + 1e-10):.2f}x")
        print(f"  수상작 승률:        {winner_wins}/{winner_total} ({win_rate:.1f}%)")

        print(f"\n  {'─' * 50}")
        if win_rate > 50:
            print(f"  결론: 수상작이 {win_rate:.0f}%의 연도에서 더 높은 entropy 분산.")
            print(f"  → 발견 11 지지.")
        else:
            print(f"  결론: 수상작 승률 {win_rate:.0f}% — 유의미하지 않음.")
            print(f"  → 발견 11 기각 또는 추가 검증 필요.")

    # JSON 저장
    out = {
        'summary': {
            'years_analyzed': winner_total,
            'winner_avg_var': round(sum(all_w_var) / len(all_w_var), 6) if all_w_var else 0,
            'nominee_avg_var': round(sum(all_n_var) / len(all_n_var), 6) if all_n_var else 0,
            'win_rate': f"{winner_wins}/{winner_total}",
        },
        'per_year': [
            {'year': y, 'is_winner': w, 'analysis': a}
            for y, w, a in results
        ],
    }
    with open('screenplay_entropy_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: screenplay_entropy_results.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
