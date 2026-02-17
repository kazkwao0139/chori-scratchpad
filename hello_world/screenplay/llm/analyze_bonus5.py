#!/usr/bin/env python
"""번외 5: EEAAO, Drive My Car, Farewell My Concubine — LLM 3-axis analysis."""
import os, sys, json, re, time, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ["CLAUDE_API_KEY"]
BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "screenplay"

PROMPT_SCREENPLAY = """You are a professional screenplay analyst. Analyze the following screenplay with STRICT numerical scoring.

SCREENPLAY TEXT:
---
{text}
---

Evaluate on THREE axes. Be brutally honest. Do NOT inflate scores.

1. INFORMATION DENSITY (0-100): What percentage of the text DIRECTLY serves the narrative?
   80+ = almost every word earns its place. 50-60 = average. Below 40 = significant bloat.

2. CONTEXT RETENTION / PAYOFF RATE (0-100): How well does the screenplay SET UP and PAY OFF its elements?
   Score = (paid-off setups / total setups) x 100. 90+ = masterful. 60-70 = decent. Below 50 = poor.

3. DIRECTION-TO-DIALOGUE RATIO: Estimate percentage of stage directions vs dialogue.
   Rate the balance quality from 1-10.

RESPOND IN THIS EXACT JSON FORMAT ONLY (no markdown, no code fences):
{{"info_density": <0-100>, "info_density_reasoning": "<reasoning>", "payoff_rate": <0-100>, "payoff_setups": "<list key setups>", "payoff_reasoning": "<reasoning>", "direction_pct": <0-100>, "dialogue_pct": <0-100>, "balance_rating": <1-10>, "balance_reasoning": "<reasoning>", "overall_craft_notes": "<1 paragraph>"}}"""

PROMPT_TRANSCRIPT = """You are a professional screenplay analyst. Analyze the following text with STRICT numerical scoring.

IMPORTANT: This is a DIALOGUE TRANSCRIPT of the film "{title}". It contains primarily spoken dialogue with minimal stage directions. For Direction:Dialogue ratio, estimate what the FULL screenplay would look like based on the film's known style.

TEXT:
---
{text}
---

Evaluate on THREE axes. Be brutally honest. Do NOT inflate scores.

1. INFORMATION DENSITY (0-100): What percentage of the dialogue DIRECTLY serves the narrative?
   80+ = almost every line earns its place. 50-60 = average. Below 40 = bloat.

2. CONTEXT RETENTION / PAYOFF RATE (0-100): How well does the text SET UP and PAY OFF its elements?
   Score = (paid-off setups / total setups) x 100. 90+ = masterful. 60-70 = decent. Below 50 = poor.

3. DIRECTION-TO-DIALOGUE RATIO: Estimate based on the ACTUAL FILM, not this transcript.
   Rate the balance quality from 1-10.

RESPOND IN THIS EXACT JSON FORMAT ONLY (no markdown, no code fences):
{{"info_density": <0-100>, "info_density_reasoning": "<reasoning>", "payoff_rate": <0-100>, "payoff_setups": "<list key setups>", "payoff_reasoning": "<reasoning>", "direction_pct": <0-100>, "dialogue_pct": <0-100>, "balance_rating": <1-10>, "balance_reasoning": "<reasoning>", "overall_craft_notes": "<1 paragraph>"}}"""


def call_claude(prompt, max_retries=5):
    for attempt in range(max_retries):
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=300,
        )
        if r.status_code == 200:
            return r.json()["content"][0]["text"]
        if r.status_code == 529:
            wait = 10 * (attempt + 1)
            print(f"    Overloaded, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        print(f"    API error: {r.status_code} {r.text[:300]}")
        return None
    return None


def parse_result(response, text_len):
    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```\w*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean)
    json_match = re.search(r'\{[\s\S]*\}', clean)
    if json_match:
        result = json.loads(json_match.group())
        result['text_len'] = text_len
        return result
    return None


# ─── Films ───
films = [
    {
        "title": "Everything Everywhere All at Once",
        "file": BASE / "_copyrighted" / "_script_cache" / "Everything_Everywhere_All_At_Once.txt",
        "type": "screenplay",
        "out": "eeaao_llm_analysis.json",
    },
    {
        "title": "Drive My Car",
        "url": "https://subslikescript.com/movie/Drive_My_Car-14039582",
        "type": "transcript",
        "out": "drive_my_car_llm_analysis.json",
    },
    {
        "title": "Farewell My Concubine",
        "url": "https://subslikescript.com/movie/Farewell_My_Concubine-106332",
        "type": "transcript",
        "out": "farewell_my_concubine_llm_analysis.json",
    },
]


for film in films:
    title = film["title"]
    out_path = OUT_DIR / film["out"]

    if out_path.exists():
        print(f"[SKIP] {title} — already done")
        continue

    print(f"\n[{title}]")

    # Get text
    if "file" in film:
        text = film["file"].read_text(encoding='utf-8')
        print(f"  Loaded from file ({len(text):,} chars)")
    else:
        from bs4 import BeautifulSoup
        r = requests.get(film["url"], timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article", class_="main-article")
        text = article.get_text(separator="\n").strip() if article else ""
        print(f"  Fetched transcript ({len(text):,} chars)")

    if len(text) < 5000:
        print(f"  SKIP — too short")
        continue

    # Truncate if needed
    if len(text) > 120000:
        text = text[:60000] + "\n\n[...MIDDLE SECTION OMITTED...]\n\n" + text[-60000:]
        print(f"  (truncated to {len(text):,} chars)")

    # Build prompt
    if film["type"] == "screenplay":
        prompt = PROMPT_SCREENPLAY.format(text=text)
    else:
        prompt = PROMPT_TRANSCRIPT.format(title=title, text=text)

    print(f"  Sending to Claude ({len(text):,} chars)...")
    response = call_claude(prompt)
    if not response:
        print(f"  FAILED")
        continue

    result = parse_result(response, len(text))
    if result:
        result["title"] = title
        result["type"] = film["type"]
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  -> Density={result['info_density']}, Payoff={result['payoff_rate']}, "
              f"Dir/Dial={result['direction_pct']}/{result['dialogue_pct']}, "
              f"Balance={result['balance_rating']}")
    else:
        print(f"  FAILED to parse response")


# ─── Summary ───
print(f"\n\n{'='*70}")
print(f"{'Title':<40} {'Density':>7} {'Payoff':>7} {'Dir%':>5} {'Dial%':>5} {'Bal':>4}")
print("-" * 70)
for film in films:
    out_path = OUT_DIR / film["out"]
    if out_path.exists():
        d = json.load(open(out_path, encoding='utf-8'))
        src = "screenplay" if film["type"] == "screenplay" else "transcript"
        print(f"{film['title']+' ('+src+')':40} {d['info_density']:>7} {d['payoff_rate']:>7} "
              f"{d['direction_pct']:>5} {d['dialogue_pct']:>5} {d['balance_rating']:>4}")
