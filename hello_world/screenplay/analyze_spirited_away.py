#!/usr/bin/env python
"""Spirited Away LLM 3-axis analysis."""
import os, sys, json, re, time, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ["CLAUDE_API_KEY"]
text = (BASE / "_copyrighted" / "_script_cache" / "Spirited_Away.txt").read_text(encoding='utf-8')
print(f"Script length: {len(text):,} chars")

PROMPT = """You are a professional screenplay analyst. Analyze the following text with STRICT numerical scoring.

IMPORTANT: This is a DIALOGUE TRANSCRIPT of the animated film "Spirited Away" (2001) by Hayao Miyazaki. It contains only spoken dialogue with no stage directions or action lines. This means:
- For Information Density and Payoff Rate: analyze based on the dialogue content as-is.
- For Direction:Dialogue ratio: since this is dialogue-only, estimate what the FULL screenplay would look like based on the film itself (Miyazaki is known for heavy visual storytelling with long wordless sequences).

TEXT:
---
""" + text + """
---

Evaluate on THREE axes. Be brutally honest. Do NOT inflate scores.

1. INFORMATION DENSITY (0-100): What percentage of the dialogue DIRECTLY serves the narrative?
2. CONTEXT RETENTION / PAYOFF RATE (0-100): How well does the screenplay SET UP and PAY OFF its elements? Score = (paid-off setups / total setups) x 100
3. DIRECTION-TO-DIALOGUE RATIO: Estimate based on the ACTUAL FILM, not this transcript. Rate the balance 1-10.

RESPOND IN THIS EXACT JSON FORMAT ONLY (no markdown, no code fences):
{
    "info_density": <0-100>,
    "info_density_reasoning": "<reasoning>",
    "payoff_rate": <0-100>,
    "payoff_setups": "<list key setups>",
    "payoff_reasoning": "<reasoning>",
    "direction_pct": <0-100>,
    "dialogue_pct": <0-100>,
    "balance_rating": <1-10>,
    "balance_reasoning": "<reasoning>",
    "overall_craft_notes": "<1 paragraph>",
    "director": "Hayao Miyazaki",
    "title": "Spirited Away"
}"""

print("Sending to Claude...")

for attempt in range(5):
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
            "messages": [{"role": "user", "content": PROMPT}],
        },
        timeout=300,
    )
    if r.status_code == 200:
        break
    if r.status_code == 529:
        wait = 10 * (attempt + 1)
        print(f"  Overloaded, waiting {wait}s...")
        time.sleep(wait)
        continue
    print(f"  API error: {r.status_code} {r.text[:300]}")
    sys.exit(1)

response = r.json()["content"][0]["text"]
clean = response.strip()
if clean.startswith("```"):
    clean = re.sub(r'^```\w*\n?', '', clean)
    clean = re.sub(r'\n?```$', '', clean)

json_match = re.search(r'\{[\s\S]*\}', clean)
if json_match:
    result = json.loads(json_match.group())
    result['text_len'] = len(text)

    print(f"\n=== Spirited Away ===")
    print(f"  Density: {result['info_density']}")
    print(f"  Payoff:  {result['payoff_rate']}")
    print(f"  Dir/Dial: {result['direction_pct']}/{result['dialogue_pct']}")
    print(f"  Balance: {result['balance_rating']}")
    print(f"  Density reason: {result['info_density_reasoning']}")
    print(f"  Payoff reason: {result['payoff_reasoning']}")
    print(f"  Setups: {result['payoff_setups']}")
    print(f"  Craft: {result['overall_craft_notes']}")

    out_path = str(BASE / "screenplay" / "spirited_away_llm_analysis.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")
else:
    print(f"Failed to parse JSON from response")
    print(response[:500])
