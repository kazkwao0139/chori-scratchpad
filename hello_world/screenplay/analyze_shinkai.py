#!/usr/bin/env python
"""Shinkai 3-film LLM 3-axis analysis from subtitle text."""
import os, sys, json, re, time, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ["CLAUDE_API_KEY"]
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "screenplay"


def extract_srt_dialogue(filepath):
    """Extract dialogue text from SRT file, stripping timestamps and tags."""
    text = filepath.read_text(encoding='utf-8')
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip empty, index numbers, timestamps
        if not line:
            continue
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'\d{2}:\d{2}:\d{2}', line):
            continue
        # Strip HTML tags
        line = re.sub(r'<[^>]+>', '', line)
        # Skip subtitle credits
        if 'subscene' in line.lower() or 'sub.trader' in line.lower() or 'opensubtitles' in line.lower():
            continue
        if line:
            lines.append(line)
    return '\n'.join(lines)


PROMPT = """You are a professional screenplay analyst. Analyze the following text with STRICT numerical scoring.

IMPORTANT: This is a DIALOGUE TRANSCRIPT of the animated film "{title}" ({year}) directed by {director}. It contains only spoken dialogue extracted from subtitles — no stage directions. For Direction:Dialogue ratio, estimate what the FULL screenplay would look like based on the film's known style.

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
{{"info_density": <0-100>, "info_density_reasoning": "<reasoning>", "payoff_rate": <0-100>, "payoff_setups": "<list key setups>", "payoff_reasoning": "<reasoning>", "direction_pct": <0-100>, "dialogue_pct": <0-100>, "balance_rating": <1-10>, "balance_reasoning": "<reasoning>", "overall_craft_notes": "<1 paragraph>", "director": "{director}", "title": "{title}"}}"""


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


films = [
    {
        "title": "Your Name",
        "year": "2016",
        "director": "Makoto Shinkai",
        "srt": BASE / "Your Name. (2016) -ja.srt",
        "out": "your_name_llm_analysis.json",
    },
    {
        "title": "Weathering with You",
        "year": "2019",
        "director": "Makoto Shinkai",
        "srt": BASE / "Weathering With You (Japanese SDH).srt",
        "out": "weathering_with_you_llm_analysis.json",
    },
    {
        "title": "Suzume",
        "year": "2022",
        "director": "Makoto Shinkai",
        "srt": BASE / "Suzume.no.Tojimari.2022.2160p.UHD.Blu-ray.Remux.HEVC.HDR.DTS-HD.MA.5.1-eXterminator.srt",
        "out": "suzume_llm_analysis.json",
    },
]

for film in films:
    out_path = OUT_DIR / film["out"]
    if out_path.exists():
        print(f"[SKIP] {film['title']}")
        continue

    print(f"\n[{film['title']}]")
    dialogue = extract_srt_dialogue(film["srt"])
    print(f"  Extracted dialogue: {len(dialogue):,} chars")

    prompt = PROMPT.format(
        title=film["title"],
        year=film["year"],
        director=film["director"],
        text=dialogue,
    )

    print(f"  Sending to Claude...")
    response = call_claude(prompt)
    if not response:
        print(f"  FAILED")
        continue

    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```\w*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean)
    json_match = re.search(r'\{[\s\S]*\}', clean)
    if json_match:
        result = json.loads(json_match.group())
        result['text_len'] = len(dialogue)
        result['type'] = 'transcript'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  -> Density={result['info_density']}, Payoff={result['payoff_rate']}, "
              f"Dir/Dial={result['direction_pct']}/{result['dialogue_pct']}, "
              f"Balance={result['balance_rating']}")
    else:
        print(f"  FAILED to parse")

# Summary
print(f"\n\n{'='*70}")
print(f"{'Title':<30} {'Density':>7} {'Payoff':>7} {'Dir%':>5} {'Dial%':>5} {'Bal':>4}")
print("-" * 70)
for film in films:
    out_path = OUT_DIR / film["out"]
    if out_path.exists():
        d = json.load(open(out_path, encoding='utf-8'))
        print(f"{film['title']:<30} {d['info_density']:>7} {d['payoff_rate']:>7} "
              f"{d['direction_pct']:>5} {d['dialogue_pct']:>5} {d['balance_rating']:>4}")
