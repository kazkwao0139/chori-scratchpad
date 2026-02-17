#!/usr/bin/env python
"""End of Evangelion — timestamp Direction% + LLM 3-axis analysis."""
import os, sys, json, re, time, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ["CLAUDE_API_KEY"]
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "screenplay"
VTT_FILE = BASE / "엔드 오브 에반게리온"

# ── 1. Timestamp-based Direction % ──
print("=" * 60)
print("TIMESTAMP ANALYSIS: End of Evangelion (1997)")
print("=" * 60)

text = VTT_FILE.read_text(encoding='utf-8')
timestamps = re.findall(
    r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})',
    text
)

def to_sec(t):
    h, m, s = t.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

entries = [(to_sec(s), to_sec(e)) for s, e in timestamps]
print(f"Entries: {len(entries)}")

# Runtime: first sub start to last sub end
runtime = entries[-1][1] - entries[0][0]
print(f"Runtime (sub span): {runtime/60:.1f} min")

# Dialogue time = union of all subtitle intervals
merged = []
for s, e in sorted(entries):
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))

dialogue_time = sum(e - s for s, e in merged)
direction_time = runtime - dialogue_time
direction_pct = direction_time / runtime * 100
dialogue_pct = dialogue_time / runtime * 100

print(f"Dialogue time: {dialogue_time:.1f}s ({dialogue_pct:.1f}%)")
print(f"Direction time: {direction_time:.1f}s ({direction_pct:.1f}%)")

# Long silences (>5s)
silences = []
for i in range(1, len(merged)):
    gap = merged[i][0] - merged[i-1][1]
    if gap > 5:
        silences.append((merged[i-1][1], merged[i][0], gap))

silences.sort(key=lambda x: -x[2])
print(f"\nLong silences (>5s): {len(silences)}")
print("Top 10:")
for s, e, g in silences[:10]:
    m1, s1 = divmod(s, 60)
    m2, s2 = divmod(e, 60)
    print(f"  {int(m1):02d}:{s1:05.2f} ~ {int(m2):02d}:{s2:05.2f}  ({g:.1f}s)")

# ── 2. Extract dialogue for LLM ──
print("\n" + "=" * 60)
print("LLM ANALYSIS")
print("=" * 60)

lines = []
for line in text.split('\n'):
    line = line.strip()
    if not line:
        continue
    if re.match(r'^\d+$', line):
        continue
    if re.match(r'\d{2}:\d{2}:\d{2}', line):
        continue
    if line.startswith('WEBVTT') or line.startswith('NOTE'):
        continue
    # Strip VTT tags
    line = re.sub(r'<[^>]+>', '', line)
    # Skip pure sound effects (SE only)
    clean = line.strip()
    if not clean:
        continue
    lines.append(clean)

dialogue = '\n'.join(lines)
print(f"Extracted dialogue: {len(dialogue):,} chars")

PROMPT = """You are a professional screenplay analyst. Analyze the following text with STRICT numerical scoring.

IMPORTANT: This is a DIALOGUE TRANSCRIPT of the animated film "The End of Evangelion" (1997) directed by Hideaki Anno. It contains spoken dialogue extracted from Japanese Netflix subtitles — no stage directions. The text includes sound effect descriptions in parentheses (e.g., （セミの鳴き声）). For Direction:Dialogue ratio, estimate what the FULL film looks like based on the film's known style.

This is the theatrical film conclusion to Neon Genesis Evangelion. It replaces episodes 25-26 of the TV series. Known for: NERV invasion sequence, Instrumentality/Third Impact, Asuka's EVA-02 battle, extensive psychological sequences, and the iconic final scene.

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
{{"info_density": <0-100>, "info_density_reasoning": "<reasoning>", "payoff_rate": <0-100>, "payoff_setups": "<list key setups>", "payoff_reasoning": "<reasoning>", "direction_pct": <0-100>, "dialogue_pct": <0-100>, "balance_rating": <1-10>, "balance_reasoning": "<reasoning>", "overall_craft_notes": "<1 paragraph>", "director": "Hideaki Anno", "title": "The End of Evangelion"}}"""

out_path = OUT_DIR / "eoe_llm_analysis.json"
if out_path.exists():
    print(f"[SKIP] LLM analysis already exists")
    result = json.load(open(out_path, encoding='utf-8'))
else:
    prompt = PROMPT.format(text=dialogue)
    print(f"Sending to Claude ({len(prompt):,} chars)...")

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
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=300,
        )
        if r.status_code == 200:
            break
        if r.status_code == 529:
            wait = 10 * (attempt + 1)
            print(f"  Overloaded, waiting {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
            continue
        print(f"  API error: {r.status_code} {r.text[:300]}")
        break

    response = r.json()["content"][0]["text"]
    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```\w*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean)
    json_match = re.search(r'\{[\s\S]*\}', clean)
    result = json.loads(json_match.group())
    result['text_len'] = len(dialogue)
    result['type'] = 'transcript'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n  Density={result['info_density']}, Payoff={result['payoff_rate']}, "
      f"Dir/Dial={result['direction_pct']}/{result['dialogue_pct']}, "
      f"Balance={result['balance_rating']}")

# ── 3. Summary comparison ──
print(f"\n{'='*60}")
print("COMPARISON: All analyzed anime films")
print(f"{'='*60}")
print(f"{'Film':<25} {'Dir%(actual)':>12} {'Dir%(LLM)':>10} {'Density':>8} {'Payoff':>7}")
print("-" * 65)

comparisons = [
    ("Spirited Away", 60.9, 65, 72, 88),
    ("End of Evangelion", direction_pct, result['direction_pct'], result['info_density'], result['payoff_rate']),
    ("Suzume", 48.6, 65, 72, 81),
    ("Your Name", 47.8, 65, 72, 85),
    ("Weathering with You", 40.5, 65, 72, 81),
]
comparisons.sort(key=lambda x: -x[1])
for name, act, llm, dens, pay in comparisons:
    print(f"{name:<25} {act:>11.1f}% {llm:>9}% {dens:>8} {pay:>7}")
