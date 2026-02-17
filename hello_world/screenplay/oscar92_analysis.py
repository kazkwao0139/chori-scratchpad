#!/usr/bin/env python
"""
92nd Academy Awards Best Picture nominees — LLM 3-axis analysis.
Fetches scripts from IMSDB/ScriptSlug, runs Claude Sonnet analysis.
"""

import json, os, time, re, sys, requests
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from bs4 import BeautifulSoup

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE / "_copyrighted" / "_script_cache"
CHECKPOINT = BASE / "screenplay" / "oscar92_checkpoint.json"
API_KEY_FILE = None  # use os.environ["CLAUDE_API_KEY"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── 92nd Oscar Best Picture Nominees ───
# (cache_name, display_title, imsdb_slug, scriptslug_slug)
NOMINEES = [
    ("1917",                "1917",                         "1917",                         "1917-2019"),
    ("Ford_v_Ferrari",      "Ford v Ferrari",               "Ford-v-Ferrari",               "ford-v-ferrari-2019"),
    ("The_Irishman",        "The Irishman",                 "Irishman,-The",                "the-irishman-2019"),
    ("Jojo_Rabbit",         "Jojo Rabbit",                  "Jojo-Rabbit",                  "jojo-rabbit-2019"),
    ("Joker",               "Joker",                        "Joker",                        "joker-2019"),
    ("Little_Women_2019",   "Little Women",                 "Little-Women",                 "little-women-2019"),
    ("Marriage_Story",      "Marriage Story",                "Marriage-Story",                "marriage-story-2019"),
    ("Once_Upon_Hollywood", "Once Upon a Time in Hollywood","Once-Upon-a-Time-in-Hollywood","once-upon-a-time-in-hollywood-2019"),
]
# Parasite already analyzed separately — will merge at the end

# ─── Script Fetchers ───
def fetch_imsdb(slug):
    """Fetch from IMSDB."""
    url = f"https://imsdb.com/scripts/{slug}.html"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (educational research)"})
        if r.status_code != 200:
            print(f"    IMSDB HTTP {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        scrtext = soup.find("td", class_="scrtext")
        if scrtext:
            text = scrtext.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text) > 5000:
                return text
        pre = soup.find("pre")
        if pre:
            text = pre.get_text(separator="\n").strip()
            if len(text) > 5000:
                return text
        print(f"    IMSDB: no script text found")
        return None
    except Exception as e:
        print(f"    IMSDB error: {e}")
        return None


def fetch_scriptslug(slug):
    """Fetch PDF from Script Slug, extract text with pdfplumber."""
    url = f"https://assets.scriptslug.com/live/pdf/scripts/{slug}.pdf"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (educational research)"})
        if r.status_code != 200:
            print(f"    ScriptSlug HTTP {r.status_code}")
            return None

        import pdfplumber, io
        pdf = pdfplumber.open(io.BytesIO(r.content))
        pages = []
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        pdf.close()
        text = "\n\n".join(pages)
        if len(text) > 5000:
            return text
        print(f"    ScriptSlug: extracted text too short ({len(text)})")
        return None
    except Exception as e:
        print(f"    ScriptSlug error: {e}")
        return None


def get_script(cache_name, imsdb_slug, scriptslug_slug):
    """Get script: cache → IMSDB → ScriptSlug."""
    # 1. Check cache
    path = CACHE_DIR / f"{cache_name}.txt"
    if path.exists():
        text = path.read_text(encoding='utf-8')
        print(f"    Cache hit ({len(text):,} chars)")
        return text

    # 2. IMSDB
    print(f"    Trying IMSDB...")
    text = fetch_imsdb(imsdb_slug)
    if text:
        path.write_text(text, encoding='utf-8')
        print(f"    IMSDB OK ({len(text):,} chars)")
        time.sleep(1.5)
        return text

    # 3. Script Slug
    print(f"    Trying ScriptSlug...")
    text = fetch_scriptslug(scriptslug_slug)
    if text:
        path.write_text(text, encoding='utf-8')
        print(f"    ScriptSlug OK ({len(text):,} chars)")
        time.sleep(1.5)
        return text

    return None


# ─── Claude API ───
ANALYSIS_PROMPT = """You are a professional screenplay analyst. Analyze the following screenplay with STRICT numerical scoring.

SCREENPLAY TEXT:
---
{script_text}
---

Evaluate on THREE axes. Be brutally honest — do NOT inflate scores.

## 1. INFORMATION DENSITY (0-100)
What percentage of the text DIRECTLY serves the narrative?
- Counts: plot advancement, character development, essential world-building, necessary exposition
- Does NOT count: redundant descriptions, filler dialogue, unnecessary tangents, over-written passages, self-indulgent digressions
- Score 80+ = almost every word earns its place (e.g., Hemingway-tight)
- Score 50-60 = average, some fat to trim
- Score below 40 = significant bloat, lots of unnecessary content

## 2. CONTEXT RETENTION / PAYOFF RATE (0-100)
How well does the screenplay SET UP and PAY OFF its elements?
- Identify: foreshadowing, Chekhov's guns, planted dialogue, recurring motifs, callback references
- Score = (paid-off setups / total setups) x 100
- Score 90+ = masterful, nearly everything planted is harvested
- Score 60-70 = decent, some loose threads
- Score below 50 = many dangling setups, poor structural discipline

## 3. DIRECTION-TO-DIALOGUE RATIO ASSESSMENT
Estimate the ratio of stage directions/action lines vs dialogue.
- Report as percentage: e.g., "Direction 45% / Dialogue 55%"
- Assess: Is this ratio APPROPRIATE for the genre and story?
- Rate the balance quality from 1-10

RESPOND IN THIS EXACT JSON FORMAT ONLY (no markdown, no code fences, no other text):
{{
    "info_density": <0-100>,
    "info_density_reasoning": "<2-3 sentences>",
    "payoff_rate": <0-100>,
    "payoff_setups": "<list key setups found>",
    "payoff_reasoning": "<2-3 sentences>",
    "direction_pct": <0-100>,
    "dialogue_pct": <0-100>,
    "balance_rating": <1-10>,
    "balance_reasoning": "<2-3 sentences>",
    "overall_craft_notes": "<1 paragraph on screenplay craft quality>"
}}"""


def call_claude(api_key, prompt, max_retries=5):
    """Call Claude API with retry for 529 overloaded."""
    for attempt in range(max_retries):
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
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
    print(f"    Max retries exceeded")
    return None


def analyze_script(api_key, title, text):
    """Send screenplay to Claude for 3-axis analysis."""
    if len(text) > 120000:
        text = text[:60000] + "\n\n[...MIDDLE SECTION OMITTED FOR LENGTH...]\n\n" + text[-60000:]
        print(f"    (truncated to {len(text):,} chars)")

    prompt = ANALYSIS_PROMPT.format(script_text=text)
    print(f"    Sending to Claude ({len(text):,} chars)...")
    response = call_claude(api_key, prompt)
    if not response:
        return None

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)
        json_match = re.search(r'\{[\s\S]*\}', clean)
        if json_match:
            result = json.loads(json_match.group())
            result['text_len'] = len(text)
            result['title'] = title
            return result
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        print(f"    Preview: {response[:500]}")
    return None


# ─── Main ───
def main():
    api_key = os.environ["CLAUDE_API_KEY"]

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT.exists():
        checkpoint = json.load(open(CHECKPOINT, encoding='utf-8'))

    print(f"=== 92nd Oscar Best Picture — LLM 3-Axis Analysis ===")
    print(f"Targets: {len(NOMINEES)} nominees (+ Parasite from existing data)")
    print(f"Already done: {len(checkpoint)}\n")

    for cache_name, title, imsdb_slug, ss_slug in NOMINEES:
        if title in checkpoint:
            print(f"[SKIP] {title}")
            continue

        print(f"\n[{title}]")

        # 1. Get script
        text = get_script(cache_name, imsdb_slug, ss_slug)
        if not text:
            print(f"  FAILED — could not get script")
            checkpoint[title] = {"error": "script_not_found"}
            continue

        # 2. Analyze
        result = analyze_script(api_key, title, text)
        if result:
            checkpoint[title] = result
            print(f"  -> Density={result['info_density']}, Payoff={result['payoff_rate']}, "
                  f"Dir/Dial={result['direction_pct']}/{result['dialogue_pct']}, "
                  f"Balance={result['balance_rating']}")
        else:
            checkpoint[title] = {"error": "analysis_failed"}
            print(f"  FAILED — API analysis error")

        # Save checkpoint after each
        with open(CHECKPOINT, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    # ─── Summary ───
    print(f"\n\n{'='*70}")
    print(f"{'Title':<38} {'Density':>7} {'Payoff':>7} {'Dir%':>5} {'Dial%':>5} {'Bal':>4}")
    print("-" * 70)

    # Include Parasite from existing analysis
    parasite_path = BASE / "screenplay" / "parasite_llm_analysis_3axis.json"
    if parasite_path.exists():
        p = json.load(open(parasite_path, encoding='utf-8'))
        print(f"{'Parasite (Winner)':38} {p['info_density']:>7} {p['payoff_rate']:>7} "
              f"{p['direction_pct']:>5} {p['dialogue_pct']:>5} {p['balance_rating']:>4}")

    for title, data in checkpoint.items():
        if 'error' in data:
            print(f"{title:38} {'ERROR':>7}")
            continue
        print(f"{title:38} {data['info_density']:>7} {data['payoff_rate']:>7} "
              f"{data['direction_pct']:>5} {data['dialogue_pct']:>5} {data['balance_rating']:>4}")

    print(f"\nCheckpoint saved to: {CHECKPOINT}")


if __name__ == "__main__":
    main()
