#!/usr/bin/env python
"""
LLM-based screenplay quality analysis for Bonferroni-significant directors.
Three axes:
  1. Information Density - 서사에 필요한 단어 비율
  2. Context Retention (Payoff Rate) - 심어놓은 대사/설정의 회수율
  3. Direction-Dialogue Balance - 지문 vs 대사 균형 (기존 메트릭 교차검증)
"""

import json, os, time, re, sys
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE / "_copyrighted" / "_script_cache"
CHECKPOINT = BASE / "screenplay" / "llm_script_quality_checkpoint.json"
API_KEY_FILE = None  # use os.environ["CLAUDE_API_KEY"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Local file overrides (for scripts not on IMSDB) ───
LOCAL_FILES = {
    "The Dark Knight": BASE / "_copyrighted" / "dark_knight_script.txt",
}

# ─── IMSDB URL mapping (VERIFIED working slugs only) ───
# Format: director -> { influence_data_title: imsdb_slug }
TARGETS = {
    # *** BONFERRONI (p < 0.000685) ***
    "Stanley Kubrick": {
        "2001 A Space Odyssey": "2001-A-Space-Odyssey",
        "Shining, The": "Shining,-The",
        "Barry Lyndon": "Barry-Lyndon",
    },
    "Alfred Hitchcock": {
        "Psycho": "Psycho",
        "Rear Window": "Rear-Window",
    },
    "Christopher Nolan": {
        "The Dark Knight": None,  # from LOCAL_FILES
        "Inception": "Inception",
        "Interstellar": "Interstellar",
        "Memento": "Memento",
    },
    "Martin Scorsese": {
        "Departed, The": "Departed,-The",
        "Taxi Driver": "Taxi-Driver",
        "Wolf of Wall Street, The": "Wolf-of-Wall-Street,-The",
        "Casino": "Casino",
        "Raging Bull": "Raging-Bull",
        "Gangs of New York": "Gangs-of-New-York",
    },
    "Steven Spielberg": {
        "Schindler's List": "Schindler's-List",
        "Saving Private Ryan": "Saving-Private-Ryan",
        "Jurassic Park": "Jurassic-Park",
        "Jaws": "Jaws",
        "Minority Report": "Minority-Report",
    },

    # * NOMINAL (p < 0.05)
    "Quentin Tarantino": {
        "Pulp Fiction": "Pulp-Fiction",
        "Django Unchained": "Django-Unchained",
        "Inglourious Basterds": "Inglourious-Basterds",
        "Reservoir Dogs": "Reservoir-Dogs",
        "Jackie Brown": "Jackie-Brown",
    },
    "David Fincher": {
        "Fight Club": "Fight-Club",
        "Se7en": "Se7en",
        "Social Network, The": "Social-Network,-The",
        "Panic Room": "Panic-Room",
    },
    "James Cameron": {
        "Aliens": "Aliens",
        "Titanic": "Titanic",
        "Avatar": "Avatar",
        "True Lies": "True-Lies",
    },
    "Coen Brothers": {
        "No Country for Old Men": "No-Country-for-Old-Men",
        "Fargo": "Fargo",
        "Big Lebowski, The": "Big-Lebowski,-The",
        "Barton Fink": "Barton-Fink",
        "True Grit": "True-Grit",
        "Blood Simple": "Blood-Simple",
        "Raising Arizona": "Raising-Arizona",
        "Burn After Reading": "Burn-After-Reading",
    },
}

# ─── IMSDB Fetcher ───
IMSDB_BASE = "https://imsdb.com/scripts/{slug}.html"

def fetch_imsdb(slug: str) -> str | None:
    """Fetch screenplay text from IMSDB."""
    url = IMSDB_BASE.format(slug=slug)
    try:
        r = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (educational research)"
        })
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {url}")
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

        print(f"  No script text found at {url}")
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def load_cached(title: str) -> str | None:
    """Load script from local cache."""
    safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    path = CACHE_DIR / f"{safe_name}.txt"
    if path.exists():
        return path.read_text(encoding='utf-8')
    return None


def save_cache(title: str, text: str):
    """Save script to local cache."""
    safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    path = CACHE_DIR / f"{safe_name}.txt"
    path.write_text(text, encoding='utf-8')


def get_script_text(title: str, slug: str | None) -> str | None:
    """Get script text from cache, local file, or IMSDB."""
    # 1. Cache
    text = load_cached(title)
    if text:
        print(f"  Loaded from cache ({len(text):,} chars)")
        return text

    # 2. Local file override
    if title in LOCAL_FILES:
        path = LOCAL_FILES[title]
        if path.exists():
            text = path.read_text(encoding='utf-8')
            save_cache(title, text)
            print(f"  Loaded from local file ({len(text):,} chars)")
            return text

    # 3. IMSDB
    if slug:
        text = fetch_imsdb(slug)
        if text:
            save_cache(title, text)
            print(f"  Fetched from IMSDB ({len(text):,} chars)")
            time.sleep(1.5)
            return text

    return None


# ─── Claude API ───
def call_claude(api_key: str, prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude API with screenplay analysis prompt."""
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    if r.status_code != 200:
        print(f"  API error: {r.status_code} {r.text[:300]}")
        return None
    data = r.json()
    return data["content"][0]["text"]


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


def analyze_script(api_key: str, title: str, text: str) -> dict | None:
    """Send screenplay to Claude for 3-axis analysis."""
    # Truncate if too long (keep ~120K chars to stay well within context)
    if len(text) > 120000:
        # Take first 60K + last 60K to capture setup and payoff
        text = text[:60000] + "\n\n[...MIDDLE SECTION OMITTED FOR LENGTH...]\n\n" + text[-60000:]
        print(f"  (truncated to {len(text):,} chars for API)")

    prompt = ANALYSIS_PROMPT.format(script_text=text)

    print(f"  Sending to Claude API ({len(text):,} chars)...")
    response = call_claude(api_key, prompt)
    if not response:
        return None

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)

        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', clean)
        if json_match:
            result = json.loads(json_match.group())
            result['text_len'] = len(text)
            return result
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON: {e}")
        print(f"  Response preview: {response[:500]}")

    return None


# ─── Main ───
def main():
    # Load API key
    api_key = os.environ["CLAUDE_API_KEY"]

    # Load existing checkpoint
    checkpoint = {}
    if CHECKPOINT.exists():
        checkpoint = json.load(open(CHECKPOINT, encoding='utf-8'))

    # Load influence data for cross-referencing
    inf = json.load(open(BASE / "screenplay" / "influence_data.json", encoding='utf-8'))
    dir_lookup = {d['name']: d for d in inf['directors']}
    # Also add Coen Brothers as alias
    for d in inf['directors']:
        if d['name'] in ('Ethan Coen', 'Joel Coen'):
            dir_lookup['Coen Brothers'] = d

    # Stats
    total = sum(len(films) for films in TARGETS.values())
    fetched = 0
    analyzed = 0
    skipped = 0
    failed_fetch = []
    failed_analysis = []

    print(f"=== LLM Script Quality Analysis ===")
    print(f"Total targets: {total} scripts across {len(TARGETS)} directors")
    print(f"Already in checkpoint: {len(checkpoint)}\n")

    for director, films in TARGETS.items():
        p_val = dir_lookup.get(director, {}).get('p', 999)
        bonf = "***" if p_val < 0.000685 else "*" if p_val < 0.05 else ""
        print(f"\n{'='*60}")
        print(f"  {director} {bonf} (p={p_val:.6f})")
        print(f"{'='*60}")

        for title, slug in films.items():
            key = f"{director}|{title}"

            # Skip if already analyzed
            if key in checkpoint:
                print(f"  [SKIP] {title}")
                skipped += 1
                continue

            print(f"\n  [{title}]")

            # 1. Get script text
            text = get_script_text(title, slug)
            if not text:
                failed_fetch.append((director, title))
                print(f"  FAILED to get script text")
                continue
            fetched += 1

            # 2. Analyze with Claude
            result = analyze_script(api_key, title, text)
            if result:
                checkpoint[key] = {
                    "director": director,
                    "title": title,
                    **result,
                }
                analyzed += 1
                print(f"  -> Density={result.get('info_density')}, "
                      f"Payoff={result.get('payoff_rate')}, "
                      f"Dir/Dial={result.get('direction_pct')}/{result.get('dialogue_pct')}, "
                      f"Balance={result.get('balance_rating')}")

                # Save checkpoint after each success
                with open(CHECKPOINT, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)

                time.sleep(0.5)
            else:
                failed_analysis.append((director, title))
                print(f"  FAILED API analysis")

    # ─── Summary ───
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Fetched: {fetched}")
    print(f"  Analyzed: {analyzed}")
    if failed_fetch:
        print(f"  Failed fetch ({len(failed_fetch)}):")
        for d, t in failed_fetch:
            print(f"    {d} - {t}")
    if failed_analysis:
        print(f"  Failed analysis ({len(failed_analysis)}):")
        for d, t in failed_analysis:
            print(f"    {d} - {t}")

    # ─── Results Table ───
    all_data = checkpoint
    if all_data:
        print(f"\n\n{'='*80}")
        print(f"  RESULTS TABLE")
        print(f"{'='*80}")
        print(f"{'Director':<22} {'Title':<28} {'Density':>7} {'Payoff':>7} {'Dir%':>5} {'Dia%':>5} {'Bal':>4}")
        print("-" * 80)

        # Group by director
        by_director = {}
        for key, data in all_data.items():
            d = data.get('director', '?')
            by_director.setdefault(d, []).append(data)

        for director in TARGETS:
            if director not in by_director:
                continue
            entries = by_director[director]
            for data in sorted(entries, key=lambda x: x.get('title', '')):
                print(f"{data.get('director',''):<22} {data.get('title',''):<28} "
                      f"{data.get('info_density','?'):>7} {data.get('payoff_rate','?'):>7} "
                      f"{data.get('direction_pct','?'):>5} {data.get('dialogue_pct','?'):>5} "
                      f"{data.get('balance_rating','?'):>4}")

            # Director average
            densities = [d['info_density'] for d in entries if 'info_density' in d]
            payoffs = [d['payoff_rate'] for d in entries if 'payoff_rate' in d]
            if densities:
                print(f"{'':22} {'--- AVERAGE ---':<28} "
                      f"{sum(densities)/len(densities):>7.1f} {sum(payoffs)/len(payoffs):>7.1f}")
            print()


if __name__ == "__main__":
    main()
