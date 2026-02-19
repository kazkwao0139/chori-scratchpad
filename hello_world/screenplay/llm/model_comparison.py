#!/usr/bin/env python
"""
LLM Model Comparison: Same prompt, same scripts, different models.
Models: Claude Opus, GPT-5.2, Gemini Pro
Runs one model at a time via --model flag.
"""

import json, os, time, re, sys, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE / "_copyrighted" / "_script_cache"
OUT_DIR = BASE / "screenplay" / "llm"

# ─── Model configs ───
MODELS = {
    "opus": {
        "api": "anthropic",
        "model_id": "claude-opus-4-6",
        "env_key": "CLAUDE_API_KEY",
    },
    "sonnet": {
        "api": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "env_key": "CLAUDE_API_KEY",
    },
    "gpt": {
        "api": "openai",
        "model_id": "gpt-5.2",
        "env_key": "GPT_API_KEY",
    },
    "gemini": {
        "api": "google",
        "model_id": "gemini-2.5-pro",
        "env_key": "GEMINI_API_KEY",
    },
}

# ─── 41 films (same as original analysis) ───
FILMS = {
    "Stanley Kubrick": ["2001_A_Space_Odyssey", "Shining_The", "Barry_Lyndon"],
    "Alfred Hitchcock": ["Psycho", "Rear_Window"],
    "Christopher Nolan": ["The_Dark_Knight", "Inception", "Interstellar", "Memento"],
    "Martin Scorsese": ["Departed_The", "Taxi_Driver", "Wolf_of_Wall_Street_The", "Casino", "Raging_Bull", "Gangs_of_New_York"],
    "Steven Spielberg": ["Schindlers_List", "Saving_Private_Ryan", "Jurassic_Park", "Jaws", "Minority_Report"],
    "Quentin Tarantino": ["Pulp_Fiction", "Django_Unchained", "Inglourious_Basterds", "Reservoir_Dogs", "Jackie_Brown"],
    "David Fincher": ["Fight_Club", "Se7en", "Social_Network_The", "Panic_Room"],
    "James Cameron": ["Aliens", "Titanic", "Avatar", "True_Lies"],
    "Coen Brothers": ["No_Country_for_Old_Men", "Fargo", "Big_Lebowski_The", "Barton_Fink", "True_Grit", "Blood_Simple", "Raising_Arizona", "Burn_After_Reading"],
}

# ─── Bonus films (번외) ───
BONUS_FILMS = {
    "Bong Joon-ho": ["Parasite"],
    "92nd Academy Awards": ["1917", "Ford_v_Ferrari", "The_Irishman", "Jojo_Rabbit", "Joker", "Little_Women_2019", "Marriage_Story"],
    "Hayao Miyazaki": ["Spirited_Away"],
    "Daniel Kwan & Daniel Scheinert": ["Everything_Everywhere_All_At_Once"],
}

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


def call_anthropic(api_key, model_id, prompt):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=600,
    )
    if r.status_code != 200:
        print(f"  API error: {r.status_code} {r.text[:300]}")
        return None
    return r.json()["content"][0]["text"]


def call_openai(api_key, model_id, prompt):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-type": "application/json",
        },
        json={
            "model": model_id,
            "max_completion_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=600,
    )
    if r.status_code != 200:
        print(f"  API error: {r.status_code} {r.text[:300]}")
        return None
    return r.json()["choices"][0]["message"]["content"]


def call_google(api_key, model_id, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096},
        },
        timeout=600,
    )
    if r.status_code != 200:
        print(f"  API error: {r.status_code} {r.text[:300]}")
        return None
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_model(model_name, api_key, prompt):
    cfg = MODELS[model_name]
    if cfg["api"] == "anthropic":
        return call_anthropic(api_key, cfg["model_id"], prompt)
    elif cfg["api"] == "openai":
        return call_openai(api_key, cfg["model_id"], prompt)
    elif cfg["api"] == "google":
        return call_google(api_key, cfg["model_id"], prompt)


def parse_response(response):
    if not response:
        return None
    clean = response.strip()
    if clean.startswith("```"):
        clean = re.sub(r'^```\w*\n?', '', clean)
        clean = re.sub(r'\n?```$', '', clean)
    json_match = re.search(r'\{[\s\S]*\}', clean)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Response preview: {response[:500]}")
    return None


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print(f"Usage: python model_comparison.py <{'|'.join(MODELS.keys())}>")
        sys.exit(1)

    model_name = sys.argv[1]
    cfg = MODELS[model_name]
    checkpoint_file = OUT_DIR / f"comparison_{model_name}_checkpoint.json"

    # Load API key from env or file
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        key_names = {
            "CLAUDE_API_KEY": "claude apikey.txt",
            "GPT_API_KEY": "gpt apikey.txt",
            "GEMINI_API_KEY": "gemini apikey.txt",
        }
        key_file = BASE.parent / key_names.get(cfg["env_key"], "")
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            print(f"No API key found for {cfg['env_key']}")
            sys.exit(1)

    # Load checkpoint
    checkpoint = {}
    if checkpoint_file.exists():
        checkpoint = json.load(open(checkpoint_file, encoding='utf-8'))

    total = sum(len(films) for films in FILMS.values())
    done = len(checkpoint)
    print(f"=== Model Comparison: {model_name} ({cfg['model_id']}) ===")
    print(f"Films: {total}, Already done: {done}\n")

    for director, films in FILMS.items():
        print(f"\n{'='*60}")
        print(f"  {director}")
        print(f"{'='*60}")

        for cache_name in films:
            key = f"{director}|{cache_name}"
            if key in checkpoint:
                print(f"  [SKIP] {cache_name}")
                continue

            # Load from cache
            script_file = CACHE_DIR / f"{cache_name}.txt"
            if not script_file.exists():
                print(f"  [MISSING] {cache_name}")
                continue

            text = script_file.read_text(encoding='utf-8')

            # Truncate if needed
            if len(text) > 120000:
                text = text[:60000] + "\n\n[...MIDDLE SECTION OMITTED FOR LENGTH...]\n\n" + text[-60000:]

            prompt = ANALYSIS_PROMPT.format(script_text=text)
            print(f"  [{cache_name}] ({len(text):,} chars)...", end=" ", flush=True)

            response = call_model(model_name, api_key, prompt)
            result = parse_response(response)

            if result:
                checkpoint[key] = {
                    "director": director,
                    "title": cache_name,
                    "model": model_name,
                    "model_id": cfg["model_id"],
                    **result,
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                print(f"D={result.get('info_density')} P={result.get('payoff_rate')} Dir={result.get('direction_pct')}%")
                time.sleep(0.5)
            else:
                print("FAILED")
                time.sleep(2)

    print(f"\n=== Done: {len(checkpoint)}/{total} films ===")
    print(f"Saved to: {checkpoint_file}")


if __name__ == "__main__":
    main()
