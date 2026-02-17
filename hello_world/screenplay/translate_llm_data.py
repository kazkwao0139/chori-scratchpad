"""Translate LLM script analysis data to Korean for DIRECTOR_INFLUENCE.md"""
import json, os, sys, time, re, requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
CHECKPOINT = BASE / "screenplay" / "llm_script_quality_checkpoint.json"
API_KEY_FILE = None  # use os.environ["CLAUDE_API_KEY"]
OUTPUT = BASE / "screenplay" / "_llm_data_section_ko.md"
TRANS_CACHE = BASE / "screenplay" / "_translation_cache.json"

api_key = os.environ["CLAUDE_API_KEY"]
cp = json.load(open(CHECKPOINT, encoding='utf-8'))

# Load translation cache
cache = {}
if TRANS_CACHE.exists():
    cache = json.load(open(TRANS_CACHE, encoding='utf-8'))

dir_order = ['Stanley Kubrick', 'Alfred Hitchcock', 'Christopher Nolan', 'Martin Scorsese',
             'Steven Spielberg', 'Quentin Tarantino', 'David Fincher', 'James Cameron', 'Coen Brothers']

by_dir = {}
for key, data in cp.items():
    d = data['director']
    by_dir.setdefault(d, []).append(data)


def call_claude(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=180,
            )
            if r.status_code != 200:
                print(f"  API error: {r.status_code}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(3)
                    continue
                return None
            return r.json()["content"][0]["text"]
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5)
            else:
                return None


TRANSLATE_PROMPT = """Translate this screenplay analysis to Korean. Keep movie titles, character names, and technical terms (setup, payoff, Density, Payoff Rate) in English. Write concisely. Output ONLY the translation with the exact same field labels.

{text}"""


def translate_film(e):
    """Translate one film's analysis. Returns dict with Korean text."""
    key = f"{e['director']}|{e['title']}"
    if key in cache:
        return cache[key]

    text = f"""Density: {e.get('info_density_reasoning', '')}
Payoff setups: {e.get('payoff_setups', '')}
Payoff: {e.get('payoff_reasoning', '')}
Balance: {e.get('balance_reasoning', '')}
Craft: {e.get('overall_craft_notes', '')}"""

    translated = call_claude(TRANSLATE_PROMPT.format(text=text))
    if not translated:
        translated = text  # fallback to English

    result = {"translated": translated}
    cache[key] = result
    # Save cache after each translation
    with open(TRANS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return result


def parse_translated(text):
    """Parse translated text into labeled sections."""
    sections = {}
    current_label = None
    current_text = []

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        matched = False
        for label in ['Density', 'Payoff setups', 'Payoff', 'Balance', 'Craft']:
            # Match "Density:", "Density :", "**Density**:" etc
            pattern = rf'^[\*]*{re.escape(label)}[\*]*\s*:\s*(.*)'
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                if current_label:
                    sections[current_label] = ' '.join(current_text)
                current_label = label
                current_text = [m.group(1).strip()] if m.group(1).strip() else []
                matched = True
                break

        if not matched and current_label:
            current_text.append(line)

    if current_label:
        sections[current_label] = ' '.join(current_text)

    return sections


# Generate markdown
lines = []
total_films = sum(len(by_dir.get(d, [])) for d in dir_order)
done = 0

for director in dir_order:
    entries = by_dir.get(director, [])
    if not entries:
        continue

    densities = [e['info_density'] for e in entries]
    payoffs = [e['payoff_rate'] for e in entries]
    dir_pcts = [e['direction_pct'] for e in entries]

    lines.append(f"### {director} (n={len(entries)}, avg density={sum(densities)/len(densities):.0f}, avg payoff={sum(payoffs)/len(payoffs):.0f}, avg dir%={sum(dir_pcts)/len(dir_pcts):.0f})")
    lines.append("")

    for e in sorted(entries, key=lambda x: x['title']):
        done += 1
        print(f"[{done}/{total_films}] {director}: {e['title']}...", end=" ")

        result = translate_film(e)
        sections = parse_translated(result["translated"])

        lines.append(f"#### {e['title']} — D={e['info_density']} / P={e['payoff_rate']} / Dir:Dia={e['direction_pct']}:{e['dialogue_pct']}")
        lines.append("")

        for label in ['Density', 'Payoff setups', 'Payoff', 'Balance', 'Craft']:
            text = sections.get(label, '')
            if text:
                if label == 'Density':
                    lines.append(f"> **{label}**: {text}")
                else:
                    lines.append(f">")
                    lines.append(f"> **{label}**: {text}")

        lines.append("")
        print("OK")
        time.sleep(0.3)

    lines.append("---")
    lines.append("")

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nDone. {done} films translated, {len(lines)} lines written to {OUTPUT}")
