"""
LLM Sweet Spot: Language-agnostic screenplay quality metric.
Runs Qwen2.5-3B perplexity on 83 IMSDB screenplays + our 3 works.

Phase 1: Fetch & cache dialogue from IMSDB
Phase 2: LLM perplexity per character (with checkpoint resume)
Phase 3: Variance analysis & cross-language sweet spot
"""

import re
import sys
import time
import json
import math
import os
from pathlib import Path
import urllib.request
from collections import defaultdict
from typing import Dict, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE = str(Path(__file__).resolve().parent.parent)
CACHE_PATH = f"{BASE}/screenplay/imsdb_dialogue_cache.json"
CHECKPOINT_PATH = f"{BASE}/screenplay/llm_sweetspot_checkpoint.json"
RESULT_PATH = f"{BASE}/screenplay/llm_sweetspot_results.json"

MODEL_NAME = "Qwen/Qwen2.5-3B"
MIN_CHARS = 500   # minimum text length per character
MIN_CHARACTERS = 3  # minimum characters per screenplay for variance


# ══════════════════════════════════════════════════════
# Phase 1: IMSDB Fetch & Cache
# ══════════════════════════════════════════════════════

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


def imsdb(slug: str) -> str:
    return f"https://imsdb.com/scripts/{slug}.html"


DATASET = [
    (1950, True, "Sunset Boulevard", imsdb("Sunset-Boulevard")),
    (1950, False, "All About Eve", imsdb("All-About-Eve")),
    (1953, True, "On the Waterfront", imsdb("On-the-Waterfront")),
    (1953, False, "The Barefoot Contessa", imsdb("Barefoot-Contessa,-The")),
    (1960, True, "The Apartment", imsdb("Apartment,-The")),
    (1960, False, "North by Northwest", imsdb("North-by-Northwest")),
    (1967, True, "In the Heat of the Night", imsdb("In-the-Heat-of-the-Night")),
    (1967, False, "Bonnie and Clyde", imsdb("Bonnie-and-Clyde")),
    (1967, False, "The Graduate", imsdb("Graduate,-The")),
    (1969, True, "Butch Cassidy", imsdb("Butch-Cassidy-and-the-Sundance-Kid")),
    (1969, False, "Easy Rider", imsdb("Easy-Rider")),
    (1972, True, "The Godfather", imsdb("Godfather")),
    (1972, False, "Cabaret", imsdb("Cabaret")),
    (1974, True, "Chinatown", imsdb("Chinatown")),
    (1974, False, "The Conversation", imsdb("Conversation,-The")),
    (1976, True, "Network", imsdb("Network")),
    (1976, False, "Rocky", imsdb("Rocky")),
    (1976, False, "Taxi Driver", imsdb("Taxi-Driver")),
    (1977, True, "Annie Hall", imsdb("Annie-Hall")),
    (1977, False, "Star Wars", imsdb("Star-Wars-A-New-Hope")),
    (1979, True, "Kramer vs. Kramer", imsdb("Kramer-vs-Kramer")),
    (1979, False, "Apocalypse Now", imsdb("Apocalypse-Now")),
    (1980, True, "Ordinary People", imsdb("Ordinary-People")),
    (1980, False, "Raging Bull", imsdb("Raging-Bull")),
    (1982, True, "Gandhi", imsdb("Gandhi")),
    (1982, False, "E.T.", imsdb("E-T--the-Extra-Terrestrial")),
    (1982, False, "Tootsie", imsdb("Tootsie")),
    (1984, True, "Amadeus", imsdb("Amadeus")),
    (1984, False, "The Killing Fields", imsdb("Killing-Fields,-The")),
    (1986, True, "Platoon", imsdb("Platoon")),
    (1986, False, "Top Gun", imsdb("Top-Gun")),
    (1988, True, "Rain Man", imsdb("Rain-Man")),
    (1988, False, "Die Hard", imsdb("Die-Hard")),
    (1989, True, "Dead Poets Society", imsdb("Dead-Poets-Society")),
    (1989, False, "When Harry Met Sally", imsdb("When-Harry-Met-Sally")),
    (1990, True, "Dances with Wolves", imsdb("Dances-with-Wolves")),
    (1990, False, "Goodfellas", imsdb("Goodfellas")),
    (1991, True, "Silence of the Lambs", imsdb("Silence-of-the-Lambs,-The")),
    (1991, False, "Thelma and Louise", imsdb("Thelma-and-Louise")),
    (1991, False, "JFK", imsdb("JFK")),
    (1992, True, "Unforgiven", imsdb("Unforgiven")),
    (1992, False, "A Few Good Men", imsdb("A-Few-Good-Men")),
    (1992, False, "Scent of a Woman", imsdb("Scent-of-a-Woman")),
    (1993, True, "Schindler's List", imsdb("Schindler's-List")),
    (1993, False, "The Fugitive", imsdb("Fugitive,-The")),
    (1993, False, "The Piano", imsdb("Piano,-The")),
    (1994, True, "Forrest Gump", imsdb("Forrest-Gump")),
    (1994, False, "Pulp Fiction", imsdb("Pulp-Fiction")),
    (1994, False, "The Shawshank Redemption", imsdb("Shawshank-Redemption,-The")),
    (1995, True, "Braveheart", imsdb("Braveheart")),
    (1995, False, "Sense and Sensibility", imsdb("Sense-and-Sensibility")),
    (1995, False, "Usual Suspects", imsdb("Usual-Suspects,-The")),
    (1996, True, "The English Patient", imsdb("English-Patient,-The")),
    (1996, False, "Fargo", imsdb("Fargo")),
    (1996, False, "Jerry Maguire", imsdb("Jerry-Maguire")),
    (1997, True, "Good Will Hunting", imsdb("Good-Will-Hunting")),
    (1997, False, "As Good as It Gets", imsdb("As-Good-As-It-Gets")),
    (1997, False, "Titanic", imsdb("Titanic")),
    (1998, True, "Shakespeare in Love", imsdb("Shakespeare-in-Love")),
    (1998, False, "Saving Private Ryan", imsdb("Saving-Private-Ryan")),
    (1998, False, "The Truman Show", imsdb("Truman-Show,-The")),
    (1999, True, "American Beauty", imsdb("American-Beauty")),
    (1999, False, "The Sixth Sense", imsdb("Sixth-Sense,-The")),
    (1999, False, "The Green Mile", imsdb("Green-Mile,-The")),
    (2000, True, "Almost Famous", imsdb("Almost-Famous")),
    (2000, False, "Gladiator", imsdb("Gladiator")),
    (2000, False, "Erin Brockovich", imsdb("Erin-Brockovich")),
    (2001, True, "A Beautiful Mind", imsdb("Beautiful-Mind,-A")),
    (2001, False, "Gosford Park", imsdb("Gosford-Park")),
    (2001, False, "Moulin Rouge", imsdb("Moulin-Rouge")),
    (2002, True, "The Pianist", imsdb("Pianist,-The")),
    (2002, False, "Gangs of New York", imsdb("Gangs-of-New-York")),
    (2002, False, "Minority Report", imsdb("Minority-Report")),
    (2003, True, "Lost in Translation", imsdb("Lost-in-Translation")),
    (2003, False, "Finding Nemo", imsdb("Finding-Nemo")),
    (2003, False, "Master and Commander", imsdb("Master-and-Commander")),
    (2004, True, "Eternal Sunshine", imsdb("Eternal-Sunshine-of-the-Spotless-Mind")),
    (2004, False, "Hotel Rwanda", imsdb("Hotel-Rwanda")),
    (2004, False, "The Aviator", imsdb("Aviator,-The")),
    (2005, True, "Crash", imsdb("Crash")),
    (2005, False, "Brokeback Mountain", imsdb("Brokeback-Mountain")),
    (2005, False, "Good Night and Good Luck", imsdb("Good-Night,-and-Good-Luck")),
    (2006, True, "The Departed", imsdb("Departed,-The")),
    (2006, False, "Little Miss Sunshine", imsdb("Little-Miss-Sunshine")),
    (2006, False, "Babel", imsdb("Babel")),
    (2007, True, "Juno", imsdb("Juno")),
    (2007, False, "Michael Clayton", imsdb("Michael-Clayton")),
    (2007, False, "Ratatouille", imsdb("Ratatouille")),
    (2008, True, "Slumdog Millionaire", imsdb("Slumdog-Millionaire")),
    (2008, False, "The Dark Knight", imsdb("Dark-Knight,-The")),
    (2008, False, "Milk", imsdb("Milk")),
    (2009, True, "The Hurt Locker", imsdb("Hurt-Locker,-The")),
    (2009, False, "Inglourious Basterds", imsdb("Inglourious-Basterds")),
    (2009, False, "Up", imsdb("Up")),
    (2010, True, "The King's Speech", imsdb("King's-Speech,-The")),
    (2010, False, "Black Swan", imsdb("Black-Swan")),
    (2010, False, "Inception", imsdb("Inception")),
    (2010, False, "The Fighter", imsdb("Fighter,-The")),
    (2010, False, "The Kids Are All Right", imsdb("Kids-Are-All-Right,-The")),
    (2011, True, "The Artist", imsdb("Artist,-The")),
    (2011, False, "The Descendants", imsdb("Descendants,-The")),
    (2011, False, "Moneyball", imsdb("Moneyball")),
    (2011, False, "Drive", imsdb("Drive")),
    (2012, True, "Argo", imsdb("Argo")),
    (2012, False, "Django Unchained", imsdb("Django-Unchained")),
    (2012, False, "Silver Linings Playbook", imsdb("Silver-Linings-Playbook")),
    (2012, False, "Lincoln", imsdb("Lincoln")),
    (2013, True, "12 Years a Slave", imsdb("12-Years-a-Slave")),
    (2013, False, "The Wolf of Wall Street", imsdb("Wolf-of-Wall-Street,-The")),
    (2013, False, "Her", imsdb("Her")),
    (2014, True, "Birdman", imsdb("Birdman")),
    (2014, False, "Whiplash", imsdb("Whiplash")),
    (2014, False, "Grand Budapest Hotel", imsdb("Grand-Budapest-Hotel,-The")),
    (2014, False, "Interstellar", imsdb("Interstellar")),
    (2015, True, "Spotlight", imsdb("Spotlight")),
    (2015, False, "The Big Short", imsdb("Big-Short,-The")),
    (2015, False, "The Revenant", imsdb("Revenant,-The")),
    (2015, False, "Ex Machina", imsdb("Ex-Machina")),
]


def phase1_fetch():
    """Fetch all IMSDB screenplays and cache dialogue."""
    if os.path.exists(CACHE_PATH):
        print(f"  Cache exists: {CACHE_PATH}")
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("  Fetching from IMSDB...")
    cache = []
    for year, is_winner, title, url in DATASET:
        marker = "W" if is_winner else " "
        print(f"  [{marker}] {year} {title}... ", end='', flush=True)

        html = fetch_script(url)
        if not html:
            print("SKIP (fetch failed)")
            continue
        text = extract_text(html)
        if not text or len(text) < 500:
            print("SKIP (no text)")
            continue
        dialogue = parse_dialogue(text)

        # Filter characters with enough text
        chars = {}
        for char, text in dialogue.items():
            if len(text) >= MIN_CHARS:
                chars[char] = text

        if len(chars) < MIN_CHARACTERS:
            print(f"SKIP ({len(chars)} chars)")
            continue

        cache.append({
            'title': title,
            'year': year,
            'is_winner': is_winner,
            'dialogue': chars,
        })
        print(f"OK ({len(chars)} chars)")
        time.sleep(0.3)

    # Add our 3 works
    # Dark Knight
    with open(f'{BASE}/_copyrighted/dark_knight_dialogue.json', 'r', encoding='utf-8') as f:
        dk = json.load(f)
    dk_chars = {}
    for char, lines in dk.items():
        text = ' '.join(lines) if isinstance(lines, list) else lines
        if len(text) >= MIN_CHARS:
            dk_chars[char] = text
    if len(dk_chars) >= MIN_CHARACTERS:
        cache.append({
            'title': 'The Dark Knight (ours)',
            'year': 2008,
            'is_winner': False,
            'dialogue': dk_chars,
            'is_ours': True,
        })

    # Parasite
    with open(f'{BASE}/_copyrighted/parasite_dialogue.json', 'r', encoding='utf-8') as f:
        p = json.load(f)
    p_chars = {}
    for char, lines in p.items():
        text = ' '.join(lines) if isinstance(lines, list) else lines
        if len(text) >= MIN_CHARS:
            p_chars[char] = text
    if len(p_chars) >= MIN_CHARACTERS:
        cache.append({
            'title': 'Parasite (ours)',
            'year': 2019,
            'is_winner': True,
            'dialogue': p_chars,
            'is_ours': True,
        })

    # Code Geass
    with open(f'{BASE}/_copyrighted/code_geass_dialogue_ja.json', 'r', encoding='utf-8') as f:
        cg = json.load(f)
    cg_chars = {}
    for char, lines in cg.items():
        text = ' '.join(lines) if isinstance(lines, list) else lines
        if len(text) >= MIN_CHARS:
            cg_chars[char] = text
    if len(cg_chars) >= MIN_CHARACTERS:
        cache.append({
            'title': 'Code Geass (ours)',
            'year': 2008,
            'is_winner': False,
            'dialogue': cg_chars,
            'is_ours': True,
        })

    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"\n  Cached {len(cache)} screenplays to {CACHE_PATH}")
    return cache


# ══════════════════════════════════════════════════════
# Phase 2: LLM Perplexity
# ══════════════════════════════════════════════════════

def compute_perplexity(model, tokenizer, text, device, max_length=2048):
    """Compute bits-per-character for a text using the LLM."""
    tokens = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=max_length).to(device)
    input_ids = tokens['input_ids']

    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                     shift_labels.view(-1))

    # Total bits
    total_nats = losses.sum().item()
    total_bits = total_nats / math.log(2)
    n_chars = len(text)

    return total_bits / n_chars if n_chars > 0 else None


def phase2_llm(cache):
    """Run LLM perplexity on all cached screenplays."""
    # Load checkpoint
    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"  Resuming from checkpoint ({len(checkpoint)} screenplays done)")

    # Check what's left
    todo = [s for s in cache if s['title'] not in checkpoint]
    if not todo:
        print("  All screenplays already processed!")
        return checkpoint

    print(f"  {len(todo)} screenplays remaining")

    # Load model
    print(f"  Loading {MODEL_NAME}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    print(f"  Model loaded on {device}")

    for idx, screenplay in enumerate(todo):
        title = screenplay['title']
        chars = screenplay['dialogue']
        print(f"\n  [{idx+1}/{len(todo)}] {title} ({len(chars)} characters)")

        char_bpc = {}
        for char_name, text in chars.items():
            bpc = compute_perplexity(model, tokenizer, text, device)
            if bpc is not None:
                char_bpc[char_name] = round(bpc, 6)
                print(f"    {char_name:>20s}: bpc={bpc:.4f} ({len(text):,d} chars)")

        if len(char_bpc) >= MIN_CHARACTERS:
            bpc_vals = list(char_bpc.values())
            mean_bpc = sum(bpc_vals) / len(bpc_vals)
            var_bpc = sum((b - mean_bpc) ** 2 for b in bpc_vals) / len(bpc_vals)

            checkpoint[title] = {
                'title': title,
                'year': screenplay['year'],
                'is_winner': screenplay['is_winner'],
                'is_ours': screenplay.get('is_ours', False),
                'n_chars': len(char_bpc),
                'bpc_per_char': char_bpc,
                'bpc_mean': round(mean_bpc, 6),
                'bpc_var': round(var_bpc, 6),
                'bpc_std': round(var_bpc ** 0.5, 6),
            }
            print(f"    => mean={mean_bpc:.4f}, var={var_bpc:.6f}, std={var_bpc**0.5:.4f}")
        else:
            print(f"    => SKIP (only {len(char_bpc)} chars with valid bpc)")

        # Save checkpoint after each screenplay
        with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    # Cleanup model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return checkpoint


# ══════════════════════════════════════════════════════
# Phase 3: Analysis
# ══════════════════════════════════════════════════════

def phase3_analysis(results):
    """Analyze LLM sweet spot."""
    print(f"\n{'=' * 70}")
    print("  LLM SWEET SPOT ANALYSIS")
    print(f"{'=' * 70}")

    imsdb = [r for r in results.values() if not r.get('is_ours', False)]
    ours = [r for r in results.values() if r.get('is_ours', False)]

    winners = [r for r in imsdb if r['is_winner']]
    nominees = [r for r in imsdb if not r['is_winner']]

    if not winners or not nominees:
        print("  Not enough data for comparison")
        return

    w_var = [r['bpc_var'] for r in winners]
    n_var = [r['bpc_var'] for r in nominees]
    all_var = sorted(w_var + n_var)

    print(f"\n  IMSDB screenplays: {len(imsdb)} total ({len(winners)} winners, {len(nominees)} nominees)")
    print(f"\n  LLM BPC VARIANCE:")
    print(f"    Winners  (n={len(w_var):2d}): mean={sum(w_var)/len(w_var):.6f}, "
          f"range=[{min(w_var):.6f}, {max(w_var):.6f}]")
    print(f"    Nominees (n={len(n_var):2d}): mean={sum(n_var)/len(n_var):.6f}, "
          f"range=[{min(n_var):.6f}, {max(n_var):.6f}]")

    # IQR
    q1 = all_var[len(all_var) // 4]
    q3 = all_var[3 * len(all_var) // 4]
    print(f"    IQR: [{q1:.6f}, {q3:.6f}]")

    # Winner vs nominee head-to-head
    win_count = 0
    total_match = 0
    years = set(r['year'] for r in imsdb)
    for year in sorted(years):
        w = [r for r in winners if r['year'] == year]
        n = [r for r in nominees if r['year'] == year]
        if w and n:
            total_match += 1
            w_v = w[0]['bpc_var']
            n_avg = sum(r['bpc_var'] for r in n) / len(n)
            if w_v > n_avg:
                win_count += 1

    if total_match > 0:
        print(f"\n  Winner vs nominee (var > avg_nom):")
        print(f"    {win_count}/{total_match} = {win_count/total_match*100:.1f}%")

    # Our works
    print(f"\n  OUR WORKS:")
    for r in ours:
        in_iqr = "IN" if q1 <= r['bpc_var'] <= q3 else "OUT"
        print(f"    {r['title']:30s} var={r['bpc_var']:.6f} [{in_iqr}]  "
              f"(mean_bpc={r['bpc_mean']:.4f}, {r['n_chars']} chars)")

    # Full ranking
    print(f"\n{'=' * 70}")
    print("  LLM BPC VAR RANKING")
    print(f"{'=' * 70}")

    all_ranked = []
    for r in imsdb:
        marker = "W" if r['is_winner'] else " "
        all_ranked.append((r['bpc_var'], f"[{marker}] {r['title']}", False))
    for r in ours:
        all_ranked.append((r['bpc_var'], f">>> {r['title']}", True))

    all_ranked.sort(key=lambda x: x[0])
    for i, (var, name, is_ours) in enumerate(all_ranked):
        flag = " <<<<" if is_ours else ""
        print(f"  {i+1:3d}. var={var:.6f}  {name}{flag}")

    # Save final results
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'imsdb_winners': [{k: v for k, v in r.items() if k != 'bpc_per_char'}
                              for r in winners],
            'imsdb_nominees': [{k: v for k, v in r.items() if k != 'bpc_per_char'}
                               for r in nominees],
            'our_works': ours,
            'stats': {
                'winner_mean_var': round(sum(w_var)/len(w_var), 6),
                'nominee_mean_var': round(sum(n_var)/len(n_var), 6),
                'iqr': [round(q1, 6), round(q3, 6)],
                'winner_win_rate': f"{win_count}/{total_match}",
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {RESULT_PATH}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  LLM SWEET SPOT — Qwen2.5-3B perplexity")
    print("  83 IMSDB screenplays + Parasite + Code Geass + Dark Knight")
    print("=" * 70)

    t0 = time.time()

    # Phase 1
    print(f"\n{'=' * 70}")
    print("  PHASE 1: Fetch & Cache")
    print(f"{'=' * 70}")
    cache = phase1_fetch()

    # Phase 2
    print(f"\n{'=' * 70}")
    print("  PHASE 2: LLM Perplexity ({} screenplays)".format(len(cache)))
    print(f"{'=' * 70}")
    results = phase2_llm(cache)

    # Phase 3
    phase3_analysis(results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
