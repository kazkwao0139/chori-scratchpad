"""
The Dark Knight screenplay parser
- OCR'd screenplay from archive.org
- Character names in ALL CAPS on standalone lines
- Dialogue follows after blank line
- Handle (CONT'D), (O.S.), (V.O.) parentheticals
- Name normalization: WAYNE→BRUCE WAYNE, DENT→HARVEY DENT, etc.
"""

import re
import json
from collections import defaultdict, Counter

SCRIPT_PATH = "D:/game-portfolio-main/SCRATCHPAD/hello_world/dark_knight/dark_knight_script.txt"
OUT_PATH = "D:/game-portfolio-main/SCRATCHPAD/hello_world/dark_knight/dark_knight_dialogue.json"
DETAIL_PATH = "D:/game-portfolio-main/SCRATCHPAD/hello_world/dark_knight/dark_knight_dialogue_detail.json"

# Scene heading pattern
SCENE_RE = re.compile(r'^(INT\.|EXT\.|CUT TO|CUT$|FADE|DISSOLVE|SMASH|CONTINUED|OMITTED)')

# Character name pattern: ALL CAPS, possibly with parenthetical
CHAR_RE = re.compile(r'^([A-Z][A-Z\s.\-\'\"]+?)(\s*\((?:CONT\'?D?|O\.S\.|V\.O\.|INTO PHONE|ON PHONE|ON TV|PRE-?LAP|OVER|FILTERED)\.?\))?\s*$')

# Stage direction indicators (lines that look like character names but aren't)
NOT_CHARACTERS = {
    'THE', 'DARK', 'KNIGHT', 'CUT', 'CUT TO', 'FADE IN', 'FADE OUT',
    'CONTINUED', 'CONTINUOUS', 'OMITTED', 'BURNING', 'DAYLIGHT',
    'CLOSE ON', 'ANGLE ON', 'END', 'SMASH CUT', 'BLACK',
    'LATER', 'NIGHT', 'DAY', 'DAWN', 'DUSK', 'MORNING',
    'RESUME', 'BACK TO', 'TITLE', 'SUPER', 'INTERCUT',
    'TIME CUT', 'MATCH CUT', 'CREDITS', 'THE END',
}

# Name aliases → canonical name
NAME_ALIASES = {
    'WAYNE': 'WAYNE',
    'BATMAN': 'BATMAN',
    '"BATMAN"': '"BATMAN"',
    'DENT': 'DENT',
    'THE JOKER': 'THE JOKER',
    'GORDON': 'GORDON',
    'RACHEL': 'RACHEL',
    'ALFRED': 'ALFRED',
    'FOX': 'FOX',
    'MARONI': 'MARONI',
    'CHECHEN': 'CHECHEN',
    'LAU': 'LAU',
    'RAMIREZ': 'RAMIREZ',
    'ENGEL': 'ENGEL',
    'REESE': 'REESE',
    'STEPHENS': 'STEPHENS',
    'MAYOR': 'MAYOR',
    'LOEB': 'LOEB',
    'SCARECROW': 'SCARECROW',
    'GAMBOL': 'GAMBOL',
    'GRUMPY': 'GRUMPY',
    'WUERTZ': 'WUERTZ',
    'BARBARA': 'BARBARA',
    'NATASCHA': 'NATASCHA',
    'JAMES': 'JAMES',
    'BANK MANAGER': 'BANK MANAGER',
    'SHOTGUN SWAT': 'SHOTGUN SWAT',
}

# Minimum line count for "main character"
MIN_LINES_MAIN = 10

# Character names that appear as 3rd-person subjects in stage directions
CHAR_SUBJECTS = [
    'Batman', 'Wayne', 'Gordon', 'Dent', 'Joker', 'Alfred', 'Fox', 'Rachel',
    'Ramirez', 'Maroni', 'Chechen', 'Lau', 'Scarecrow', 'Gambol', 'Stephens',
    'Lucius', 'Harvey', 'Barbara', 'Reese', 'Wuertz', 'Loeb', 'Surrillo',
]

# Action verbs that indicate stage directions
STAGE_VERBS = (
    'looks', 'turns', 'moves', 'gets', 'pulls', 'rises', 'walks', 'stands',
    'steps', 'grabs', 'reaches', 'opens', 'closes', 'picks', 'puts', 'takes',
    'watches', 'stares', 'crouches', 'climbs', 'runs', 'drives', 'fires',
    'hits', 'kicks', 'throws', 'drops', 'falls', 'sits', 'spins', 'slides',
    'rolls', 'ducks', 'jumps', 'enters', 'exits', 'nods', 'shakes', 'waves',
    'points', 'aims', 'checks', 'hands', 'holds', 'sets', 'peers', 'leans',
    'presses', 'bites', 'examines', 'catches', 'slams', 'rips', 'smashes',
)


def is_stage_direction_text(text):
    """Check if a collected dialogue block is actually a stage direction"""
    text = text.strip()
    if not text:
        return False

    # INSERT CUT / SMASH CUT etc.
    if re.match(r'^INSERT|^SMASH|^GORDON\s+(STANDS|TAKES|ON THE)', text, re.IGNORECASE):
        return True

    # 3rd person: "Batman moves...", "Gordon waves...", "He turns..."
    for subj in CHAR_SUBJECTS + ['He', 'She', 'They', 'His', 'Her', 'The', 'As the', 'As he', 'As she']:
        pattern = rf'^{re.escape(subj)}\s+(\w+)'
        m = re.match(pattern, text)
        if m:
            verb = m.group(1).lower().rstrip('s')
            # Check if the verb is a stage direction verb
            if verb in STAGE_VERBS or verb + 's' in STAGE_VERBS:
                return True
            # "The [Noun] [verb]" pattern
            if subj == 'The' and re.match(r'^The\s+[A-Z]\w+\s+\w+s?\b', text):
                next_word = text.split()[2].lower().rstrip('s')
                if next_word in STAGE_VERBS or next_word + 's' in STAGE_VERBS:
                    return True

    # All-caps lines embedded in dialogue are usually directions
    if text == text.upper() and len(text) > 25 and not text.endswith('?') and not text.endswith('!'):
        return True

    return False


def is_stage_direction(line):
    """Check if a line is a stage direction, not dialogue"""
    line = line.strip()
    if SCENE_RE.match(line):
        return True
    if line in NOT_CHARACTERS:
        return True
    return False


def parse_screenplay(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogue = defaultdict(list)       # character -> [dialogue strings]
    detail = []                        # [{character, line, line_num}]
    current_char = None
    current_dialogue = []
    stats = Counter()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            # If we were collecting dialogue, save it
            if current_char and current_dialogue:
                full_line = ' '.join(current_dialogue).strip()
                # Clean OCR artifacts
                full_line = re.sub(r'\s+', ' ', full_line)
                if full_line and len(full_line) > 1:
                    # Filter stage directions
                    if is_stage_direction_text(full_line):
                        stats['stage_filtered'] += 1
                    else:
                        dialogue[current_char].append(full_line)
                        detail.append({
                            'character': current_char,
                            'line': full_line,
                            'line_num': i
                        })
                        stats['lines'] += 1
                current_dialogue = []
            i += 1
            continue

        # Check for scene headings
        if SCENE_RE.match(stripped):
            if current_char and current_dialogue:
                full_line = ' '.join(current_dialogue).strip()
                full_line = re.sub(r'\s+', ' ', full_line)
                if full_line and len(full_line) > 1:
                    if is_stage_direction_text(full_line):
                        stats['stage_filtered'] += 1
                    else:
                        dialogue[current_char].append(full_line)
                        detail.append({
                            'character': current_char,
                            'line': full_line,
                            'line_num': i
                        })
                        stats['lines'] += 1
                current_dialogue = []
            current_char = None
            i += 1
            continue

        # Check for character name
        m = CHAR_RE.match(stripped)
        if m:
            name = m.group(1).strip()

            # Skip non-character entries
            if name in NOT_CHARACTERS:
                i += 1
                continue

            # Skip very short/long names
            if len(name) < 2 or len(name) > 30:
                i += 1
                continue

            # Save previous dialogue if any
            if current_char and current_dialogue:
                full_line = ' '.join(current_dialogue).strip()
                full_line = re.sub(r'\s+', ' ', full_line)
                if full_line and len(full_line) > 1:
                    if is_stage_direction_text(full_line):
                        stats['stage_filtered'] += 1
                    else:
                        dialogue[current_char].append(full_line)
                        detail.append({
                            'character': current_char,
                            'line': full_line,
                            'line_num': i
                        })
                        stats['lines'] += 1
                current_dialogue = []

            # Normalize name
            canonical = NAME_ALIASES.get(name, name)
            current_char = canonical
            stats['char_labels'] += 1
            i += 1
            continue

        # Check for parenthetical stage direction within dialogue
        if stripped.startswith('(') and stripped.endswith(')'):
            # This is a parenthetical like "(touches a button)" - skip
            i += 1
            continue

        # If we have a current character, this is dialogue
        if current_char:
            # Check if this looks like a stage direction embedded in dialogue area
            if re.match(r'^[A-Z][a-z].*\b(FIRES|TURNS|MOVES|GRABS|PULLS|PUSHES|SLAMS|RUNS|WALKS|LOOKS|FALLS|RISES|STEPS)\b', stripped):
                # Likely stage direction - end current dialogue
                if current_dialogue:
                    full_line = ' '.join(current_dialogue).strip()
                    full_line = re.sub(r'\s+', ' ', full_line)
                    if full_line and len(full_line) > 1:
                        if is_stage_direction_text(full_line):
                            stats['stage_filtered'] += 1
                        else:
                            dialogue[current_char].append(full_line)
                            detail.append({
                                'character': current_char,
                                'line': full_line,
                                'line_num': i
                            })
                            stats['lines'] += 1
                    current_dialogue = []
                current_char = None
            else:
                current_dialogue.append(stripped)

        i += 1

    # Save last dialogue
    if current_char and current_dialogue:
        full_line = ' '.join(current_dialogue).strip()
        full_line = re.sub(r'\s+', ' ', full_line)
        if full_line and len(full_line) > 1:
            if is_stage_direction_text(full_line):
                stats['stage_filtered'] += 1
            else:
                dialogue[current_char].append(full_line)
                detail.append({
                    'character': current_char,
                    'line': full_line,
                    'line_num': i
                })
                stats['lines'] += 1

    return dict(dialogue), detail, stats


if __name__ == '__main__':
    dialogue, detail, stats = parse_screenplay(SCRIPT_PATH)

    # Save JSON
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=2)

    with open(DETAIL_PATH, 'w', encoding='utf-8') as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    print("=== The Dark Knight Dialogue Parsing ===")
    print(f"Total lines: {stats['lines']}")
    print(f"Stage directions filtered: {stats['stage_filtered']}")
    print(f"Character labels found: {stats['char_labels']}")
    print(f"Unique characters: {len(dialogue)}")
    print()

    total = stats['lines']
    print(f"{'Character':<20} {'Lines':>6} {'%':>6}  Sample")
    print("-" * 80)
    for char, lines in sorted(dialogue.items(), key=lambda x: -len(x[1])):
        pct = len(lines) / total * 100 if total > 0 else 0
        sample = lines[0][:50] if lines else ''
        print(f"{char:<20} {len(lines):>6} {pct:>5.1f}%  \"{sample}...\"")
