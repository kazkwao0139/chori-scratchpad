"""기생충 넷플릭스 CC자막(VTT) → 캐릭터별 대사 JSON 파싱"""

import re
import json
from collections import defaultdict, Counter

VTT_PATH = "D:/game-portfolio-main/기생충 대본"
OUT_PATH = "D:/game-portfolio-main/parasite_dialogue.json"

def strip_vtt_tags(line: str) -> str:
    """VTT 태그 제거 → 순수 텍스트"""
    return re.sub(r'<[^>]+>', '', line).strip()

def is_sound_effect(text: str) -> bool:
    """[효과음] 또는 ♪노래♪ 판별"""
    text = text.strip()
    return bool(re.match(r'^\[.*\]$', text)) or (text.startswith('♪') and text.endswith('♪'))

def parse_vtt(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # VTT 블록 분리 (빈 줄 기준)
    blocks = raw.split('\n\n')

    dialogue = defaultdict(list)  # character -> [대사들]
    current_char = None
    stats = Counter()

    for block in blocks:
        lines = block.strip().split('\n')

        # 타임스탬프 포함된 블록만 처리
        timestamp_line = None
        content_lines = []
        for line in lines:
            if '-->' in line:
                timestamp_line = line
            elif timestamp_line is not None and line.strip() and not line.strip().isdigit():
                content_lines.append(line)

        if not content_lines:
            continue

        # 텍스트 추출
        texts = [strip_vtt_tags(l) for l in content_lines]
        texts = [t for t in texts if t]  # 빈 문자열 제거

        for text in texts:
            # 효과음/음악 스킵
            if is_sound_effect(text):
                stats['sound_effects'] += 1
                continue

            # 대사 중간에 끼어있는 [효과음] 제거
            text = re.sub(r'\[.*?\]', '', text).strip()
            if not text:
                continue

            # 패턴 1: 단독 캐릭터 라벨 "(기우)"
            char_only = re.match(r'^\(([^)]+)\)$', text)
            if char_only:
                current_char = char_only.group(1)
                continue

            # 패턴 2: "- (기택) 대사" (동시대사)
            multi = re.match(r'^-\s*\(([^)]+)\)\s*(.+)', text)
            if multi:
                char_name = multi.group(1)
                line_text = multi.group(2).strip()
                if line_text and not is_sound_effect(line_text):
                    dialogue[char_name].append(line_text)
                    stats['lines'] += 1
                current_char = char_name
                continue

            # 패턴 3: "(기우) 대사" (인라인)
            inline = re.match(r'^\(([^)]+)\)\s+(.+)', text)
            if inline:
                char_name = inline.group(1)
                line_text = inline.group(2).strip()
                if line_text and not is_sound_effect(line_text):
                    dialogue[char_name].append(line_text)
                    stats['lines'] += 1
                current_char = char_name
                continue

            # 패턴 4: 캐릭터 이름 없는 대사 → 이전 캐릭터의 연속
            if current_char and not text.startswith('♪'):
                dialogue[current_char].append(text)
                stats['lines'] += 1

    return dict(dialogue), stats

if __name__ == '__main__':
    dialogue, stats = parse_vtt(VTT_PATH)

    # JSON 저장
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print(f"=== 기생충 대사 파싱 완료 ===")
    print(f"총 대사 수: {stats['lines']}")
    print(f"효과음 스킵: {stats['sound_effects']}")
    print(f"캐릭터 수: {len(dialogue)}")
    print()
    print(f"{'캐릭터':<12} {'대사 수':>6}  {'샘플'}")
    print("-" * 60)
    for char, lines in sorted(dialogue.items(), key=lambda x: -len(x[1])):
        sample = lines[0][:40] if lines else ''
        print(f"{char:<12} {len(lines):>6}  \"{sample}...\"")
