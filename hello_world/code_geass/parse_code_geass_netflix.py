"""Code Geass Netflix 일본어 자막 파싱
SRT 파일에서 （캐릭터명）대사 패턴을 추출하여
캐릭터별 일본어 대사 JSON을 생성한다.

Netflix 자막 포맷:
  （ルルーシュ）９分で済む       ← 캐릭터+대사 같은 줄
  （ダールトン）\n감사합니다      ← 캐릭터명만, 대사는 다음 줄
  （心臓の鼓動）                 ← 효과음 (제외)
  ♪～                           ← 노래 (제외)
"""

import json
import re
import os

# ── 설정 ──
S1_DIR = "D:/game-portfolio-main/code geass/netflix_s1/Code Geass Season 1 Netflix Rip (Timed to Coalgirls 1080p)"
S2_DIR = "D:/game-portfolio-main/code geass/netflix_s2/Code Geass Season 2 (Netflix Rip timed to Coalgirls 1080p)"

# ── 효과음/비대사 키워드 (제외 대상) ──
SFX_KEYWORDS = {
    '鼓動', '音', '爆発', 'アラーム', '通信', '銃声', '衝撃',
    'ドア', 'エンジン', '足音', '拍手', '笑い声', '悲鳴',
    '無線', '電話', 'ブザー', 'サイレン', 'ノイズ', '静寂',
    'モニター', 'スピーカー', 'チャイム', 'ベル', '機械',
    '風', '雨', '雷', '水', '叫び', '泣き声', '歓声',
}

# ── 캐릭터 이름 정규화 ──
NAME_ALIASES = {
    # Lelouch
    'ルルーシュ': 'ルルーシュ',
    'ルルーシュ･ランペルージ': 'ルルーシュ',
    'ルルーシュ・ランペルージ': 'ルルーシュ',
    'ルルーシュ･ヴィ･ブリタニア': 'ルルーシュ',
    'ルルーシュ・ヴィ・ブリタニア': 'ルルーシュ',
    'ゼロ': 'ルルーシュ',
    # Suzaku
    'スザク': 'スザク',
    '枢木スザク': 'スザク',
    '枢木(くるるぎ)スザク': 'スザク',
    'スザクたち': 'スザク',
    '枢木(くるるぎ)スザクたち': 'スザク',
    # C.C.
    'Ｃ.Ｃ.': 'C.C.',
    'Ｃ．Ｃ．': 'C.C.',
    'C.C.': 'C.C.',
    'Ｃ.Ｃ.(シーツー)': 'C.C.',
    'シーツー': 'C.C.',
    # Kallen
    'カレン': 'カレン',
    '紅月カレン': 'カレン',
    'カレン･シュタットフェルト': 'カレン',
    'カレン・シュタットフェルト': 'カレン',
    # Nunnally
    'ナナリー': 'ナナリー',
    'ナナリー･ランペルージ': 'ナナリー',
    'ナナリー・ランペルージ': 'ナナリー',
    'ナナリー･ヴィ･ブリタニア': 'ナナリー',
    # Shirley
    'シャーリー': 'シャーリー',
    'シャーリー･フェネット': 'シャーリー',
    # Cornelia
    'コーネリア': 'コーネリア',
    'コーネリア･リ･ブリタニア': 'コーネリア',
    # Euphemia
    'ユーフェミア': 'ユーフェミア',
    'ユーフェミア･リ･ブリタニア': 'ユーフェミア',
    'ユフィ': 'ユーフェミア',
    # Schneizel
    'シュナイゼル': 'シュナイゼル',
    'シュナイゼル･エル･ブリタニア': 'シュナイゼル',
    # Ohgi
    '扇': '扇',
    '扇要': '扇',
    '扇(おうぎ)要': '扇',
    # Villetta
    'ヴィレッタ': 'ヴィレッタ',
    'ヴィレッタ･ヌゥ': 'ヴィレッタ',
    # Diethard
    'ディートハルト': 'ディートハルト',
    'ディートハルト･リート': 'ディートハルト',
    # Rivalz
    'リヴァル': 'リヴァル',
    'リヴァル･カルデモンド': 'リヴァル',
    # Milly
    'ミレイ': 'ミレイ',
    'ミレイ･アッシュフォード': 'ミレイ',
    # Lloyd
    'ロイド': 'ロイド',
    'ロイド･アスプルンド': 'ロイド',
    # Cecile
    'セシル': 'セシル',
    'セシル･クルーミー': 'セシル',
    # Jeremiah
    'ジェレミア': 'ジェレミア',
    'ジェレミア･ゴットバルト': 'ジェレミア',
    'オレンジ': 'ジェレミア',
    # Tamaki
    '玉城': '玉城',
    '玉城真一郎': '玉城',
    # Tohdoh
    '藤堂': '藤堂',
    '藤堂鏡志朗': '藤堂',
    # Xingke
    '星刻': '星刻',
    '黎星刻': '星刻',
    # Charles
    'シャルル': 'シャルル',
    'シャルル･ジ･ブリタニア': 'シャルル',
    '皇帝': 'シャルル',
    # Marianne
    'マリアンヌ': 'マリアンヌ',
    'マリアンヌ･ヴィ･ブリタニア': 'マリアンヌ',
    # Rolo
    'ロロ': 'ロロ',
    'ロロ･ランペルージ': 'ロロ',
    # Anya
    'アーニャ': 'アーニャ',
    'アーニャ･アールストレイム': 'アーニャ',
    # Gino
    'ジノ': 'ジノ',
    'ジノ･ヴァインベルグ': 'ジノ',
    # Others
    'クロヴィス': 'クロヴィス',
    'クロヴィス･ラ･ブリタニア': 'クロヴィス',
    'バトレー': 'バトレー',
    'バトレー･アスプリウス': 'バトレー',
    'ギルフォード': 'ギルフォード',
    'ダールトン': 'ダールトン',
    'アンドレアス･ダールトン': 'ダールトン',
    'ラクシャータ': 'ラクシャータ',
    '篠崎咲世子': '咲世子',
    '咲世子': '咲世子',
    'マオ': 'マオ',
    'Ｖ.Ｖ.': 'V.V.',
    'V.V.': 'V.V.',
    'Ｖ．Ｖ．': 'V.V.',
    '天子': '天子',
    '神楽耶': '神楽耶',
    '千葉': '千葉',
    '朝比奈': '朝比奈',
    '仙波': '仙波',
    '卜部': '卜部',
    '卜部(うらべ)巧雪': '卜部',
    '卜部巧雪': '卜部',
    'カノン': 'カノン',
    'ニーナ': 'ニーナ',
    'ニーナ･アインシュタイン': 'ニーナ',
    'ビスマルク': 'ビスマルク',
    '南': '南',
    '杉山': '杉山',
}


def is_sfx(text: str) -> bool:
    """효과음인지 판별"""
    for kw in SFX_KEYWORDS:
        if kw in text:
            return True
    return False


def is_song(text: str) -> bool:
    """노래인지 판별"""
    return text.startswith('♪') or text.startswith('♬')


def normalize_name(raw: str) -> str:
    """캐릭터 이름 정규화"""
    name = raw.strip()
    # 읽기 도움말 제거: 枢木(くるるぎ)スザク → 枢木スザク
    name = re.sub(r'\(([ぁ-ん]+)\)', '', name)
    name = re.sub(r'（([ぁ-ん]+)）', '', name)
    # 복수형 '~たち' 제거
    name = re.sub(r'たち$', '', name)
    name = name.strip()

    if name in NAME_ALIASES:
        return NAME_ALIASES[name]
    return name


def parse_srt_file(filepath: str) -> list:
    """SRT 파일에서 (character, dialogue, timestamp) 추출

    Returns: [(char_name, dialogue_text, start_time, end_time), ...]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # HTML-like 태그 제거 (일부 에피소드에 <c.Japanese>...</c.Japanese> 존재)
    content = re.sub(r'<c\.Japanese>', '', content)
    content = re.sub(r'</c\.Japanese>', '', content)
    content = re.sub(r'<[^>]+>', '', content)  # 기타 HTML 태그도 제거

    # SRT 블록 파싱
    blocks = re.split(r'\n\s*\n', content.strip())
    results = []
    current_char = None

    # 캐릭터명 패턴: （...）
    char_pattern = re.compile(r'（([^）]+)）(.*)')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # 첫 줄: 번호, 둘째 줄: 타임스탬프
        try:
            time_match = re.match(
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                lines[1]
            )
            if not time_match:
                continue
            start_time = time_match.group(1)
            end_time = time_match.group(2)
        except (IndexError, ValueError):
            continue

        # 텍스트 줄 합치기
        text_lines = lines[2:]
        if not text_lines:
            continue

        full_text = '\n'.join(text_lines)

        # 노래 제외
        if is_song(full_text.strip()):
            continue

        # 캐릭터명 추출
        first_line = text_lines[0].strip()
        m = char_pattern.match(first_line)

        if m:
            raw_name = m.group(1).strip()
            rest_text = m.group(2).strip()

            # 효과음 체크
            if is_sfx(raw_name):
                # 효과음 뒤에 캐릭터 대사가 있을 수 있음
                # 예: （アラーム）\n（マスター）うっ
                # 다음 줄 체크
                for tl in text_lines[1:]:
                    m2 = char_pattern.match(tl.strip())
                    if m2:
                        raw2 = m2.group(1).strip()
                        rest2 = m2.group(2).strip()
                        if not is_sfx(raw2):
                            char_name = normalize_name(raw2)
                            if rest2:
                                results.append((char_name, rest2, start_time, end_time))
                                current_char = char_name
                continue

            char_name = normalize_name(raw_name)
            current_char = char_name

            # 같은 줄에 대사가 있는 경우
            if rest_text:
                dialogue = rest_text
                # 다음 줄도 이어지는 대사
                for tl in text_lines[1:]:
                    tl = tl.strip()
                    m2 = char_pattern.match(tl)
                    if m2:
                        # 같은 블록에 다른 캐릭터
                        raw2 = m2.group(1).strip()
                        rest2 = m2.group(2).strip()
                        if not is_sfx(raw2):
                            # 현재 대사 저장
                            results.append((char_name, dialogue, start_time, end_time))
                            char_name = normalize_name(raw2)
                            current_char = char_name
                            dialogue = rest2
                    else:
                        if not is_song(tl):
                            dialogue += tl
                if dialogue:
                    results.append((char_name, dialogue, start_time, end_time))
            else:
                # 캐릭터명만 있고, 대사는 다음 줄
                dialogue_parts = []
                for tl in text_lines[1:]:
                    tl = tl.strip()
                    if not tl or is_song(tl):
                        continue
                    m2 = char_pattern.match(tl)
                    if m2:
                        raw2 = m2.group(1).strip()
                        rest2 = m2.group(2).strip()
                        if not is_sfx(raw2):
                            if dialogue_parts:
                                results.append((char_name, ''.join(dialogue_parts), start_time, end_time))
                                dialogue_parts = []
                            char_name = normalize_name(raw2)
                            current_char = char_name
                            if rest2:
                                dialogue_parts.append(rest2)
                    else:
                        dialogue_parts.append(tl)
                if dialogue_parts:
                    results.append((char_name, ''.join(dialogue_parts), start_time, end_time))

        else:
            # 캐릭터 표시 없음 → 이전 화자의 이어지는 대사
            if current_char:
                dialogue = full_text.strip()
                if dialogue and not is_song(dialogue):
                    # 효과음 단독 줄 제외
                    clean = re.sub(r'（[^）]+）', '', dialogue).strip()
                    if clean:
                        results.append((current_char, clean, start_time, end_time))

    return results


def get_srt_files(directory: str, pattern: str) -> list:
    """디렉토리에서 SRT 파일 목록을 에피소드 순으로 반환"""
    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith('.jp.srt'):
            # 에피소드 번호 추출
            m = re.search(pattern, f)
            if m:
                ep_num = int(m.group(1))
                files.append((ep_num, os.path.join(directory, f)))
    return files


# ══════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════
all_dialogue = {}  # char -> [dialogue lines]
all_dialogue_with_meta = {}  # char -> [{text, season, episode, time}]
episode_meta = []
total_lines = 0

print("=" * 60)
print("Code Geass Netflix 일본어 자막 파싱")
print("=" * 60)

for season, directory, ep_pattern in [
    ("S1", S1_DIR, r'Code_Geass_(\d+)_'),
    ("R2", S2_DIR, r'Code_Geass_R2_(\d+)_'),
]:
    srt_files = get_srt_files(directory, ep_pattern)
    print(f"\n{season}: {len(srt_files)}화 발견")

    for ep_num, filepath in srt_files:
        label = f"  {season} E{ep_num:02d}"
        try:
            pairs = parse_srt_file(filepath)
            ep_chars = {}

            for char, dialogue, start_t, end_t in pairs:
                if char not in all_dialogue:
                    all_dialogue[char] = []
                    all_dialogue_with_meta[char] = []
                all_dialogue[char].append(dialogue)
                all_dialogue_with_meta[char].append({
                    'text': dialogue,
                    'season': season,
                    'episode': ep_num,
                    'start': start_t,
                    'end': end_t,
                })
                ep_chars[char] = ep_chars.get(char, 0) + 1

            total_lines += len(pairs)
            episode_meta.append({
                'season': season,
                'episode': ep_num,
                'lines': len(pairs),
                'characters': len(ep_chars),
                'top_chars': sorted(ep_chars.items(), key=lambda x: -x[1])[:5],
            })
            top3 = ', '.join(f'{c}({n})' for c, n in sorted(ep_chars.items(), key=lambda x: -x[1])[:3])
            print(f"{label}: {len(pairs):>4}줄, {len(ep_chars):>3}명  [{top3}]")

        except Exception as e:
            print(f"{label}: ERROR - {e}")
            episode_meta.append({
                'season': season,
                'episode': ep_num,
                'lines': 0,
                'characters': 0,
                'error': str(e),
            })

# ── 결과 정리 ──
print(f"\n{'='*60}")
print(f"총 {total_lines}줄, {len(all_dialogue)}명 캐릭터")
print(f"\n캐릭터별 대사 수 (상위 30):")
sorted_chars = sorted(all_dialogue.items(), key=lambda x: -len(x[1]))
for i, (char, lines) in enumerate(sorted_chars[:30]):
    bar = '█' * (len(lines) // 20)
    print(f"  {i+1:>2}. {char:>12}: {len(lines):>4}줄  {bar}")

# ── JSON 저장 ──
output_path = 'D:/game-portfolio-main/code_geass_dialogue_ja.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_dialogue, f, ensure_ascii=False, indent=2)
print(f"\n저장: {output_path}")

# 메타 정보
meta_path = 'D:/game-portfolio-main/code_geass_episodes_ja.json'
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(episode_meta, f, ensure_ascii=False, indent=2)
print(f"저장: {meta_path}")

# 상세 정보 (타임스탬프 포함)
detail_path = 'D:/game-portfolio-main/code_geass_dialogue_ja_detail.json'
with open(detail_path, 'w', encoding='utf-8') as f:
    json.dump(all_dialogue_with_meta, f, ensure_ascii=False, indent=2)
print(f"저장: {detail_path}")
