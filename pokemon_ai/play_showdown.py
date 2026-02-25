"""인간 vs AI — Showdown 서버 대전 (Nash 균형 기반).

사용법:
  python play_showdown.py --paste data/my_team.txt
  python play_showdown.py --paste data/my_team.txt --budget 20 --games 3
  python play_showdown.py --paste data/my_team.txt --format gen9bssregi --username NashBot
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import GameData
from rule_evaluator import RuleBasedEvaluator
from battle_sim import BattleSimulator, parse_showdown_paste
from nash_solver import NashSolver
from endgame_solver import EndgameNashSolver
from poke_env import AccountConfiguration
from agent import NashPlayer, team_to_showdown_paste


def show_team(label, team):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for i, p in enumerate(team):
        types = "/".join(p.types)
        moves = ", ".join(p.moves)
        print(f"\n  [{i+1}] {p.name} ({types})")
        print(f"      특성: {p.ability}")
        print(f"      아이템: {p.item}")
        print(f"      기술: {moves}")


def main():
    parser = argparse.ArgumentParser(description="Nash 균형 Showdown 대전 봇")
    parser.add_argument("--paste", type=str, default=None,
                        help="Showdown paste 팀 파일 경로 (예: data/my_team.txt)")
    parser.add_argument("--budget", type=float, default=30.0,
                        help="Nash 시간 예산 (초, 기본 30)")
    parser.add_argument("--format", type=str, default="gen9bssregi",
                        help="배틀 포맷 (기본 gen9bssregi)")
    parser.add_argument("--username", type=str, default="MacroNash",
                        help="AI 닉네임 (기본 MacroNash)")
    parser.add_argument("--games", type=int, default=1,
                        help="연속 대전 수 (기본 1)")
    args = parser.parse_args()

    # ── 엔진 초기화 ──
    print("GameData 로드 중...")
    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)

    print("RuleBasedEvaluator 초기화...")
    evaluator = RuleBasedEvaluator(gd)

    endgame = EndgameNashSolver(sim, gd, evaluator, max_depth=20)
    nash = NashSolver(sim, gd, evaluator, endgame_solver=endgame,
                      move_time_budget=args.budget)

    # ── 팀 로드 ──
    # format ID에서 내부 format_name 추출 (gen9bssregi → bss)
    fmt_id = args.format.lower()
    if "bss" in fmt_id or "battlestadium" in fmt_id:
        format_name = "bss"
    elif "ou" in fmt_id:
        format_name = "ou"
    else:
        format_name = "bss"

    if args.paste:
        paste_path = args.paste
        if not os.path.isabs(paste_path):
            paste_path = os.path.join(os.path.dirname(__file__), paste_path)
        with open(paste_path, "r", encoding="utf-8") as f:
            paste_text = f.read()
        ai_team = parse_showdown_paste(gd, paste_text)
        if not ai_team:
            print(f"[에러] {paste_path}에서 팀을 파싱할 수 없습니다.")
            sys.exit(1)
        # paste 원문을 그대로 poke-env에 전달 (Showdown 포맷 그대로)
        team_paste = paste_text
        print(f"\n팀 로드: {paste_path} ({len(ai_team)}마리)")
    else:
        # --paste 없으면 Smogon 통계 기반 자동 팀 생성
        from battle_sim import load_sample_teams
        teams_path = os.path.join(os.path.dirname(__file__), "data", "sample_teams.txt")
        if not os.path.exists(teams_path):
            print("[에러] --paste 플래그로 팀 파일을 지정하거나,")
            print(f"       {teams_path}에 sample_teams.txt를 넣어주세요.")
            sys.exit(1)
        import random
        sample_teams = load_sample_teams(gd, teams_path)
        ai_team = random.choice(sample_teams)
        team_names = [p.name for p in ai_team]
        team_paste = team_to_showdown_paste(gd, team_names, format_name, level=50)
        print(f"\nSample teams: {len(sample_teams)}개 중 1개 랜덤 선택")

    show_team("AI의 팀", ai_team)

    # ── NashPlayer 생성 ──
    print(f"\n{'='*50}")
    print("  Showdown 접속 중... (Nash Equilibrium)")
    print(f"{'='*50}")

    player = NashPlayer(
        game_data=gd,
        format_name=format_name,
        log_decisions=True,
        nash_solver=nash,
        network_evaluator=evaluator,
        preview_checkpoint_path=None,
        battle_format=args.format,
        team=team_paste,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(args.username, None),
    )

    print(f"\n  AI 준비 완료!")
    print(f"  브라우저: http://localhost:8000")
    print(f"  AI 유저명: {player.username}")
    print(f"  포맷: {args.format}")
    print(f"  Nash 예산: {args.budget}초")
    print(f"  → http://localhost:8000 에서 {args.username}에게 챌린지를 보내세요!")
    print(f"\n  대기 중...\n")

    async def run():
        await player.accept_challenges(None, n_challenges=args.games)
        print(f"\n{'='*50}")
        print(f"  대전 완료!")
        print(f"  승: {player.n_won_battles} / 패: {player.n_lost_battles}")
        print(f"{'='*50}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
