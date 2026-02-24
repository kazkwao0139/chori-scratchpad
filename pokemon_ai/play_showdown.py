"""인간 vs AI — Showdown 서버 대전 (Nash 균형 기반)."""
import asyncio
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "entropy_adam"))

from data_loader import GameData, _to_id
from neural_net import (
    PokemonNet, NetworkEvaluator,
    TeamPreviewNet, PreviewEvaluator,
)
from rule_evaluator import RuleBasedEvaluator
from battle_sim import BattleSimulator, make_pokemon_from_stats, load_sample_teams
from nash_solver import NashSolver
from endgame_solver import EndgameNashSolver
from poke_env import AccountConfiguration
from agent import NashPlayer, team_to_showdown_paste

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# Gen 9 SV에 없는 포켓몬 (Min Source Gen = 9 위반)
NOT_IN_GEN9 = {
    _to_id(n) for n in [
        "Ho-Oh", "Zacian-Crowned", "Eternatus", "Smeargle",
        "Muk-Alola", "Weezing-Galar",
    ]
}

# 제한전설 (Reg J: 2마리까지)
RESTRICTED = {
    _to_id(n) for n in [
        "Koraidon", "Miraidon", "Calyrex-Shadow", "Calyrex-Ice",
        "Kyogre", "Lugia", "Lunala", "Rayquaza",
        "Necrozma-Dusk-Mane", "Terapagos",
    ]
}


def load_model(cls, path, device):
    """체크포인트 로드."""
    model = cls().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def build_legal_team(build_eval, gd, max_restricted=2):
    """Reg J 합법 팀 생성. 빌드 모델로 뽑되 불법 포켓몬 교체."""
    team, _ = build_eval.build_team(temperature=0.3)

    # 합법 필터링
    legal_team = []
    restricted_count = 0
    for p in team:
        pid = _to_id(p.name)
        if pid in NOT_IN_GEN9:
            continue
        if pid in RESTRICTED:
            if restricted_count >= max_restricted:
                continue
            restricted_count += 1
        legal_team.append(p)

    # 부족하면 후보 풀에서 채우기
    if len(legal_team) < 6:
        used = {_to_id(p.name) for p in legal_team}
        for candidate in build_eval.candidate_pokemon:
            if len(legal_team) >= 6:
                break
            cid = _to_id(candidate.name)
            if cid in used or cid in NOT_IN_GEN9:
                continue
            if cid in RESTRICTED and restricted_count >= max_restricted:
                continue
            if cid in RESTRICTED:
                restricted_count += 1
            legal_team.append(candidate)
            used.add(cid)

    return legal_team


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)

    # ── 룰 기반 평가 (신경망 대체) ──
    print("RuleBasedEvaluator 초기화...")
    net_eval = RuleBasedEvaluator(gd)

    # Nash Solver 생성 (인간 대전: 시간 기반 iterative deepening)
    endgame = EndgameNashSolver(sim, gd, net_eval, max_depth=20)
    nash = NashSolver(sim, gd, net_eval, endgame_solver=endgame,
                      move_time_budget=40)

    # ── 프리뷰/빌드 모델 ──
    preview_path = os.path.join(CHECKPOINT_DIR, "preview_best.pt")
    if not os.path.exists(preview_path):
        preview_path = os.path.join(CHECKPOINT_DIR, "preview_final.pt")
    build_path = os.path.join(CHECKPOINT_DIR, "build_best.pt")
    if not os.path.exists(build_path):
        build_path = os.path.join(CHECKPOINT_DIR, "build_final.pt")

    preview_model = load_model(TeamPreviewNet, preview_path, device)

    # AI 팀 — sample_teams에서 선택
    import random
    teams_path = os.path.join(os.path.dirname(__file__), "data", "sample_teams.txt")
    sample_teams = load_sample_teams(gd, teams_path)
    ai_team_full = random.choice(sample_teams)
    team_names = [p.name for p in ai_team_full]
    team_paste = team_to_showdown_paste(gd, team_names, "bss", level=50)
    print(f"\nSample teams: {len(sample_teams)}개 중 1개 선택")
    show_team("AI의 팀", ai_team_full)

    # NashPlayer 생성
    print(f"\n{'='*50}")
    print("  Showdown 접속 중... (Nash Equilibrium)")
    print(f"{'='*50}")

    player = NashPlayer(
        game_data=gd,
        format_name="bss",
        log_decisions=True,
        nash_solver=nash,
        network_evaluator=net_eval,
        preview_checkpoint_path=preview_path,
        battle_format="gen9bssregi",
        team=team_paste,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("MacroNash", None),
    )

    print(f"\n  AI 준비 완료!")
    print(f"  브라우저: http://localhost:8000")
    print(f"  AI 유저명: {player.username}")
    print(f"  포맷: [Gen 9] BSS Reg I")
    print(f"  → 위 유저명에게 챌린지를 보내세요!")
    print(f"\n  대기 중...\n")

    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    async def run():
        await player.accept_challenges(None, n_challenges=n_games)
        print(f"\n{'='*50}")
        print(f"  대전 완료!")
        print(f"  승: {player.n_won_battles} / 패: {player.n_lost_battles}")
        print(f"{'='*50}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
