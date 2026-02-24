"""AI 승리 과정을 보여주는 데모 — 3판 풀 로그."""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import GameData
from battle_sim import BattleSimulator, load_sample_teams
from nash_solver import NashSolver
from endgame_solver import EndgameNashSolver
from rule_evaluator import RuleBasedEvaluator


def action_str(gd, state, player, action):
    side = state.sides[player]
    active = side.active
    if 4 <= action <= 8:
        idx = action - 4
        if idx < len(side.team):
            return f"Switch→{side.team[idx].name}"
        return f"Switch→#{idx}"
    is_tera = action >= 9
    move_idx = action - 9 if is_tera else action
    if move_idx < len(active.moves):
        mid = active.moves[move_idx]
        mv = gd.get_move(mid)
        name = mv["name"] if mv else mid
        return f"{'Tera+' if is_tera else ''}{name}"
    return f"Action#{action}"


def play_verbose(sim, nash, gd, team1, team2, game_num):
    t1_names = "/".join(p.name for p in team1)
    t2_names = "/".join(p.name for p in team2)
    print(f"\n{'='*60}")
    print(f"  Game {game_num}: {t1_names}")
    print(f"      vs {t2_names}")
    print(f"{'='*60}")

    state = sim.create_battle_state(list(team1), list(team2))
    t0 = time.time()

    for turn_i in range(60):
        if state.is_terminal:
            break

        legal_p1 = sim.get_legal_actions(state, 0)
        legal_p2 = sim.get_legal_actions(state, 1)
        if not legal_p1 or not legal_p2:
            break

        gv = 0.0
        try:
            p1_dict, _, gv, _, _, _, _ = nash.get_nash_strategies_both(
                state, add_noise=False)
        except Exception:
            p1_dict = {}

        if p1_dict:
            actions = list(p1_dict.keys())
            probs = [p1_dict[a] for a in actions]
            a1 = random.choices(actions, weights=probs, k=1)[0]
            # 확률 분포 표시
            strat_str = "  ".join(
                f"{action_str(gd, state, 0, a)}:{p:.0%}"
                for a, p in sorted(p1_dict.items(), key=lambda x: -x[1])
                if p >= 0.05
            )
        else:
            a1 = random.choice(legal_p1)
            strat_str = "fallback random"

        a2 = random.choice(legal_p2)

        p1a = state.sides[0].active
        p2a = state.sides[1].active
        p1_hp = "/".join(
            f"{p.name}:{p.cur_hp}/{p.max_hp}" for p in state.sides[0].team if not p.fainted)
        p2_hp = "/".join(
            f"{p.name}:{p.cur_hp}/{p.max_hp}" for p in state.sides[1].team if not p.fainted)

        print(f"\n  T{state.turn:2d} [{p1a.name} vs {p2a.name}]  GV={gv:+.3f}")
        print(f"     AI: {action_str(gd, state, 0, a1)}  (Nash: {strat_str})")
        print(f"     상대: {action_str(gd, state, 1, a2)}")
        print(f"     HP  우리: [{p1_hp}]")
        print(f"     HP  상대: [{p2_hp}]")

        state = sim.step(state, a1, a2)

    elapsed = time.time() - t0
    winner = state.winner if state.is_terminal else -1
    tag = "WIN" if winner == 0 else "LOSE" if winner == 1 else "DRAW"
    print(f"\n  >>> 결과: {tag}  ({state.turn}턴, {elapsed:.1f}s)")
    return winner


def main():
    random.seed(42)
    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)
    teams_path = os.path.join(os.path.dirname(__file__), "data", "sample_teams.txt")
    all_teams = load_sample_teams(gd, teams_path)

    evaluator = RuleBasedEvaluator(gd)
    endgame = EndgameNashSolver(sim, gd, evaluator, max_depth=4)
    nash = NashSolver(sim, gd, evaluator, endgame_solver=endgame, midgame_depth=4)

    print("3판 데모 시작!\n")

    for g in range(3):
        t1_full = random.choice(all_teams)
        t2_full = random.choice(all_teams)
        t1 = evaluator.select_team(t1_full, t2_full)
        t2 = random.sample(t2_full, 3)
        play_verbose(sim, nash, gd, t1, t2, g + 1)


if __name__ == "__main__":
    main()
