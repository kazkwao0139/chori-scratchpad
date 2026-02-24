"""2-Player MCTS — 양쪽 최적 수 탐색.

핵심 변경 (기존 대비):
- Max/Min 2-ply: 아군 수 선택(Max) → 상대 수 선택(Min) → step → 다음 턴
- 상대도 UCB 탐색: "불리하면 교체" 같은 최적 플레이가 자연스럽게 나옴
- 상대 분기 Top-K: 시뮬레이션 예산 안에서 깊이 유지
- 면역 필터링: 0× 기술은 트리에서 완전 제거
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data_loader import GameData, _to_id
from battle_sim import BattleSimulator, BattleState, ActionType


# ═══════════════════════════════════════════════════════════════
#  MCTS 노드
# ═══════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """2-Player MCTS 트리 노드.

    Max 노드: root_player가 행동 → UCB 최대화
    Min 노드: 상대가 행동 → UCB 최소화 (= 상대 관점 최대화)
    """
    state: BattleState
    acting_player: int        # 이 노드에서 행동할 플레이어 (0 or 1)
    is_max: bool              # True = 아군 턴 (maximize), False = 적 턴 (minimize)
    parent: Optional[MCTSNode] = None
    action_from_parent: int = -1
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0  # 항상 root_player 관점
    prior: float = 0.0
    is_expanded: bool = False

    # 2-ply: Max 노드 → Min 노드로 넘어갈 때 Max의 선택을 기억
    pending_action: int = -1

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_max(self, c: float = 1.414, pw: float = 2.0) -> float:
        """Max 노드용 UCB: 높을수록 좋음."""
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        return (self.q_value
                + c * math.sqrt(math.log(parent_n) / self.visit_count)
                + pw * self.prior / (1 + self.visit_count))

    def ucb1_min(self, c: float = 1.414, pw: float = 2.0) -> float:
        """Min 노드용 UCB: 상대 관점 (우리 Q 낮을수록 좋음)."""
        if self.visit_count == 0:
            return float("inf")
        parent_n = self.parent.visit_count if self.parent else 1
        return (-self.q_value
                + c * math.sqrt(math.log(parent_n) / self.visit_count)
                + pw * self.prior / (1 + self.visit_count))


# ═══════════════════════════════════════════════════════════════
#  MCTS 엔진
# ═══════════════════════════════════════════════════════════════

class MCTS:
    """2-Player Monte Carlo Tree Search for Pokemon battles."""

    def __init__(
        self,
        game_data: GameData,
        simulator: BattleSimulator,
        n_simulations: int = 800,
        rollout_depth: int = 30,
        exploration_weight: float = 1.414,
        prior_weight: float = 2.0,
        n_rollout_per_leaf: int = 4,
        use_parallel: bool = True,
        n_workers: int = 4,
        format_name: str = "bss",
        max_opp_branches: int = 5,
        network_evaluator=None,
        dirichlet_alpha: float = 0.0,
        dirichlet_weight: float = 0.25,
    ):
        self.gd = game_data
        self.sim = simulator
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.prior_weight = prior_weight
        self.n_rollout_per_leaf = n_rollout_per_leaf
        self.use_parallel = use_parallel
        self.n_workers = n_workers
        self.format_name = format_name
        self.max_opp_branches = max_opp_branches

        # AlphaZero: network → Smogon prior/rollout 대체
        self.network = network_evaluator
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        self.root_player = 0  # search() 에서 설정

        if use_parallel:
            self.executor = ThreadPoolExecutor(max_workers=n_workers)

    # ─── 메인 탐색 ───────────────────────────────────────────

    def search(self, state: BattleState, player: int,
               n_simulations: int | None = None,
               time_limit: float | None = None) -> dict[int, float]:
        """MCTS 탐색 → {action: visit_count}.

        2-ply 구조: 루트(Max) → 상대(Min) → step → 다음 Max → ...
        """
        n_sims = n_simulations or self.n_simulations
        self.root_player = player

        root = MCTSNode(
            state=state,
            acting_player=player,
            is_max=True,
        )
        self._expand(root)

        # Dirichlet 노이즈: 셀프플레이 탐색 다양성 보장
        if self.dirichlet_alpha > 0 and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children))
            eps = self.dirichlet_weight
            for child, eta in zip(root.children.values(), noise):
                child.prior = (1 - eps) * child.prior + eps * eta

        start_time = time.time()
        for i in range(n_sims):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # 1. Selection — Max/Min 교대 UCB
            node = self._select(root)

            # 2. Expansion
            if not node.state.is_terminal and not node.is_expanded:
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))

            # 3. Simulation (Rollout)
            value = self._simulate(node)

            # 4. Backpropagation
            self._backpropagate(node, value)

        return {a: c.visit_count for a, c in root.children.items()}

    def get_best_action(self, state: BattleState, player: int,
                        **kwargs) -> int:
        visits = self.search(state, player, **kwargs)
        if not visits:
            legal = self.sim.get_legal_actions(state, player)
            return random.choice(legal) if legal else 0
        return max(visits, key=visits.get)

    def get_action_probs(self, state: BattleState, player: int,
                         temperature: float = 1.0,
                         **kwargs) -> dict[int, float]:
        visits = self.search(state, player, **kwargs)
        if not visits:
            legal = self.sim.get_legal_actions(state, player)
            return {a: 1.0 / len(legal) for a in legal}
        if temperature == 0:
            best = max(visits, key=visits.get)
            return {a: 1.0 if a == best else 0.0 for a in visits}
        total = sum(v ** (1.0 / temperature) for v in visits.values())
        return {a: (v ** (1.0 / temperature)) / total
                for a, v in visits.items()}

    # ─── Selection ───────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Max/Min 교대 UCB 탐색."""
        c = self.exploration_weight
        pw = self.prior_weight
        while node.is_expanded and node.children and not node.state.is_terminal:
            if node.is_max:
                node = max(node.children.values(),
                           key=lambda n: n.ucb1_max(c, pw))
            else:
                node = max(node.children.values(),
                           key=lambda n: n.ucb1_min(c, pw))
        return node

    # ─── Expansion ───────────────────────────────────────────

    def _expand(self, node: MCTSNode):
        """2-ply 확장.

        Max 노드 → 아군 액션별 자식 (Min 노드, 같은 state)
        Min 노드 → 상대 액션별 자식 (Max 노드, step() 후 새 state)
        """
        if node.state.is_terminal:
            node.is_expanded = True
            return

        # 행동할 수 있는 액션
        filter_imm = (node.acting_player == self.root_player)
        legal = self.sim.get_legal_actions(
            node.state, node.acting_player, filter_immune=filter_imm)
        if not legal:
            node.is_expanded = True
            return

        priors = self._compute_priors(node, legal)

        if node.is_max:
            # ── Max 노드: 아군 수 선택 ──
            # 자식 = Min 노드 (상대 턴, 같은 state, 아군 행동 기억)
            for action in legal:
                child = MCTSNode(
                    state=node.state,
                    acting_player=1 - self.root_player,
                    is_max=False,
                    parent=node,
                    action_from_parent=action,
                    prior=priors.get(action, 1.0 / len(legal)),
                    pending_action=action,
                )
                node.children[action] = child
        else:
            # ── Min 노드: 상대 수 선택 ──
            # 아군 행동 = pending_action, 상대 행동 = action
            # → step() → 자식 = Max 노드 (다음 턴, 새 state)
            our_action = node.pending_action

            # 상대 분기 수 제한 (Top-K by prior)
            sorted_actions = sorted(legal,
                                    key=lambda a: priors.get(a, 0),
                                    reverse=True)
            actions_to_use = sorted_actions[:self.max_opp_branches]

            for action in actions_to_use:
                if self.root_player == 0:
                    next_state = self.sim.step(node.state, our_action, action)
                else:
                    next_state = self.sim.step(node.state, action, our_action)

                child = MCTSNode(
                    state=next_state,
                    acting_player=self.root_player,
                    is_max=True,
                    parent=node,
                    action_from_parent=action,
                    prior=priors.get(action, 1.0 / len(legal)),
                )
                node.children[action] = child

        node.is_expanded = True

    # ─── Prior 계산 ──────────────────────────────────────────

    def _compute_priors(self, node: MCTSNode,
                        legal_actions: list[int]) -> dict[int, float]:
        """Prior 계산: Network 있으면 Policy Net, 없으면 Smogon prior."""
        # AlphaZero: Policy Net이 Smogon prior 대체
        if self.network:
            priors, _ = self.network.evaluate(
                node.state, node.acting_player, legal_actions)
            return priors

        # 기존: Smogon 사용률 + 타입 상성 기반 prior
        side = node.state.sides[node.acting_player]
        opp_side = node.state.sides[1 - node.acting_player]
        active = side.active
        defender = opp_side.active

        species_stats = self.gd.get_stats(active.species_id, self.format_name)
        move_probs = species_stats.get("moves", {}) if species_stats else {}

        priors = {}
        for action in legal_actions:
            if action <= ActionType.MOVE4:
                move_idx = action - ActionType.MOVE1
                move_id = (active.moves[move_idx]
                           if move_idx < len(active.moves) else None)
                prior = move_probs.get(move_id, 0.01) if move_id else 0.01
            elif action <= ActionType.SWITCH5:
                # 교체 prior: 불리한 대면이면 약간 올림
                prior = self._switch_prior(active, defender, side, action)
                priors[action] = prior
                continue
            else:  # 테라 기술
                move_idx = action - ActionType.TERA_MOVE1
                move_id = (active.moves[move_idx]
                           if move_idx < len(active.moves) else None)
                prior = (move_probs.get(move_id, 0.01) * 0.8) if move_id else 0.01

            # 타입 상성 보정
            if move_id and defender and not defender.fainted:
                move_data = self.gd.get_move(move_id)
                if move_data and not move_data["is_status"]:
                    eff = self.gd.effectiveness(move_data["type"],
                                                defender.types)
                    if eff == 0:
                        prior *= 0.001
                    elif eff <= 0.25:
                        prior *= 0.05
                    elif eff == 0.5:
                        prior *= 0.5
                    elif eff >= 4.0:
                        prior *= 3.0
                    elif eff >= 2.0:
                        prior *= 2.0

            priors[action] = prior

        total = sum(priors.values())
        if total > 0:
            priors = {a: p / total for a, p in priors.items()}
        return priors

    def _switch_prior(self, active, defender, side, action) -> float:
        """교체 prior — 대면 불리도에 따라 조절.

        수동 가중치가 아니라 MCTS 탐색의 초기 안내용.
        실제 최적 수는 UCB+롤아웃으로 결정됨.
        """
        base = 0.05

        # 상대 STAB 타입에 약점이 잡히면 교체 prior 증가
        if defender and not defender.fainted:
            for t in defender.types:
                eff = self.gd.effectiveness(t, active.types)
                if eff >= 2.0:
                    base = 0.12  # 약점 잡힘 → 교체 고려
                    break

        # HP가 낮으면 교체 고려
        if active.hp_pct < 0.3:
            base = max(base, 0.10)

        # 벤치 포켓몬이 유리한 매치업이면 보너스
        switch_idx = action - ActionType.SWITCH1
        if switch_idx < len(side.team):
            bench_poke = side.team[switch_idx]
            if not bench_poke.fainted and defender and not defender.fainted:
                # 벤치가 상대 STAB 반감?
                for t in defender.types:
                    eff = self.gd.effectiveness(t, bench_poke.types)
                    if eff <= 0.5:
                        base *= 1.5
                        break

        return base

    # ─── Simulation (Rollout) ────────────────────────────────

    def _simulate(self, node: MCTSNode) -> float:
        """롤아웃 → root_player 관점 승률 (0~1)."""
        if node.state.is_terminal:
            return 1.0 if node.state.winner == self.root_player else 0.0

        # AlphaZero: Value Net이 롤아웃 대체 (1회 forward pass)
        if self.network:
            value = self.network.evaluate_value(
                node.state, self.root_player)
            return (value + 1.0) / 2.0  # [-1,+1] → [0,1]

        if self.use_parallel and self.n_rollout_per_leaf > 1:
            return self._simulate_parallel(node)
        return self._simulate_single(node)

    def _simulate_single(self, node: MCTSNode) -> float:
        wins = 0
        for _ in range(self.n_rollout_per_leaf):
            winner = self.sim.rollout(node.state, self.rollout_depth)
            if winner == self.root_player:
                wins += 1
        return wins / self.n_rollout_per_leaf

    def _simulate_parallel(self, node: MCTSNode) -> float:
        futures = [
            self.executor.submit(self.sim.rollout,
                                 node.state, self.rollout_depth)
            for _ in range(self.n_rollout_per_leaf)
        ]
        wins = sum(1 for f in futures if f.result() == self.root_player)
        return wins / self.n_rollout_per_leaf

    # ─── Backpropagation ─────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, value: float):
        """승률을 루트까지 역전파. 값은 항상 root_player 관점."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    # ─── 정적 평가 (롤아웃 미완료 시 보조) ───────────────────

    def _evaluate(self, state: BattleState) -> float:
        """HP + 생존수 기반 빠른 평가. root_player 관점 [0, 1]."""
        our = state.sides[self.root_player]
        opp = state.sides[1 - self.root_player]

        our_hp = sum(p.hp_pct for p in our.team if not p.fainted)
        opp_hp = sum(p.hp_pct for p in opp.team if not p.fainted)
        our_alive = our.alive_count
        opp_alive = opp.alive_count

        # 정규화: (our - opp) 범위를 [0, 1]로
        hp_diff = our_hp - opp_hp
        alive_diff = our_alive - opp_alive
        max_team = max(len(our.team), len(opp.team), 1)

        score = 0.5 + 0.25 * (hp_diff / max_team) + 0.25 * (alive_diff / max_team)
        return max(0.0, min(1.0, score))

    # ─── 디버그/시각화 ───────────────────────────────────────

    def get_search_info(self, state: BattleState, player: int,
                        **kwargs) -> dict:
        """탐색 결과를 상세히 반환 (agent.py에서 사용)."""
        visits = self.search(state, player, **kwargs)
        side = state.sides[player]
        active = side.active

        info = {
            "active_pokemon": active.name,
            "hp": f"{active.cur_hp}/{active.max_hp}",
            "actions": [],
        }

        total_visits = sum(visits.values()) or 1
        for action, count in sorted(visits.items(), key=lambda x: -x[1]):
            action_name = self._action_to_str(state, player, action)
            info["actions"].append({
                "action": action,
                "name": action_name,
                "visits": count,
                "probability": count / total_visits,
            })

        return info

    def _action_to_str(self, state: BattleState, player: int,
                       action: int) -> str:
        side = state.sides[player]
        active = side.active

        if action <= ActionType.MOVE4:
            idx = action - ActionType.MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return f"Move: {move['name']}" if move else f"Move {idx}"
            return f"Move {idx}"
        elif action <= ActionType.SWITCH5:
            idx = action - ActionType.SWITCH1
            if idx < len(side.team):
                return f"Switch: {side.team[idx].name}"
            return f"Switch {idx}"
        else:
            idx = action - ActionType.TERA_MOVE1
            if idx < len(active.moves):
                move = self.gd.get_move(active.moves[idx])
                return f"Tera+{move['name']}" if move else f"Tera Move {idx}"
            return f"Tera Move {idx}"


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """2-Player MCTS 검증."""
    print("=== 2-Player MCTS 검증 ===\n")

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)
    mcts = MCTS(
        game_data=gd, simulator=sim,
        n_simulations=200, rollout_depth=30,
        n_rollout_per_leaf=2, use_parallel=True, n_workers=4,
        format_name="bss", max_opp_branches=5,
    )

    from battle_sim import make_pokemon_from_stats

    # ── 테스트 1: 면역 기술 필터링 ──
    print("--- 테스트 1: Koraidon vs Landorus (지진 면역) ---")
    from battle_sim import Pokemon, Side
    koraidon = Pokemon(
        species_id='koraidon', name='Koraidon',
        types=['Fire', 'Dragon'],
        base_stats={'hp':100,'atk':135,'def':115,'spa':85,'spd':100,'spe':135},
        stats={'hp':340,'atk':335,'def':266,'spa':206,'spd':237,'spe':335},
        ability='orichalcumpulse', item='choicescarf',
        moves=['earthquake', 'flareblitz', 'uturn', 'dracometeor'],
        tera_type='Fire',
    )
    landorus = Pokemon(
        species_id='landorustherian', name='Landorus-Therian',
        types=['Ground', 'Flying'],
        base_stats={'hp':89,'atk':145,'def':90,'spa':105,'spd':80,'spe':91},
        stats={'hp':320,'atk':350,'def':216,'spa':246,'spd':196,'spe':218},
        ability='intimidate', item='leftovers',
        moves=['earthquake', 'uturn', 'stealthrock', 'knockoff'],
        tera_type='Water',
    )
    test_state = BattleState(
        sides=[
            Side(team=[koraidon], active_idx=0),
            Side(team=[landorus], active_idx=0),
        ]
    )
    info = mcts.get_search_info(test_state, 0, n_simulations=200)
    print(f"활성: {info['active_pokemon']}")
    for a in info["actions"]:
        print(f"  {a['name']:25s} | {a['visits']:4d} ({a['probability']:.1%})")
    eq_visits = sum(a["visits"] for a in info["actions"]
                    if "Earthquake" in a["name"] and "Tera" not in a["name"])
    print(f"Earthquake 방문: {eq_visits} → {'통과' if eq_visits == 0 else '실패'}\n")

    # ── 테스트 2: 상대가 불리하면 교체하는지 ──
    print("--- 테스트 2: 상대 관점 탐색 (불리 대면 교체) ---")
    # 상대: 물타입 vs 코라이돈(불/드래곤) → 불리 → 교체?
    vaporeon = Pokemon(
        species_id='vaporeon', name='Vaporeon',
        types=['Water'],
        base_stats={'hp':130,'atk':65,'def':60,'spa':110,'spd':95,'spe':65},
        stats={'hp':400,'atk':166,'def':156,'spa':256,'spd':226,'spe':166},
        ability='waterabsorb', item='leftovers',
        moves=['scald', 'icebeam', 'wish', 'protect'],
        tera_type='Water',
    )
    ttar = Pokemon(
        species_id='tyranitar', name='Tyranitar',
        types=['Rock', 'Dark'],
        base_stats={'hp':100,'atk':134,'def':110,'spa':95,'spd':100,'spe':61},
        stats={'hp':341,'atk':304,'def':256,'spa':226,'spd':236,'spe':158},
        ability='sandstream', item='choiceband',
        moves=['stoneedge', 'crunch', 'earthquake', 'icepunch'],
        tera_type='Rock',
    )

    # 상대 팀: Vaporeon (active) + Tyranitar (벤치)
    opp_state = BattleState(
        sides=[
            Side(team=[koraidon], active_idx=0),
            Side(team=[vaporeon, ttar], active_idx=0),
        ]
    )
    # 상대(player=1) 관점으로 탐색
    opp_info = mcts.get_search_info(opp_state, 1, n_simulations=200)
    print(f"상대 활성: {opp_info['active_pokemon']} vs Koraidon")
    for a in opp_info["actions"]:
        print(f"  {a['name']:25s} | {a['visits']:4d} ({a['probability']:.1%})")
    # Tyranitar로 교체 비율 확인
    switch_visits = sum(a["visits"] for a in opp_info["actions"]
                        if "Switch" in a["name"])
    total_v = sum(a["visits"] for a in opp_info["actions"])
    print(f"교체 비율: {switch_visits}/{total_v} "
          f"({switch_visits/total_v*100:.0f}%)\n")

    # ── 테스트 3: MCTS vs Random 대전 ──
    print("--- 테스트 3: 2P-MCTS vs Random 대전 ---")
    n_games = 10
    mcts_wins = 0
    for game in range(n_games):
        t1 = [
            make_pokemon_from_stats(gd, "Koraidon", "bss"),
            make_pokemon_from_stats(gd, "Ting-Lu", "bss"),
            make_pokemon_from_stats(gd, "Flutter Mane", "bss"),
        ]
        t2 = [
            make_pokemon_from_stats(gd, "Miraidon", "bss"),
            make_pokemon_from_stats(gd, "Gholdengo", "bss"),
            make_pokemon_from_stats(gd, "Great Tusk", "bss"),
        ]
        s = sim.create_battle_state(t1, t2)

        for turn in range(50):
            if s.is_terminal:
                break
            a1 = mcts.get_best_action(s, 0, n_simulations=100)
            legal2 = sim.get_legal_actions(s, 1)
            a2 = random.choice(legal2) if legal2 else 0
            s = sim.step(s, a1, a2)

        if s.winner == 0:
            mcts_wins += 1
        print(f"  게임 {game+1}: {'MCTS 승' if s.winner == 0 else 'Random 승'} "
              f"(턴 {s.turn})")

    print(f"\nMCTS 승률: {mcts_wins}/{n_games} ({mcts_wins/n_games:.0%})")
    print("검증 완료!")


if __name__ == "__main__":
    verify()
