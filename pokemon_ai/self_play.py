"""셀프플레이 — ReplayBuffer + TeamSampler + SelfPlayWorker.

- ReplayBuffer: 순환 버퍼 (max 500K), 디스크 저장/로드
- TeamSampler: Smogon Top-50에서 가중 랜덤 3마리
- play_game(): 양쪽 MCTS → policy dist 기록 → 결과 할당
"""

from __future__ import annotations

import os
import random
import pickle
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from data_loader import GameData, _to_id
from battle_sim import (
    BattleSimulator, BattleState, make_pokemon_from_stats, NUM_ACTIONS,
)
from mcts import MCTS
from neural_net import (
    encode_state, STATE_DIM, encode_preview_state,
    PREVIEW_STATE_DIM, NUM_COMBOS, COMBO_TABLE, PreviewEvaluator,
    BUILD_STATE_DIM, N_CANDIDATES,
)


# ═══════════════════════════════════════════════════════════════
#  Training Example
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingExample:
    """셀프플레이 학습 데이터 1개."""
    state: np.ndarray     # (STATE_DIM,) float32
    policy: np.ndarray    # (NUM_ACTIONS,) float32 — MCTS 방문 분포
    value: float          # +1 (승) / -1 (패)


@dataclass
class PreviewTrainingExample:
    """팀 프리뷰 학습 데이터 1개."""
    state: np.ndarray     # (PREVIEW_STATE_DIM,) float32
    policy: np.ndarray    # (NUM_COMBOS,) float32 — 프리뷰 선택 분포
    value: float          # +1 (승) / -1 (패)


@dataclass
class BuildTrainingExample:
    """팀 빌드 학습 데이터 1개."""
    state: np.ndarray     # (BUILD_STATE_DIM,) float32
    policy: np.ndarray    # (N_CANDIDATES,) float32 — 선택 분포
    value: float          # +1/-1
    step: int             # 0-5


# ═══════════════════════════════════════════════════════════════
#  ReplayBuffer
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    """순환 버퍼 — max_size examples 저장."""

    def __init__(self, max_size: int = 500_000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, example: TrainingExample):
        self.buffer.append(example)

    def add_batch(self, examples: list[TrainingExample]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """랜덤 샘플 → (states, policies, values) 텐서."""
        batch = random.sample(
            list(self.buffer), min(batch_size, len(self.buffer)))
        states = torch.tensor(
            np.array([e.state for e in batch]), dtype=torch.float32)
        policies = torch.tensor(
            np.array([e.policy for e in batch]), dtype=torch.float32)
        values = torch.tensor(
            [e.value for e in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        """디스크에 저장."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)
        print(f"ReplayBuffer 저장: {len(self.buffer)} examples → {path}")

    def load(self, path: str):
        """디스크에서 로드."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.buffer.maxlen)
        print(f"ReplayBuffer 로드: {len(self.buffer)} examples ← {path}")


# ═══════════════════════════════════════════════════════════════
#  PreviewReplayBuffer
# ═══════════════════════════════════════════════════════════════

class PreviewReplayBuffer:
    """팀 프리뷰 학습용 순환 버퍼."""

    def __init__(self, max_size: int = 100_000):
        self.buffer: deque[PreviewTrainingExample] = deque(maxlen=max_size)

    def add(self, example: PreviewTrainingExample):
        self.buffer.append(example)

    def add_batch(self, examples: list[PreviewTrainingExample]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """랜덤 샘플 → (states, policies, values) 텐서."""
        batch = random.sample(
            list(self.buffer), min(batch_size, len(self.buffer)))
        states = torch.tensor(
            np.array([e.state for e in batch]), dtype=torch.float32)
        policies = torch.tensor(
            np.array([e.policy for e in batch]), dtype=torch.float32)
        values = torch.tensor(
            [e.value for e in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)
        print(f"PreviewReplayBuffer 저장: {len(self.buffer)} examples → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.buffer.maxlen)
        print(f"PreviewReplayBuffer 로드: {len(self.buffer)} examples ← {path}")


# ═══════════════════════════════════════════════════════════════
#  BuildReplayBuffer
# ═══════════════════════════════════════════════════════════════

class BuildReplayBuffer:
    """팀 빌드 학습용 순환 버퍼."""

    def __init__(self, max_size: int = 200_000):
        self.buffer: deque[BuildTrainingExample] = deque(maxlen=max_size)

    def add(self, example: BuildTrainingExample):
        self.buffer.append(example)

    def add_batch(self, examples: list[BuildTrainingExample]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """랜덤 샘플 → (states, policies, values) 텐서."""
        batch = random.sample(
            list(self.buffer), min(batch_size, len(self.buffer)))
        states = torch.tensor(
            np.array([e.state for e in batch]), dtype=torch.float32)
        policies = torch.tensor(
            np.array([e.policy for e in batch]), dtype=torch.float32)
        values = torch.tensor(
            [e.value for e in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)
        print(f"BuildReplayBuffer 저장: {len(self.buffer)} examples → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.buffer.maxlen)
        print(f"BuildReplayBuffer 로드: {len(self.buffer)} examples ← {path}")


# ═══════════════════════════════════════════════════════════════
#  TeamSampler
# ═══════════════════════════════════════════════════════════════

class TeamSampler:
    """Smogon 사용률 기반 팀 샘플러 — 다양성 보장."""

    def __init__(self, game_data: GameData, format_name: str = "bss",
                 top_n: int = 50):
        self.gd = game_data
        self.format_name = format_name

        stats = (game_data.bss_stats if format_name == "bss"
                 else game_data.vgc_stats)

        # 사용률 Top-N
        sorted_pokemon = sorted(
            stats.items(),
            key=lambda x: x[1].get("usage_pct", 0),
            reverse=True,
        )[:top_n]

        self.pokemon_ids = [pid for pid, _ in sorted_pokemon]
        self.names = [s.get("name", pid) for pid, s in sorted_pokemon]
        weights = [s.get("usage_pct", 0.01) for _, s in sorted_pokemon]

        # 정규화
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def sample_team(self, team_size: int = 6) -> list:
        """가중 랜덤으로 team_size 마리 샘플 (중복 없음)."""
        indices = []
        avail = list(range(len(self.pokemon_ids)))
        avail_w = list(self.weights)

        for _ in range(team_size):
            if not avail:
                break
            picked = random.choices(avail, weights=avail_w, k=1)[0]
            pos = avail.index(picked)
            avail.pop(pos)
            avail_w.pop(pos)
            # 재정규화
            total = sum(avail_w) if avail_w else 1
            if total > 0:
                avail_w = [w / total for w in avail_w]
            indices.append(picked)

        # Pokemon 객체 생성
        team = []
        for idx in indices:
            name = self.names[idx]
            try:
                poke = make_pokemon_from_stats(
                    self.gd, name, self.format_name)
                team.append(poke)
            except (ValueError, KeyError):
                continue
        return team


# ═══════════════════════════════════════════════════════════════
#  SelfPlayWorker
# ═══════════════════════════════════════════════════════════════

def _sample_action(probs: dict[int, float], temperature: float) -> int:
    """확률 분포에서 액션 샘플."""
    if not probs:
        return 0
    actions = list(probs.keys())
    weights = list(probs.values())
    if temperature < 0.01:
        return actions[int(np.argmax(weights))]
    return random.choices(actions, weights=weights, k=1)[0]


def _sample_combo(probs: np.ndarray, temperature: float) -> int:
    """콤보 확률 분포에서 인덱스 샘플."""
    if temperature < 0.01:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-8) / temperature
    logits -= logits.max()
    exp_logits = np.exp(logits)
    temp_probs = exp_logits / exp_logits.sum()
    return int(np.random.choice(NUM_COMBOS, p=temp_probs))


def play_game(
    mcts_p1: MCTS,
    mcts_p2: MCTS,
    sim: BattleSimulator,
    gd: GameData,
    team1: list,
    team2: list,
    preview_eval: PreviewEvaluator | None = None,
    build_examples_p1: list | None = None,
    build_examples_p2: list | None = None,
    n_simulations: int = 200,
    temp_threshold: int = 10,
    temp_high: float = 1.0,
    temp_low: float = 0.1,
    preview_temperature: float = 1.0,
) -> tuple[list[TrainingExample], list[PreviewTrainingExample], int]:
    """한 게임 셀프플레이 → (배틀 examples, 프리뷰 examples, winner).

    흐름:
      1. 팀 프리뷰 (6마리→3마리 선출)
      2. 배틀 (기존과 동일)

    게임 종료 후: winner 기준 value target 할당 (+1/-1).
    빌드 examples가 제공되면 게임 결과로 value 채움.
    """
    preview_examples_raw = []  # (state_enc, policy, player)

    # ── 1단계: 팀 프리뷰 ──
    if preview_eval and len(team1) == 6 and len(team2) == 6:
        # P1: team1 vs team2
        combo1_probs, _ = preview_eval.evaluate(team1, team2)
        combo1_idx = _sample_combo(combo1_probs, preview_temperature)
        selected1 = list(COMBO_TABLE[combo1_idx])

        # P2: team2 vs team1
        combo2_probs, _ = preview_eval.evaluate(team2, team1)
        combo2_idx = _sample_combo(combo2_probs, preview_temperature)
        selected2 = list(COMBO_TABLE[combo2_idx])

        # 프리뷰 학습 데이터 기록
        p1_state = encode_preview_state(team1, team2, gd).numpy()
        p1_policy = np.zeros(NUM_COMBOS, dtype=np.float32)
        p1_policy[combo1_idx] = 1.0
        preview_examples_raw.append((p1_state, p1_policy, 0))

        p2_state = encode_preview_state(team2, team1, gd).numpy()
        p2_policy = np.zeros(NUM_COMBOS, dtype=np.float32)
        p2_policy[combo2_idx] = 1.0
        preview_examples_raw.append((p2_state, p2_policy, 1))

        # 3마리 추출
        team1_battle = [team1[i] for i in selected1]
        team2_battle = [team2[i] for i in selected2]
    else:
        # 프리뷰 없이: 팀을 그대로 사용 (3마리 이하)
        team1_battle = team1[:3] if len(team1) > 3 else team1
        team2_battle = team2[:3] if len(team2) > 3 else team2

    # ── 2단계: 배틀 ──
    state = sim.create_battle_state(team1_battle, team2_battle)
    raw_examples = []   # (state_encoded, policy, player)
    max_turns = 100

    for turn in range(max_turns):
        if state.is_terminal:
            break

        # Temperature 스케줄: 초반 탐색, 후반 착취
        temp = temp_high if turn < temp_threshold else temp_low

        # ── P1 MCTS ──
        probs1 = mcts_p1.get_action_probs(
            state, 0, temperature=temp, n_simulations=n_simulations)

        state_enc_1 = encode_state(state, 0, gd).numpy()
        policy1 = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, p in probs1.items():
            if 0 <= a < NUM_ACTIONS:
                policy1[a] = p
        ps = policy1.sum()
        if ps > 0:
            policy1 /= ps
        raw_examples.append((state_enc_1, policy1, 0))

        # ── P2 MCTS ──
        probs2 = mcts_p2.get_action_probs(
            state, 1, temperature=temp, n_simulations=n_simulations)

        state_enc_2 = encode_state(state, 1, gd).numpy()
        policy2 = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, p in probs2.items():
            if 0 <= a < NUM_ACTIONS:
                policy2[a] = p
        ps = policy2.sum()
        if ps > 0:
            policy2 /= ps
        raw_examples.append((state_enc_2, policy2, 1))

        # ── 액션 선택 ──
        a1 = (_sample_action(probs1, temp) if probs1
              else random.choice(
                  sim.get_legal_actions(state, 0) or [0]))
        a2 = (_sample_action(probs2, temp) if probs2
              else random.choice(
                  sim.get_legal_actions(state, 1) or [0]))

        state = sim.step(state, a1, a2)

    # ── 결과 할당 ──
    winner = state.winner if state.is_terminal else -1

    battle_result = []
    for state_enc, policy, player in raw_examples:
        if winner >= 0:
            value = 1.0 if player == winner else -1.0
        else:
            value = 0.0
        battle_result.append(TrainingExample(
            state=state_enc, policy=policy, value=value))

    preview_result = []
    for state_enc, policy, player in preview_examples_raw:
        if winner >= 0:
            value = 1.0 if player == winner else -1.0
        else:
            value = 0.0
        preview_result.append(PreviewTrainingExample(
            state=state_enc, policy=policy, value=value))

    # 빌드 examples value 할당
    if build_examples_p1:
        for ex in build_examples_p1:
            ex.value = (1.0 if winner == 0 else -1.0) if winner >= 0 else 0.0
    if build_examples_p2:
        for ex in build_examples_p2:
            ex.value = (1.0 if winner == 1 else -1.0) if winner >= 0 else 0.0

    return battle_result, preview_result, winner


# ═══════════════════════════════════════════════════════════════
#  Nash 기반 셀프플레이
# ═══════════════════════════════════════════════════════════════

def compute_td_lambda(
    nash_values: list[float],
    terminal_value: float,
    td_lambda: float = 0.8,
    rewards: list[float] | None = None,
) -> list[float]:
    """TD(λ) 가치 타겟 계산 — 역순 반복, 중간 보상 지원.

    rewards가 없으면 기존 방식:
      V_target(t) = (1 - λ) · V(t+1) + λ · G(t+1)

    rewards가 있으면 dense reward 포함:
      V_target(t) = r(t) + (1 - λ) · V(t+1) + λ · G(t+1)

    Args:
        nash_values: 각 턴의 Nash game_value (한쪽 플레이어 관점)
        terminal_value: 게임 결과 (+1/-1/0)
        td_lambda: TD(λ) 파라미터
        rewards: 각 턴의 dense reward (없으면 0)

    Returns:
        targets: 각 턴의 TD(λ) 가치 타겟
    """
    T = len(nash_values)
    if T == 0:
        return []

    targets = [0.0] * T
    G = terminal_value

    for t in range(T - 1, -1, -1):
        r = rewards[t] if rewards else 0.0
        if t + 1 < T:
            next_v = nash_values[t + 1]
        else:
            next_v = terminal_value
        G = r + (1 - td_lambda) * next_v + td_lambda * G
        targets[t] = G

    return targets


def _compute_dense_reward_p0(
    state_before: BattleState,
    state_after: BattleState,
    sim: BattleSimulator,
    gv_before: float = 0.0,
    gv_after: float = 0.0,
    w_dominance: float = 0.15,
    w_action: float = 0.05,
    w_ko: float = 0.10,
) -> float:
    """턴별 dense reward 계산 (P0 관점).

    최우선: 기점 유도 (Δgame_value — 지배적 위치 생성)
    보조: 상대 행동 가짓수 감소, KO 수적 우위
    """
    # 1. 기점 보상: Nash game_value 변화 — 최우선
    dominance_delta = gv_after - gv_before

    # 2. 상대 행동 가짓수 압박
    opp_actions_before = len(sim.get_legal_actions(state_before, 1))
    opp_actions_after = len(sim.get_legal_actions(state_after, 1))
    my_actions_before = len(sim.get_legal_actions(state_before, 0))
    my_actions_after = len(sim.get_legal_actions(state_after, 0))
    action_delta = ((opp_actions_before - opp_actions_after)
                    / max(opp_actions_before, 1)
                    - (my_actions_before - my_actions_after)
                    / max(my_actions_before, 1))

    # 3. KO 차이
    opp_ko = (sum(1 for p in state_after.sides[1].team if p.fainted)
              - sum(1 for p in state_before.sides[1].team if p.fainted))
    my_ko = (sum(1 for p in state_after.sides[0].team if p.fainted)
             - sum(1 for p in state_before.sides[0].team if p.fainted))
    ko_delta = opp_ko - my_ko

    return (w_dominance * dominance_delta
            + w_action * action_delta
            + w_ko * ko_delta)


def play_game_nash(
    nash_solver,
    sim: BattleSimulator,
    gd: GameData,
    team1: list,
    team2: list,
    td_lambda: float = 0.8,
    temp_threshold: int = 10,
    temp_high: float = 1.0,
    temp_low: float = 0.1,
) -> tuple[list[TrainingExample], int]:
    """Nash 균형 기반 한 게임 셀프플레이 → (examples, winner).

    매 턴:
      1. NashSolver로 양쪽 전략 + game_value + CFR regret 계산
      2. 전략에서 액션 샘플
      3. sim.step → dense reward 계산
      4. belief용 관측 기록

    게임 끝:
      Dense reward 포함 TD(λ) 가치 타겟 계산

    Returns:
        examples: TrainingExample 리스트
        winner: 0/1/-1
    """
    state = sim.create_battle_state(team1, team2)
    max_turns = 100
    nash_solver.reset_revealed()

    # raw 데이터: (state_enc, policy, player, nash_value_p0)
    raw = []
    # 턴별 dense reward (P0 관점)
    dense_rewards_p0 = []
    prev_game_value = 0.0  # 이전 턴 game_value (기점 보상용)

    for turn in range(max_turns):
        if state.is_terminal:
            break

        state_before = state.clone()

        # Nash 전략 계산 (양쪽 + game_value + CFR regret + reversal)
        (p1_dict, p2_dict, game_value,
         p1_regret, p2_regret,
         p1_reversal, p2_reversal) = \
            nash_solver.get_nash_strategies_both(state, add_noise=True)

        # 온도 조절
        temp = temp_high if turn < temp_threshold else temp_low

        # P1 기록 — CFR regret + 역전 포텐셜 반영
        state_enc_1 = encode_state(state, 0, gd).numpy()
        policy1 = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, p in p1_dict.items():
            if 0 <= a < NUM_ACTIONS:
                bonus = max(0.0, p1_regret.get(a, 0.0))
                # 지고 있을 때: 양수 reversal이 있는 행동 강화
                rev_bonus = 0.0
                if game_value < 0:
                    rev = p1_reversal.get(a, 0.0)
                    if rev > 0:
                        rev_bonus = rev  # 역전 가능한 수에 보너스
                policy1[a] = p + 0.1 * bonus + 0.2 * rev_bonus
        ps1 = policy1.sum()
        if ps1 > 0:
            policy1 /= ps1
        raw.append((state_enc_1, policy1, 0, game_value))

        # P2 기록
        state_enc_2 = encode_state(state, 1, gd).numpy()
        policy2 = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, p in p2_dict.items():
            if 0 <= a < NUM_ACTIONS:
                bonus = max(0.0, p2_regret.get(a, 0.0))
                rev_bonus = 0.0
                if game_value > 0:  # P2 관점에서 지고 있음
                    rev = p2_reversal.get(a, 0.0)
                    if rev > 0:
                        rev_bonus = rev
                policy2[a] = p + 0.1 * bonus + 0.2 * rev_bonus
        ps2 = policy2.sum()
        if ps2 > 0:
            policy2 /= ps2
        raw.append((state_enc_2, policy2, 1, -game_value))

        # 액션 샘플링
        a1 = (_sample_action(p1_dict, temp) if p1_dict
              else random.choice(sim.get_legal_actions(state, 0) or [0]))
        a2 = (_sample_action(p2_dict, temp) if p2_dict
              else random.choice(sim.get_legal_actions(state, 1) or [0]))

        # 관측 기록 (belief용): 사용된 기술 추적
        if a1 <= 3:  # MOVE1~MOVE4
            active = state.sides[0].active
            if a1 < len(active.moves):
                nash_solver.observe_move(0, state.sides[0].active_idx, active.moves[a1])
        if a2 <= 3:
            active = state.sides[1].active
            if a2 < len(active.moves):
                nash_solver.observe_move(1, state.sides[1].active_idx, active.moves[a2])

        state = sim.step(state, a1, a2)

        # Dense reward (P0 관점)
        reward_p0 = _compute_dense_reward_p0(
            state_before, state, sim,
            gv_before=prev_game_value, gv_after=game_value)
        dense_rewards_p0.append(reward_p0)
        prev_game_value = game_value

    # 게임 결과
    winner = state.winner if state.is_terminal else -1

    # Terminal value
    if winner == 0:
        terminal_p0 = 1.0
    elif winner == 1:
        terminal_p0 = -1.0
    else:
        terminal_p0 = 0.0

    # 플레이어별 분리 → Dense reward 포함 TD(λ)
    p0_indices = [i for i, (_, _, pl, _) in enumerate(raw) if pl == 0]
    p1_indices = [i for i, (_, _, pl, _) in enumerate(raw) if pl == 1]

    p0_nash_values = [raw[i][3] for i in p0_indices]
    p1_nash_values = [raw[i][3] for i in p1_indices]

    # Dense rewards를 플레이어별로 분리 (P1은 부호 반전)
    p0_rewards = [dense_rewards_p0[i] for i in range(len(dense_rewards_p0))
                  if i < len(p0_indices)]
    p1_rewards = [-dense_rewards_p0[i] for i in range(len(dense_rewards_p0))
                  if i < len(p1_indices)]
    # 길이 맞추기
    p0_rewards = p0_rewards[:len(p0_nash_values)]
    p1_rewards = p1_rewards[:len(p1_nash_values)]

    p0_targets = compute_td_lambda(p0_nash_values, terminal_p0, td_lambda,
                                   rewards=p0_rewards)
    p1_targets = compute_td_lambda(p1_nash_values, -terminal_p0, td_lambda,
                                   rewards=p1_rewards)

    # TrainingExample 생성
    examples = []
    p0_idx = 0
    p1_idx = 0
    for state_enc, policy, player, _ in raw:
        if player == 0:
            value = p0_targets[p0_idx]
            p0_idx += 1
        else:
            value = p1_targets[p1_idx]
            p1_idx += 1
        examples.append(TrainingExample(
            state=state_enc, policy=policy,
            value=float(max(-1.0, min(1.0, value)))))

    return examples, winner


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """self_play.py 검증."""
    import time
    print("=== Self-Play 검증 ===\n")

    gd = GameData(device="cpu")
    sim = BattleSimulator(gd)

    # TeamSampler 테스트
    sampler = TeamSampler(gd, "bss", top_n=50)
    print(f"TeamSampler: {len(sampler.pokemon_ids)} 포켓몬")
    team6 = sampler.sample_team(6)
    print(f"샘플 팀 (6마리): {[p.name for p in team6]}")
    team3 = sampler.sample_team(3)
    print(f"샘플 팀 (3마리): {[p.name for p in team3]}")

    # ReplayBuffer 테스트
    buf = ReplayBuffer(max_size=1000)
    dummy = TrainingExample(
        state=np.zeros(STATE_DIM, dtype=np.float32),
        policy=np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS,
        value=1.0)
    buf.add(dummy)
    assert len(buf) == 1

    # PreviewReplayBuffer 테스트
    pbuf = PreviewReplayBuffer(max_size=1000)
    pdummy = PreviewTrainingExample(
        state=np.zeros(PREVIEW_STATE_DIM, dtype=np.float32),
        policy=np.ones(NUM_COMBOS, dtype=np.float32) / NUM_COMBOS,
        value=1.0)
    pbuf.add(pdummy)
    assert len(pbuf) == 1
    print(f"PreviewReplayBuffer: {len(pbuf)} examples")

    # BuildReplayBuffer 테스트
    bbuf = BuildReplayBuffer(max_size=1000)
    bdummy = BuildTrainingExample(
        state=np.zeros(BUILD_STATE_DIM, dtype=np.float32),
        policy=np.ones(N_CANDIDATES, dtype=np.float32) / N_CANDIDATES,
        value=1.0, step=0)
    bbuf.add(bdummy)
    assert len(bbuf) == 1
    bs, bp, bv = bbuf.sample(1)
    print(f"BuildReplayBuffer: {len(bbuf)} examples, "
          f"sample shapes: {bs.shape}, {bp.shape}, {bv.shape}")

    # play_game 테스트 (프리뷰 없이 — 3마리)
    mcts = MCTS(
        game_data=gd, simulator=sim,
        n_simulations=50, rollout_depth=20,
        n_rollout_per_leaf=1, use_parallel=False,
        format_name="bss", max_opp_branches=3,
    )

    team1 = sampler.sample_team(3)
    team2 = sampler.sample_team(3)
    if len(team1) >= 3 and len(team2) >= 3:
        t0 = time.time()
        battle_ex, preview_ex, winner = play_game(
            mcts, mcts, sim, gd, team1, team2,
            n_simulations=50)
        elapsed = time.time() - t0
        print(f"\nplay_game (3v3, no preview): "
              f"{len(battle_ex)} battle, {len(preview_ex)} preview, "
              f"winner={winner} ({elapsed:.1f}s)")
        if battle_ex:
            print(f"  첫 예시: state shape={battle_ex[0].state.shape}, "
                  f"policy shape={battle_ex[0].policy.shape}, "
                  f"value={battle_ex[0].value}")
            buf.add_batch(battle_ex)
            s, p, v = buf.sample(4)
            print(f"  Sample: states={s.shape}, "
                  f"policies={p.shape}, values={v.shape}")

    # play_game 테스트 (프리뷰 포함 — 6마리)
    from neural_net import TeamPreviewNet
    p_model = TeamPreviewNet()
    p_eval = PreviewEvaluator(p_model, gd, device="cpu")

    team1_6 = sampler.sample_team(6)
    team2_6 = sampler.sample_team(6)
    if len(team1_6) >= 6 and len(team2_6) >= 6:
        t0 = time.time()
        battle_ex, preview_ex, winner = play_game(
            mcts, mcts, sim, gd, team1_6, team2_6,
            preview_eval=p_eval,
            n_simulations=50, preview_temperature=1.0)
        elapsed = time.time() - t0
        print(f"\nplay_game (6→3, preview): "
              f"{len(battle_ex)} battle, {len(preview_ex)} preview, "
              f"winner={winner} ({elapsed:.1f}s)")
        if preview_ex:
            print(f"  프리뷰 예시: state shape={preview_ex[0].state.shape}, "
                  f"policy shape={preview_ex[0].policy.shape}, "
                  f"value={preview_ex[0].value}")
            pbuf.add_batch(preview_ex)
            ps, pp, pv = pbuf.sample(2)
            print(f"  Preview sample: states={ps.shape}, "
                  f"policies={pp.shape}, values={pv.shape}")

    print("\n검증 완료!")


if __name__ == "__main__":
    verify()
