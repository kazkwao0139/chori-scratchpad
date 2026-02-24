"""Showdown Sim Bridge — Node.js showdown_sim_server.js와 JSON IPC 통신.

Python subprocess로 Node.js 프로세스를 관리.
동기 호출로 init/fork/step/get_request/destroy 제공.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading


class ShowdownBridge:
    """Node.js Showdown sim 서버와 IPC 통신."""

    def __init__(self):
        server_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "showdown_sim_server.js",
        )
        self.proc = subprocess.Popen(
            ["node", server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self._lock = threading.Lock()

        # stderr에서 ready 메시지 확인 (비동기로 읽기)
        # 실패해도 진행 가능 — 첫 명령에서 블로킹됨
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        """stderr 로그를 조용히 소비."""
        try:
            while True:
                line = self.proc.stderr.readline()
                if not line:
                    break
        except Exception:
            pass

    def _call(self, cmd: dict) -> dict:
        """동기 IPC: JSON 한 줄 보내고 응답 한 줄 받기."""
        with self._lock:
            line = json.dumps(cmd) + "\n"
            self.proc.stdin.write(line.encode("utf-8"))
            self.proc.stdin.flush()
            resp_line = self.proc.stdout.readline().decode("utf-8")
            if not resp_line:
                raise RuntimeError("showdown_sim_server.js 프로세스가 종료됨")
            return json.loads(resp_line)

    # ─── 공개 API ────────────────────────────────────────────

    def ping(self) -> dict:
        """서버 상태 확인."""
        return self._call({"type": "ping"})

    def init_battle(self, p1team: str = "", p2team: str = "",
                    format_id: str = "gen9bssregj") -> dict:
        """새 배틀 생성.

        Args:
            p1team: packed team string (Showdown 형식)
            p2team: packed team string
            format_id: 배틀 포맷

        Returns:
            {battle_id, p1request, p2request, ended}
        """
        return self._call({
            "type": "init",
            "p1team": p1team,
            "p2team": p2team,
            "format": format_id,
        })

    def init_from_json(self, battle_json) -> dict:
        """JSON에서 배틀 복원.

        Returns:
            {battle_id, p1request, p2request, ended}
        """
        return self._call({
            "type": "init_from_json",
            "battle_json": battle_json,
        })

    def fork(self, battle_id: str) -> dict:
        """배틀 복제 (toJSON → fromJSON).

        Returns:
            {new_battle_id}
        """
        return self._call({
            "type": "fork",
            "battle_id": battle_id,
        })

    def step(self, battle_id: str, p1choice: str, p2choice: str) -> dict:
        """턴 진행.

        Args:
            battle_id: 배틀 ID
            p1choice: p1 선택 (예: "move 1", "switch 2")
            p2choice: p2 선택

        Returns:
            {p1request, p2request, ended, winner}
        """
        return self._call({
            "type": "step",
            "battle_id": battle_id,
            "p1choice": p1choice,
            "p2choice": p2choice,
        })

    def get_state(self, battle_id: str) -> dict:
        """현재 상태 직렬화 (toJSON).

        Returns:
            {battle_json}
        """
        return self._call({
            "type": "get_state",
            "battle_id": battle_id,
        })

    def get_request(self, battle_id: str, player: int) -> dict | None:
        """한쪽 request 조회.

        Args:
            player: 0 (p1) or 1 (p2)

        Returns:
            {request} — request 객체 또는 None
        """
        resp = self._call({
            "type": "get_request",
            "battle_id": battle_id,
            "player": player,
        })
        return resp.get("request")

    def destroy(self, battle_id: str) -> bool:
        """배틀 인스턴스 정리.

        Returns:
            True if destroyed, False if not found
        """
        resp = self._call({
            "type": "destroy",
            "battle_id": battle_id,
        })
        return resp.get("ok", False)

    def close(self):
        """Node.js 프로세스 종료."""
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def __del__(self):
        self.close()


# ═══════════════════════════════════════════════════════════════
#  검증
# ═══════════════════════════════════════════════════════════════

def verify():
    """ShowdownBridge 검증."""
    print("=== ShowdownBridge 검증 ===\n")

    bridge = ShowdownBridge()

    # 1. ping
    resp = bridge.ping()
    print(f"1. ping: {resp}")
    assert resp.get("pong"), "ping 실패"

    # 2. init (random battle)
    resp = bridge.init_battle(format_id="gen9randombattle")
    bid = resp["battle_id"]
    print(f"2. init: battle_id={bid}, ended={resp['ended']}")
    assert resp["p1request"] is not None, "p1request 없음"
    assert resp["p2request"] is not None, "p2request 없음"

    # 3. get_request
    p1req = bridge.get_request(bid, 0)
    p2req = bridge.get_request(bid, 1)
    print(f"3. get_request: p1 moves={len(p1req['active'][0]['moves'])}, "
          f"p2 moves={len(p2req['active'][0]['moves'])}")

    # 4. fork
    fork_resp = bridge.fork(bid)
    fork_id = fork_resp["new_battle_id"]
    print(f"4. fork: new_battle_id={fork_id}")

    # 5. step (원본에서)
    step_resp = bridge.step(bid, "move 1", "move 1")
    print(f"5. step: ended={step_resp['ended']}, "
          f"winner='{step_resp.get('winner', '')}'")

    # 6. fork된 배틀은 독립적 — step 전 상태 유지
    fork_req = bridge.get_request(fork_id, 0)
    print(f"6. fork 독립성: fork된 배틀 p1 request 존재={fork_req is not None}")

    # 7. destroy
    bridge.destroy(bid)
    bridge.destroy(fork_id)
    print(f"7. destroy: 완료")

    # 8. 여러 턴 진행 테스트
    resp = bridge.init_battle(format_id="gen9randombattle")
    game_id = resp["battle_id"]
    turns = 0
    while not resp.get("ended", False) and turns < 50:
        # 첫 번째 합법 수 선택
        p1r = resp.get("p1request") or bridge.get_request(game_id, 0)
        p2r = resp.get("p2request") or bridge.get_request(game_id, 1)
        if not p1r or not p2r:
            break

        p1c = _pick_first_action(p1r)
        p2c = _pick_first_action(p2r)
        resp = bridge.step(game_id, p1c, p2c)
        turns += 1

    print(f"8. 전체 게임: {turns}턴, ended={resp.get('ended')}, "
          f"winner='{resp.get('winner', '')}'")
    bridge.destroy(game_id)

    bridge.close()
    print("\n검증 완료!")


def _pick_first_action(request: dict) -> str:
    """request에서 첫 번째 합법 수 선택."""
    if not request:
        return "default"

    # forceSwitch인 경우
    if request.get("forceSwitch"):
        side = request.get("side", {})
        for i, poke in enumerate(side.get("pokemon", [])):
            if not poke.get("active") and poke.get("condition", "0 fnt") != "0 fnt":
                return f"switch {i + 1}"
        return "default"

    # wait인 경우
    if request.get("wait"):
        return "default"

    # 일반 턴
    if "active" in request:
        moves = request["active"][0].get("moves", [])
        for i, m in enumerate(moves):
            if not m.get("disabled"):
                return f"move {i + 1}"

    return "default"


if __name__ == "__main__":
    verify()
