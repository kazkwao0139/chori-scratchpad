# Pokemon Battle AI

Gen 9 BSS (Battle Stadium Singles) AI.
Nash equilibrium search + rule-based Stockfish-style evaluation.

## Requirements

- Python 3.10+
- PyTorch (CUDA optional)
- Node.js 18+

```bash
pip install torch numpy poke-env
```

## Quick Start

### 1. Start Showdown Server

```bash
cd pokemon-showdown
npm install
node pokemon-showdown start --no-security
```

Server runs at `http://localhost:8000`.

### 2. Run AI

```bash
python play_showdown.py
```

Open `http://localhost:8000`, challenge **MacroNash** to `[Gen 9] BSS Reg I`.

### 3. Offline Simulation (no server needed)

```bash
python demo_game.py
```

## File Structure

```
play_showdown.py        Entry point — Showdown connection
agent.py                NashPlayer — team preview, poke-env bridge
nash_solver.py          Nash equilibrium solver + iterative deepening
endgame_solver.py       Deep endgame search (2v2, 1v1)
rule_evaluator.py       Position evaluator (12 components, macro-aware)
macro_search.py         Macro strategy detection (setup_sweep / break_clean / cycle)
battle_sim.py           Turn simulator (priority, speed, damage, effects)
damage_calc.py          Exact damage calculator
data_loader.py          Pokedex, moves, abilities, items, Smogon sets
```

## How It Works

1. **Team Preview** — 20x20 matchup matrix, Nash equilibrium selection
2. **Macro Strategy** — Classify team archetype, adjust evaluation weights per phase
3. **Per-Turn Search** — Payoff matrix (my moves x opponent moves) via simulator → Nash mixed strategy
4. **Iterative Deepening** — Time-budgeted, goes deeper in unclear positions

## License

MIT
