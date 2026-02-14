# Ballistic Cache Engine

> RK4 탄도 시뮬레이션을 사전 계산해서 O(1)로 조회하는 header-only C++ 라이브러리.

## 결론부터

아이디어는 좋았다. 반복되는 결정론적 연산을 캐싱한다는 발상, yaw를 canonical frame으로 제거해서 14GB → 60MB로 줄인 수학적 기교도 마음에 들었다. 근데 RK4 자체가 별로 안 비싸다. 틱당 연산량이 캐싱을 정당화할 만큼 크지 않다. 쓸모 없으면 거기까지다.

UE5에 그대로 들어가긴 한다. header-only, STL only, 엔진 수정 없음.

## 문제

FPS 서버에서 총알 궤적을 매 틱마다 RK4로 시뮬레이션한다. 근데 탄도는 결정론적이다 — 같은 입력이면 같은 포물선이 나온다. 매 틱 다시 계산하는 건 낭비다.

## 해결

시작 시 (pitch, 풍속, 풍향) 조합을 전부 사전 계산해둔다. 런타임에 `fire()`가 보간 파라미터를 O(1)로 풀고, `get_position()`이 trilinear 보간으로 아무 틱의 위치를 반환한다. ODE 풀이 없음.

**핵심 최적화**: yaw를 canonical frame(총알이 XY 평면에서 발사)으로 제거한다. 런타임에 월드 좌표로 회전만 하면 된다. 캐시 크기가 ~14GB → ~60MB로 줄어든다.

## API

| 메서드 | 복잡도 | 설명 |
|--------|--------|------|
| `precompute(params, config)` | O(N) | 룩업 테이블 구축 (시작 시 1회) |
| `fire(origin, dir, wind)` | O(1) | 발사체 등록, `TrajectoryID` 반환 |
| `get_position(id, tick)` | O(1) | trilinear 보간으로 위치+속도 반환 |
| `check_hit(id, tick, pos, radius)` | O(1) | 구체 히트 판정 |
| `simulate_direct(...)` | O(ticks) | 직접 RK4 시뮬레이션 (검증용) |

## 메모리

기본 설정: `91 × 6 × 13 × 301 × 28 bytes ≈ 60 MB` (무기 타입당).

활성 발사체 하나당 48 bytes (보간 파라미터만 저장, 궤적 복사 없음).

## 물리 모델

```
dv/dt = -g·ĵ - (ρ·Cd·A)/(2m) · |v_rel| · v_rel
v_rel = v - v_wind
```

`dt = 1/tick_rate`로 RK4 적분.

## 한계

- pitch 범위: 기본 0°–90° (설정 가능)
- 맵 충돌 없음 (순수 궤적 엔진)
- 보간 오차: 최대 사거리에서 ~0.1–2m (해상도에 따라 다름)
- 바람의 수직 성분은 canonical frame에서 무시됨

---
---

# Ballistic Cache Engine (English)

Header-only C++ library that precomputes ballistic trajectories for O(1) runtime lookup. Designed for FPS game servers where thousands of projectiles need per-tick position queries.

## Bottom Line

The idea was good. Caching deterministic computations, eliminating yaw via canonical frame to cut 14 GB → 60 MB — clever stuff. But RK4 itself is cheap. Per-tick cost doesn't justify the cache. If it's not useful, that's that.

Drops straight into UE5 though. Header-only, STL only, no engine modifications.

## Problem

FPS games simulate bullet trajectories every tick with RK4 integration. But ballistics are deterministic — same input always produces the same arc. Per-tick simulation is wasted compute.

## Solution

At startup, precompute trajectories for all (pitch, wind_speed, wind_angle) combinations. At runtime, `fire()` resolves interpolation parameters in O(1), and `get_position()` returns any tick's position via trilinear interpolation — no ODE solving needed.

**Key optimization**: Yaw is eliminated by computing in a canonical frame (bullet fires in the XY plane). At runtime, results are rotated to world space. This reduces the cache from ~14 GB to ~60 MB per weapon.

## Quick Start

```cpp
#include "ballistic_cache.h"
using namespace ballistic;

// 1. Configure weapon
BallisticParams params;
params.muzzle_velocity  = 900.0f;   // m/s
params.drag_coefficient = 0.295f;
params.bullet_mass      = 0.00980f; // kg
params.cross_section    = 0.0000456f;

// 2. Precompute (once at startup)
BallisticCache cache;
cache.precompute(params);

// 3. Fire a shot
TrajectoryID id = cache.fire(
    0, 1.5f, 0,           // origin
    0.8f, 0.1f, 0.2f,     // direction (auto-normalized)
    2.0f, 0, -1.0f         // wind velocity
);

// 4. Query position at any tick
TrajectoryPoint pt = cache.get_position(id, 150);  // tick 150

// 5. Hit detection
bool hit = cache.check_hit(id, 150,
    pt.x + 0.3f, pt.y, pt.z,  // player position
    0.5f                        // hitbox radius
);
```

## Build & Run Benchmark

```bash
# GCC / Clang
g++ -std=c++17 -O2 -o bench ballistic_benchmark.cpp
./bench

# MSVC
cl /std:c++17 /O2 /EHsc /Fe:bench.exe ballistic_benchmark.cpp
bench.exe
```

## API

### `BallisticParams`
| Field | Default | Description |
|---|---|---|
| `muzzle_velocity` | 900 | Initial speed (m/s) |
| `drag_coefficient` | 0.295 | Aerodynamic Cd |
| `bullet_mass` | 0.00980 | Mass (kg) |
| `cross_section` | 0.0000456 | Frontal area (m²) |
| `gravity` | 9.81 | Gravitational acceleration (m/s²) |

### `PrecomputeConfig`
| Field | Default | Description |
|---|---|---|
| `pitch_steps` | 91 | Elevation angle resolution (0°–90°) |
| `wind_speed_steps` | 6 | Wind magnitude steps (0–10 m/s) |
| `wind_angle_steps` | 13 | Wind direction steps (0°–180°) |
| `max_ticks` | 300 | Max trajectory length |
| `tick_rate` | 30 | Server tick rate (Hz) |

### `BallisticCache`
| Method | Complexity | Description |
|---|---|---|
| `precompute(params, config)` | O(N) | Build lookup table (one-time) |
| `fire(origin, dir, wind)` | O(1) | Register projectile, returns `TrajectoryID` |
| `get_position(id, tick)` | O(1) | Position + velocity at tick via trilinear interpolation |
| `check_hit(id, tick, pos, radius)` | O(1) | Sphere hit test |
| `simulate_direct(...)` | O(ticks) | Direct RK4 simulation (for validation) |

## Memory

Default config: `91 × 6 × 13 × 301 × 28 bytes ≈ 60 MB` per weapon type.

Each active projectile costs 48 bytes (interpolation parameters only — no trajectory copy).

## Physics Model

Quadratic drag with relative wind velocity:

```
dv/dt = -g·ĵ - (ρ·Cd·A)/(2m) · |v_rel| · v_rel
v_rel = v - v_wind
```

Integrated with RK4 at `dt = 1/tick_rate`.

## UE5 Integration

1. Copy `ballistic_cache.h` into your Source directory
2. Include it in your projectile manager:
   ```cpp
   #include "ballistic_cache.h"
   ```
3. Create a `UBallisticSubsystem` that calls `precompute()` in `Initialize()` and exposes `fire()`/`get_position()` to gameplay code
4. No engine modifications needed — pure C++ with STL only

## Limitations

- Pitch range: 0°–90° by default (configurable via `PrecomputeConfig`)
- No map collision (pure trajectory engine)
- Interpolation introduces small errors (~0.1–2 m at max range depending on resolution)
- Wind vertical component is ignored in the canonical frame
