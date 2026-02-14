// =============================================================================
// Ballistic Cache Engine — Benchmark
// =============================================================================
// Compares per-tick RK4 simulation vs cached trajectory lookup.
//
// Build:
//   g++ -std=c++17 -O2 -o bench ballistic_benchmark.cpp
//   (MSVC) cl /std:c++17 /O2 /EHsc /Fe:bench.exe ballistic_benchmark.cpp
//
// Run:
//   ./bench            (Linux/macOS)
//   bench.exe          (Windows)
// =============================================================================

#include "ballistic_cache.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <cmath>

using namespace ballistic;
using Clock = std::chrono::high_resolution_clock;

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    // --- Weapon profile: 7.62 NATO ---
    BallisticParams params;
    params.muzzle_velocity  = 900.0f;
    params.drag_coefficient = 0.295f;
    params.bullet_mass      = 0.00980f;
    params.cross_section    = 0.0000456f;
    params.gravity          = 9.81f;

    PrecomputeConfig config;
    config.pitch_steps       = 91;
    config.pitch_min         = 0.0f;
    config.pitch_max         = 90.0f;
    config.wind_speed_steps  = 6;
    config.wind_speed_max    = 10.0f;
    config.wind_angle_steps  = 13;
    config.max_ticks         = 300;
    config.tick_rate         = 30.0f;

    std::printf("========================================\n");
    std::printf("  Ballistic Cache Engine — Benchmark\n");
    std::printf("========================================\n\n");

    std::printf("Weapon: 7.62 NATO  |  Muzzle velocity: %.0f m/s\n", params.muzzle_velocity);
    std::printf("Cache resolution: pitch %d x wind_speed %d x wind_angle %d\n",
                config.pitch_steps, config.wind_speed_steps, config.wind_angle_steps);
    std::printf("Ticks per trajectory: %d  (%.1f s @ %.0f TPS)\n\n",
                config.max_ticks, config.max_flight_time(), config.tick_rate);

    // =======================================================================
    // Phase 1: Precompute
    // =======================================================================
    std::printf("[1] Precomputing cache...\n");
    BallisticCache cache;
    auto t0 = Clock::now();
    cache.precompute(params, config);
    auto t1 = Clock::now();
    std::printf("    Precompute time : %8.1f ms\n", ms_between(t0, t1));
    std::printf("    Table memory    : %8.2f MB\n\n",
                cache.table_memory() / (1024.0 * 1024.0));

    // =======================================================================
    // Phase 2: Generate random shots
    // =======================================================================
    constexpr int NUM_SHOTS = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pitch_dist(5.0f, 85.0f);   // degrees
    std::uniform_real_distribution<float> yaw_dist(0.0f, 360.0f);    // degrees
    std::uniform_real_distribution<float> wind_dist(-5.0f, 5.0f);    // m/s

    struct Shot {
        float dx, dy, dz;
        float wx, wz;
    };
    std::vector<Shot> shots(NUM_SHOTS);
    for (auto& s : shots) {
        float pitch = pitch_dist(rng) * kDegToRad;
        float yaw   = yaw_dist(rng)   * kDegToRad;
        s.dx = std::cos(pitch) * std::cos(yaw);
        s.dy = std::sin(pitch);
        s.dz = std::cos(pitch) * std::sin(yaw);
        s.wx = wind_dist(rng);
        s.wz = wind_dist(rng);
    }

    // =======================================================================
    // Phase 3: Direct (per-tick RK4) simulation — baseline
    // =======================================================================
    std::printf("[2] Direct RK4 simulation (%d shots x %d ticks)...\n",
                NUM_SHOTS, config.max_ticks);

    std::vector<std::vector<TrajectoryPoint>> direct_results(NUM_SHOTS);
    auto t2 = Clock::now();
    for (int i = 0; i < NUM_SHOTS; ++i) {
        direct_results[i] = BallisticCache::simulate_direct(
            params, 0, 0, 0,
            shots[i].dx, shots[i].dy, shots[i].dz,
            shots[i].wx, 0, shots[i].wz,
            config.dt(), config.max_ticks);
    }
    auto t3 = Clock::now();
    double direct_total_ms = ms_between(t2, t3);
    std::printf("    Total time      : %8.2f ms\n", direct_total_ms);

    // =======================================================================
    // Phase 4: Cache fire() — create trajectories
    // =======================================================================
    std::printf("[3] Cache fire() (%d shots)...\n", NUM_SHOTS);

    std::vector<TrajectoryID> ids(NUM_SHOTS);
    auto t4 = Clock::now();
    for (int i = 0; i < NUM_SHOTS; ++i) {
        ids[i] = cache.fire(
            0, 0, 0,
            shots[i].dx, shots[i].dy, shots[i].dz,
            shots[i].wx, 0, shots[i].wz);
    }
    auto t5 = Clock::now();
    double fire_ms = ms_between(t4, t5);
    std::printf("    Total time      : %8.4f ms\n", fire_ms);

    // =======================================================================
    // Phase 5: Cache get_position() — per-tick queries
    // =======================================================================
    int total_queries = NUM_SHOTS * config.ticks_per_entry();
    std::printf("[4] Cache get_position() (%d queries)...\n", total_queries);

    float checksum = 0.0f; // prevent dead-code elimination
    auto t6 = Clock::now();
    for (int i = 0; i < NUM_SHOTS; ++i) {
        for (int tick = 0; tick <= config.max_ticks; ++tick) {
            TrajectoryPoint pt = cache.get_position(ids[i], tick);
            checksum += pt.x;
        }
    }
    auto t7 = Clock::now();
    double query_ms = ms_between(t6, t7);
    std::printf("    Total time      : %8.2f ms\n", query_ms);
    std::printf("    Per query       : %8.1f ns\n",
                query_ms * 1e6 / total_queries);

    // =======================================================================
    // Phase 6: Per-tick simulation comparison (game-loop style)
    // =======================================================================
    std::printf("[5] Per-tick game-loop comparison (%d bullets x %d ticks)...\n",
                NUM_SHOTS, config.max_ticks);

    // Direct: advance each bullet one RK4 step per tick
    std::vector<detail::OdeState> states(NUM_SHOTS);
    for (int i = 0; i < NUM_SHOTS; ++i) {
        float len = std::sqrt(shots[i].dx*shots[i].dx
                            + shots[i].dy*shots[i].dy
                            + shots[i].dz*shots[i].dz);
        states[i] = {
            0, 0, 0,
            params.muzzle_velocity * shots[i].dx / len,
            params.muzzle_velocity * shots[i].dy / len,
            params.muzzle_velocity * shots[i].dz / len
        };
    }

    auto t8 = Clock::now();
    for (int tick = 1; tick <= config.max_ticks; ++tick) {
        for (int i = 0; i < NUM_SHOTS; ++i) {
            states[i] = detail::rk4_step(
                states[i], params,
                shots[i].wx, 0.0f, shots[i].wz,
                config.dt());
        }
    }
    auto t9 = Clock::now();
    double direct_pertick_ms = ms_between(t8, t9);

    // Cache: get_position for each bullet each tick
    float checksum2 = 0.0f;
    auto t10 = Clock::now();
    for (int tick = 1; tick <= config.max_ticks; ++tick) {
        for (int i = 0; i < NUM_SHOTS; ++i) {
            TrajectoryPoint pt = cache.get_position(ids[i], tick);
            checksum2 += pt.x;
        }
    }
    auto t11 = Clock::now();
    double cache_pertick_ms = ms_between(t10, t11);

    std::printf("    Direct (per-tick RK4)  : %8.2f ms\n", direct_pertick_ms);
    std::printf("    Cache  (get_position)  : %8.2f ms\n", cache_pertick_ms);
    std::printf("    Speedup                : %8.1fx\n",
                direct_pertick_ms / cache_pertick_ms);

    // =======================================================================
    // Phase 7: Accuracy comparison
    // =======================================================================
    std::printf("\n[6] Accuracy (cache vs direct RK4)...\n");

    double total_error  = 0.0;
    double max_error    = 0.0;
    double total_vel_err = 0.0;
    double max_vel_err  = 0.0;
    int    comparisons  = 0;

    for (int i = 0; i < NUM_SHOTS; ++i) {
        for (int tick = 0; tick <= config.max_ticks; ++tick) {
            TrajectoryPoint cached = cache.get_position(ids[i], tick);
            const TrajectoryPoint& exact = direct_results[i][tick];

            float ex = cached.x - exact.x;
            float ey = cached.y - exact.y;
            float ez = cached.z - exact.z;
            double pos_err = std::sqrt(ex*ex + ey*ey + ez*ez);

            float evx = cached.vx - exact.vx;
            float evy = cached.vy - exact.vy;
            float evz = cached.vz - exact.vz;
            double vel_err = std::sqrt(evx*evx + evy*evy + evz*evz);

            total_error  += pos_err;
            total_vel_err += vel_err;
            if (pos_err > max_error) max_error = pos_err;
            if (vel_err > max_vel_err) max_vel_err = vel_err;
            ++comparisons;
        }
    }

    std::printf("    Position error (avg)   : %10.4f m\n",
                total_error / comparisons);
    std::printf("    Position error (max)   : %10.4f m\n", max_error);
    std::printf("    Velocity error (avg)   : %10.4f m/s\n",
                total_vel_err / comparisons);
    std::printf("    Velocity error (max)   : %10.4f m/s\n", max_vel_err);

    // =======================================================================
    // Phase 8: check_hit() benchmark
    // =======================================================================
    constexpr int PLAYERS_PER_CHECK = 4;
    int hit_checks = NUM_SHOTS * config.max_ticks * PLAYERS_PER_CHECK;
    std::printf("\n[7] check_hit() (%d checks)...\n", hit_checks);

    int hits = 0;
    auto t12 = Clock::now();
    for (int tick = 1; tick <= config.max_ticks; ++tick) {
        for (int i = 0; i < NUM_SHOTS; ++i) {
            for (int p = 0; p < PLAYERS_PER_CHECK; ++p) {
                // Synthetic player positions (unlikely to hit, just benchmarking)
                float px = static_cast<float>(p * 100 + tick);
                float py = 1.5f;
                float pz = static_cast<float>(i * 10);
                if (cache.check_hit(ids[i], tick, px, py, pz, 0.5f))
                    ++hits;
            }
        }
    }
    auto t13 = Clock::now();
    double hit_ms = ms_between(t12, t13);
    std::printf("    Total time      : %8.2f ms\n", hit_ms);
    std::printf("    Per check       : %8.1f ns\n",
                hit_ms * 1e6 / hit_checks);
    std::printf("    Hits found      : %d\n", hits);

    // =======================================================================
    // Summary
    // =======================================================================
    std::printf("\n========================================\n");
    std::printf("  Summary\n");
    std::printf("========================================\n");
    std::printf("  Precompute          : %8.1f ms  (one-time)\n", ms_between(t0, t1));
    std::printf("  Table memory        : %8.2f MB\n",
                cache.table_memory() / (1024.0 * 1024.0));
    std::printf("  fire() per shot     : %8.1f us\n",
                fire_ms * 1000.0 / NUM_SHOTS);
    std::printf("  get_position()      : %8.1f ns/query\n",
                query_ms * 1e6 / total_queries);
    std::printf("  check_hit()         : %8.1f ns/check\n",
                hit_ms * 1e6 / hit_checks);
    std::printf("  Game-loop speedup   : %8.1fx  (vs per-tick RK4)\n",
                direct_pertick_ms / cache_pertick_ms);
    std::printf("  Avg position error  : %8.4f m\n",
                total_error / comparisons);
    std::printf("  Max position error  : %8.4f m\n", max_error);
    std::printf("  (checksum: %.2f / %.2f)\n", checksum, checksum2);
    std::printf("========================================\n");

    return 0;
}
