// =============================================================================
// Ballistic Cache Engine — Header-Only C++ Library (Unreal-Ready)
// =============================================================================
// Precomputes ballistic trajectories and caches them for O(1) lookup.
// Instead of per-tick RK4 simulation, fire() resolves the full trajectory
// from a precomputed table via trilinear interpolation.
//
// Key optimization: yaw is eliminated by computing in a canonical frame
// (bullet fires in the XY plane) and rotating at runtime.
//
// Cache dimensions: pitch(91) x wind_speed(6) x wind_angle(13) x ticks(301)
// Memory per weapon: ~60 MB
// =============================================================================
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

// Unreal Engine compatibility: avoid name collisions
#ifdef __UNREAL__
#include "CoreMinimal.h"
#endif

namespace ballistic {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr float kPi         = 3.14159265358979323846f;
constexpr float kDegToRad   = kPi / 180.0f;
constexpr float kRadToDeg   = 180.0f / kPi;
constexpr float kAirDensity = 1.225f; // kg/m^3 at sea level, 15 C

// ---------------------------------------------------------------------------
// TrajectoryID — handle returned by fire()
// ---------------------------------------------------------------------------
using TrajectoryID = uint32_t;
constexpr TrajectoryID INVALID_TRAJECTORY = UINT32_MAX;

// ---------------------------------------------------------------------------
// BallisticParams — per-weapon ballistic profile
// ---------------------------------------------------------------------------
struct BallisticParams {
    float muzzle_velocity  = 900.0f;      // m/s  (7.62 NATO ~850)
    float drag_coefficient = 0.295f;      // Cd   (typical rifle bullet)
    float bullet_mass      = 0.00980f;    // kg   (9.8 g)
    float cross_section    = 0.0000456f;  // m^2  (7.62 mm diameter circle)
    float gravity          = 9.81f;       // m/s^2
};

// ---------------------------------------------------------------------------
// TrajectoryPoint — single sample on a trajectory
// ---------------------------------------------------------------------------
struct TrajectoryPoint {
    float time;           // seconds since fire
    float x, y, z;        // position (m)
    float vx, vy, vz;     // velocity (m/s) — for damage calc

    float speed() const {
        return std::sqrt(vx * vx + vy * vy + vz * vz);
    }
};

// ---------------------------------------------------------------------------
// PrecomputeConfig — controls cache resolution / memory
// ---------------------------------------------------------------------------
struct PrecomputeConfig {
    // Pitch: elevation angle (0 = horizontal, 90 = straight up)
    int   pitch_steps  = 91;
    float pitch_min    = 0.0f;   // degrees
    float pitch_max    = 90.0f;  // degrees

    // Wind speed magnitude (m/s)
    int   wind_speed_steps = 6;
    float wind_speed_max   = 10.0f;

    // Wind angle relative to firing direction (0 = tailwind, 180 = headwind)
    int   wind_angle_steps = 13;

    // Tick budget
    int   max_ticks = 300;
    float tick_rate = 30.0f;  // ticks per second

    float dt()              const { return 1.0f / tick_rate; }
    float max_flight_time() const { return max_ticks / tick_rate; }
    int   ticks_per_entry() const { return max_ticks + 1; }  // includes t=0
};

// ===========================================================================
// detail — ODE solver internals (RK4 with quadratic drag)
// ===========================================================================
namespace detail {

struct OdeState {
    float x, y, z;
    float vx, vy, vz;
};

struct OdeDerivative {
    float dx, dy, dz;
    float dvx, dvy, dvz;
};

// Compute derivatives: gravity + quadratic drag (relative to wind)
inline OdeDerivative compute_derivatives(
    const OdeState& s,
    const BallisticParams& p,
    float wind_x, float wind_y, float wind_z)
{
    // velocity relative to the air mass
    float rvx = s.vx - wind_x;
    float rvy = s.vy - wind_y;
    float rvz = s.vz - wind_z;
    float rel_speed = std::sqrt(rvx * rvx + rvy * rvy + rvz * rvz);

    // drag factor: (rho * Cd * A) / (2 * m) * |v_rel|
    float k = (kAirDensity * p.drag_coefficient * p.cross_section)
              / (2.0f * p.bullet_mass);
    float drag = k * rel_speed;

    return {
        s.vx,  s.vy,  s.vz,
        -drag * rvx,
        -p.gravity - drag * rvy,
        -drag * rvz
    };
}

// Single RK4 step
inline OdeState rk4_step(
    const OdeState& s,
    const BallisticParams& p,
    float wx, float wy, float wz,
    float dt)
{
    auto advance = [](const OdeState& s, const OdeDerivative& d, float h) -> OdeState {
        return {
            s.x  + d.dx  * h, s.y  + d.dy  * h, s.z  + d.dz  * h,
            s.vx + d.dvx * h, s.vy + d.dvy * h, s.vz + d.dvz * h
        };
    };

    auto k1 = compute_derivatives(s, p, wx, wy, wz);
    auto k2 = compute_derivatives(advance(s, k1, dt * 0.5f), p, wx, wy, wz);
    auto k3 = compute_derivatives(advance(s, k2, dt * 0.5f), p, wx, wy, wz);
    auto k4 = compute_derivatives(advance(s, k3, dt),        p, wx, wy, wz);

    float h6 = dt / 6.0f;
    return {
        s.x  + h6 * (k1.dx  + 2*k2.dx  + 2*k3.dx  + k4.dx),
        s.y  + h6 * (k1.dy  + 2*k2.dy  + 2*k3.dy  + k4.dy),
        s.z  + h6 * (k1.dz  + 2*k2.dz  + 2*k3.dz  + k4.dz),
        s.vx + h6 * (k1.dvx + 2*k2.dvx + 2*k3.dvx + k4.dvx),
        s.vy + h6 * (k1.dvy + 2*k2.dvy + 2*k3.dvy + k4.dvy),
        s.vz + h6 * (k1.dvz + 2*k2.dvz + 2*k3.dvz + k4.dvz)
    };
}

// Simulate a full trajectory in the canonical frame.
// Canonical frame: bullet fires in the XY plane, direction = (cos(pitch), sin(pitch), 0).
// Wind is expressed in the same frame.
inline void simulate_trajectory(
    TrajectoryPoint* dest,
    int max_ticks,
    const BallisticParams& params,
    float pitch_rad,
    float wind_speed,
    float wind_angle_rad,
    float dt)
{
    float cos_p = std::cos(pitch_rad);
    float sin_p = std::sin(pitch_rad);

    float wx = wind_speed * std::cos(wind_angle_rad);
    float wz = wind_speed * std::sin(wind_angle_rad);

    OdeState s = {
        0.0f, 0.0f, 0.0f,
        params.muzzle_velocity * cos_p,
        params.muzzle_velocity * sin_p,
        0.0f
    };

    dest[0] = { 0.0f, s.x, s.y, s.z, s.vx, s.vy, s.vz };

    for (int tick = 1; tick <= max_ticks; ++tick) {
        s = rk4_step(s, params, wx, 0.0f, wz, dt);
        float t = tick * dt;
        dest[tick] = { t, s.x, s.y, s.z, s.vx, s.vy, s.vz };
    }
}

} // namespace detail

// ===========================================================================
// BallisticCache — the main engine
// ===========================================================================
class BallisticCache {
public:
    BallisticCache() = default;

    // -----------------------------------------------------------------------
    // precompute() — build the trajectory lookup table
    // -----------------------------------------------------------------------
    void precompute(const BallisticParams& params,
                    const PrecomputeConfig& config = PrecomputeConfig{})
    {
        params_          = params;
        config_          = config;
        ticks_per_entry_ = config.ticks_per_entry();

        int total_entries = config.pitch_steps
                          * config.wind_speed_steps
                          * config.wind_angle_steps;
        table_.resize(static_cast<size_t>(total_entries) * ticks_per_entry_);

        float dt = config.dt();

        for (int ip = 0; ip < config.pitch_steps; ++ip) {
            float pitch_deg = pitch_for_index(ip);
            float pitch_rad = pitch_deg * kDegToRad;

            for (int iw = 0; iw < config.wind_speed_steps; ++iw) {
                float ws = wind_speed_for_index(iw);

                for (int ia = 0; ia < config.wind_angle_steps; ++ia) {
                    float wa_deg = wind_angle_for_index(ia);
                    float wa_rad = wa_deg * kDegToRad;

                    TrajectoryPoint* dest = &table_[flat_index(ip, iw, ia, 0)];
                    detail::simulate_trajectory(
                        dest, config.max_ticks,
                        params, pitch_rad, ws, wa_rad, dt);
                }
            }
        }

        precomputed_ = true;
    }

    // -----------------------------------------------------------------------
    // fire() — register a new projectile, returns a TrajectoryID
    //          The trajectory is resolved lazily in get_position().
    // -----------------------------------------------------------------------
    TrajectoryID fire(
        float origin_x, float origin_y, float origin_z,
        float dir_x,    float dir_y,    float dir_z,
        float wind_x = 0.0f, float wind_y = 0.0f, float wind_z = 0.0f)
    {
        if (!precomputed_) return INVALID_TRAJECTORY;

        // Normalise direction
        float len = std::sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
        if (len < 1e-8f) return INVALID_TRAJECTORY;
        dir_x /= len;  dir_y /= len;  dir_z /= len;

        // Decompose direction into pitch + yaw
        float horiz = std::sqrt(dir_x*dir_x + dir_z*dir_z);
        float pitch_rad = std::atan2(dir_y, horiz);
        float yaw_rad   = std::atan2(dir_z, dir_x);

        float cos_yaw = std::cos(yaw_rad);
        float sin_yaw = std::sin(yaw_rad);

        // Rotate wind into canonical frame (rotate by -yaw around Y)
        float cwx =  wind_x * cos_yaw + wind_z * sin_yaw;
        float cwz = -wind_x * sin_yaw + wind_z * cos_yaw;
        // (vertical wind component ignored in canonical drag model)

        // Polar decomposition of canonical wind
        float ws = std::sqrt(cwx * cwx + cwz * cwz);
        float wa_deg = 0.0f;
        bool  mirror = false;

        if (ws > 1e-6f) {
            // wind_angle in [0, 180]: atan2(|cwz|, cwx)
            wa_deg = std::atan2(std::abs(cwz), cwx) * kRadToDeg;
            if (cwz < 0.0f) mirror = true;
        }

        // Clamp to table bounds
        float pitch_deg = pitch_rad * kRadToDeg;
        pitch_deg = std::clamp(pitch_deg, config_.pitch_min, config_.pitch_max);
        ws        = std::clamp(ws,        0.0f, config_.wind_speed_max);
        wa_deg    = std::clamp(wa_deg,    0.0f, 180.0f);

        // Fractional indices
        float fp = (config_.pitch_steps > 1)
            ? (pitch_deg - config_.pitch_min)
              / (config_.pitch_max - config_.pitch_min) * (config_.pitch_steps - 1)
            : 0.0f;
        float fw = (config_.wind_speed_steps > 1)
            ? ws / config_.wind_speed_max * (config_.wind_speed_steps - 1)
            : 0.0f;
        float fa = (config_.wind_angle_steps > 1)
            ? wa_deg / 180.0f * (config_.wind_angle_steps - 1)
            : 0.0f;

        ActiveTrajectory at;
        at.ip0 = std::clamp(static_cast<int>(fp), 0, std::max(0, config_.pitch_steps - 2));
        at.iw0 = std::clamp(static_cast<int>(fw), 0, std::max(0, config_.wind_speed_steps - 2));
        at.ia0 = std::clamp(static_cast<int>(fa), 0, std::max(0, config_.wind_angle_steps - 2));
        at.tp  = (config_.pitch_steps > 1)      ? fp - at.ip0 : 0.0f;
        at.tw  = (config_.wind_speed_steps > 1)  ? fw - at.iw0 : 0.0f;
        at.ta  = (config_.wind_angle_steps > 1)  ? fa - at.ia0 : 0.0f;

        at.origin_x = origin_x;
        at.origin_y = origin_y;
        at.origin_z = origin_z;
        at.cos_yaw  = cos_yaw;
        at.sin_yaw  = sin_yaw;
        at.mirror_z  = mirror;

        TrajectoryID id = static_cast<TrajectoryID>(trajectories_.size());
        trajectories_.push_back(at);
        return id;
    }

    // -----------------------------------------------------------------------
    // get_position() — O(1) trilinear interpolation + rotation
    // -----------------------------------------------------------------------
    TrajectoryPoint get_position(TrajectoryID id, int tick) const
    {
        if (id >= trajectories_.size())          return {};
        if (tick < 0 || tick >= ticks_per_entry_) return {};

        const auto& t = trajectories_[id];

        // Fetch the 8 surrounding table entries for this tick
        const auto& p000 = table_[flat_index(t.ip0,     t.iw0,     t.ia0,     tick)];
        const auto& p100 = table_[flat_index(t.ip0 + 1, t.iw0,     t.ia0,     tick)];
        const auto& p010 = table_[flat_index(t.ip0,     t.iw0 + 1, t.ia0,     tick)];
        const auto& p110 = table_[flat_index(t.ip0 + 1, t.iw0 + 1, t.ia0,     tick)];
        const auto& p001 = table_[flat_index(t.ip0,     t.iw0,     t.ia0 + 1, tick)];
        const auto& p101 = table_[flat_index(t.ip0 + 1, t.iw0,     t.ia0 + 1, tick)];
        const auto& p011 = table_[flat_index(t.ip0,     t.iw0 + 1, t.ia0 + 1, tick)];
        const auto& p111 = table_[flat_index(t.ip0 + 1, t.iw0 + 1, t.ia0 + 1, tick)];

        // Trilinear interpolation using pointer-to-member
        auto trilerp = [&](float TrajectoryPoint::*f) -> float {
            float c00 = (p000.*f) + t.tp * ((p100.*f) - (p000.*f));
            float c10 = (p010.*f) + t.tp * ((p110.*f) - (p010.*f));
            float c01 = (p001.*f) + t.tp * ((p101.*f) - (p001.*f));
            float c11 = (p011.*f) + t.tp * ((p111.*f) - (p011.*f));
            float c0  = c00 + t.tw * (c10 - c00);
            float c1  = c01 + t.tw * (c11 - c01);
            return c0  + t.ta * (c1  - c0);
        };

        float cx  = trilerp(&TrajectoryPoint::x);
        float cy  = trilerp(&TrajectoryPoint::y);
        float cz  = trilerp(&TrajectoryPoint::z);
        float cvx = trilerp(&TrajectoryPoint::vx);
        float cvy = trilerp(&TrajectoryPoint::vy);
        float cvz = trilerp(&TrajectoryPoint::vz);

        // Mirror Z for negative canonical wind-z
        if (t.mirror_z) { cz = -cz;  cvz = -cvz; }

        // Rotate canonical frame -> world frame (yaw around Y)
        float wx  = cx  * t.cos_yaw - cz  * t.sin_yaw + t.origin_x;
        float wy  = cy  + t.origin_y;
        float wz  = cx  * t.sin_yaw + cz  * t.cos_yaw + t.origin_z;
        float wvx = cvx * t.cos_yaw - cvz * t.sin_yaw;
        float wvy = cvy;
        float wvz = cvx * t.sin_yaw + cvz * t.cos_yaw;

        return { tick * config_.dt(), wx, wy, wz, wvx, wvy, wvz };
    }

    // -----------------------------------------------------------------------
    // check_hit() — sphere-vs-point hit test at a given tick
    // -----------------------------------------------------------------------
    bool check_hit(TrajectoryID id, int tick,
                   float px, float py, float pz,
                   float hitbox_radius) const
    {
        TrajectoryPoint pt = get_position(id, tick);
        float dx = pt.x - px;
        float dy = pt.y - py;
        float dz = pt.z - pz;
        return (dx*dx + dy*dy + dz*dz) <= (hitbox_radius * hitbox_radius);
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Remove all active trajectories (does not touch the precomputed table)
    void clear_trajectories() { trajectories_.clear(); }

    /// Number of active trajectories
    size_t trajectory_count() const { return trajectories_.size(); }

    /// Table memory in bytes
    size_t table_memory() const {
        return table_.size() * sizeof(TrajectoryPoint);
    }

    /// Total memory (table + active trajectories)
    size_t memory_usage() const {
        return table_memory()
             + trajectories_.size() * sizeof(ActiveTrajectory);
    }

    bool              is_precomputed() const { return precomputed_; }
    const PrecomputeConfig& config()   const { return config_; }
    const BallisticParams&  params()   const { return params_; }

    // -----------------------------------------------------------------------
    // simulate_direct() — per-tick RK4 in world space (for benchmarking)
    // -----------------------------------------------------------------------
    static std::vector<TrajectoryPoint> simulate_direct(
        const BallisticParams& params,
        float ox, float oy, float oz,
        float dx, float dy, float dz,
        float wx, float wy, float wz,
        float dt, int max_ticks)
    {
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len < 1e-8f) return {};
        dx /= len;  dy /= len;  dz /= len;

        detail::OdeState s = {
            ox, oy, oz,
            params.muzzle_velocity * dx,
            params.muzzle_velocity * dy,
            params.muzzle_velocity * dz
        };

        std::vector<TrajectoryPoint> result;
        result.reserve(max_ticks + 1);
        result.push_back({ 0.0f, s.x, s.y, s.z, s.vx, s.vy, s.vz });

        for (int tick = 1; tick <= max_ticks; ++tick) {
            s = detail::rk4_step(s, params, wx, wy, wz, dt);
            result.push_back({ tick * dt, s.x, s.y, s.z, s.vx, s.vy, s.vz });
        }
        return result;
    }

private:
    // Lazy-evaluation trajectory descriptor (48 bytes)
    struct ActiveTrajectory {
        float origin_x, origin_y, origin_z;
        float cos_yaw, sin_yaw;
        bool  mirror_z;
        int   ip0, iw0, ia0;   // base table indices
        float tp,  tw,  ta;    // interpolation fractions [0,1)
    };

    // --- table helpers ---

    int flat_index(int ip, int iw, int ia, int tick) const {
        return ((ip * config_.wind_speed_steps + iw)
                    * config_.wind_angle_steps + ia)
                    * ticks_per_entry_ + tick;
    }

    float pitch_for_index(int i) const {
        if (config_.pitch_steps <= 1) return config_.pitch_min;
        return config_.pitch_min
             + i * (config_.pitch_max - config_.pitch_min) / (config_.pitch_steps - 1);
    }

    float wind_speed_for_index(int i) const {
        if (config_.wind_speed_steps <= 1) return 0.0f;
        return i * config_.wind_speed_max / (config_.wind_speed_steps - 1);
    }

    float wind_angle_for_index(int i) const {
        if (config_.wind_angle_steps <= 1) return 0.0f;
        return i * 180.0f / (config_.wind_angle_steps - 1);
    }

    // --- data ---
    BallisticParams   params_;
    PrecomputeConfig  config_;
    int               ticks_per_entry_ = 0;
    bool              precomputed_     = false;

    std::vector<TrajectoryPoint>   table_;          // flat 4-D table
    std::vector<ActiveTrajectory> trajectories_;  // active projectiles
};

} // namespace ballistic
