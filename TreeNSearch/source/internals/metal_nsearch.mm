// ─────────────────────────────────────────────────────────────────────────────
// Metal GPU backend implementation (Apple Silicon).  Objective-C++.
// See metal_nsearch.h for the algorithm overview and the C++ interface.
// ─────────────────────────────────────────────────────────────────────────────
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_nsearch.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>

namespace {

// ── Metal Shading Language source, compiled once at runtime ──────────────────
static const char* kKernelSource = R"METAL(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// All-32-bit-scalar layout so it matches the C++ struct byte-for-byte (no padding).
struct Params {
    float ox, oy, oz;     // grid origin (min corner)
    float inv_cs;         // 1 / cell_size
    int   dx, dy, dz;     // grid dimensions
    int   R;              // cell search range per axis
    int   symmetric;      // 1 -> include neighbour if d2 <= r_j^2 as well
    int   setOffsetI;     // global index of first point of the searching set
    int   setOffsetJ;     // global index of first point of the neighbour set
    int   setI, setJ;     // set ids
    int   nSearch;        // number of searching points
    int   nCells;         // dx*dy*dz
    int   totalPoints;    // points across all sets
    float r2;             // squared radius (fast uniform-radius path only)
    int   hasBoundary;    // SPH phases: add per-particle boundary (Bender2019) term
    float dt;             // timestep h (pressure/divergence solve)
    float density0;       // rest density (pressure/divergence solve)
    float aij_scale;      // aij_pj multiplier: h^2 (pressure) or h (divergence)
    int   divergence;     // 1 -> divergence mode (s_i=-densityAdv, deficiency cutoff)
    float viscosity;      // XSPH velocity-smoothing strength (0 = off)
};

static inline int cell_index(float3 p, constant Params& P)
{
    int cx = (int)floor((p.x - P.ox) * P.inv_cs);
    int cy = (int)floor((p.y - P.oy) * P.inv_cs);
    int cz = (int)floor((p.z - P.oz) * P.inv_cs);
    cx = clamp(cx, 0, P.dx - 1);
    cy = clamp(cy, 0, P.dy - 1);
    cz = clamp(cz, 0, P.dz - 1);
    return (cz * P.dy + cy) * P.dx + cx;
}

// 1. Hash every point into a cell and accumulate per-cell counts.
// Points are stored padded to float4 so each load is a single coalesced vector read.
kernel void k_hash(device const float4*       points  [[buffer(0)]],
                   device uint*               cellId  [[buffer(1)]],
                   device atomic_uint*        counts  [[buffer(2)]],
                   constant Params&           P       [[buffer(3)]],
                   uint                       gid     [[thread_position_in_grid]])
{
    if ((int)gid >= P.totalPoints) return;
    int lin = cell_index(points[gid].xyz, P);
    cellId[gid] = (uint)lin;
    atomic_fetch_add_explicit(&counts[lin], 1u, memory_order_relaxed);
}

// 2. Counting-sort scatter. As well as placing each point in its cell's slot, we
// reorder the point payload (position / radius / set / original index) into cell
// order so the neighbour sweep reads contiguous memory instead of gathering
// through an index indirection — the key optimisation for dense neighbourhoods.
kernel void k_scatter(device const uint*      cellId       [[buffer(0)]],
                      device atomic_uint*      cellOffset   [[buffer(1)]],
                      device const float4*     points       [[buffer(2)]],
                      device const float*      radii        [[buffer(3)]],
                      device const int*        pointSet     [[buffer(4)]],
                      device float4*           sortedPoints [[buffer(5)]],
                      device float*            sortedRadii  [[buffer(6)]],
                      device int*              sortedSet    [[buffer(7)]],
                      device int*              sortedOrig   [[buffer(8)]],
                      constant Params&         P            [[buffer(9)]],
                      uint                     gid          [[thread_position_in_grid]])
{
    if ((int)gid >= P.totalPoints) return;
    uint lin  = cellId[gid];
    uint slot = atomic_fetch_add_explicit(&cellOffset[lin], 1u, memory_order_relaxed);
    sortedPoints[slot] = points[gid];
    sortedRadii[slot]  = radii[gid];
    sortedSet[slot]    = pointSet[gid];
    sortedOrig[slot]   = (int)gid;
}

// Shared neighbour sweep. Returns the neighbour count; if `out` is non-null also
// writes the set_j-local neighbour ids starting at out[0].
static inline int sweep(uint i,
                        device const float4* points,        // query points, original order
                        device const float*  radii,         // query radii,  original order
                        device const float4* sortedPoints,  // neighbour payload, cell order
                        device const float*  sortedRadii,
                        device const int*    sortedSet,
                        device const int*    sortedOrig,
                        device const uint*   cellStart,
                        constant Params&     P,
                        device int*          out)   // null in the count pass
{
    int gid_i = P.setOffsetI + (int)i;
    float3 pi = points[gid_i].xyz;
    float ri = radii[gid_i];
    float r2i = ri * ri;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    int count = 0;
    for (int dz = -P.R; dz <= P.R; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -P.R; dy <= P.R; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -P.R; dx <= P.R; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {       // contiguous, coalesced reads
                    if (sortedSet[s] != P.setJ) continue;
                    int gid_j = sortedOrig[s];
                    if (gid_j == gid_i) continue;          // a point is never its own neighbour
                    float3 e = pi - sortedPoints[s].xyz;
                    float d2 = dot(e, e);
                    bool hit = d2 <= r2i;
                    if (!hit && P.symmetric != 0) {
                        float rj = sortedRadii[s];
                        hit = d2 <= rj * rj;
                    }
                    if (hit) {
                        if (out != nullptr) {
                            out[count] = gid_j - P.setOffsetJ;   // set_j-local index
                        }
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

// 3a. Count pass: number of neighbours per searching point.
kernel void k_count(device const float4* points       [[buffer(0)]],
                    device const float*  radii        [[buffer(1)]],
                    device const float4* sortedPoints [[buffer(2)]],
                    device const float*  sortedRadii  [[buffer(3)]],
                    device const int*    sortedSet    [[buffer(4)]],
                    device const int*    sortedOrig   [[buffer(5)]],
                    device const uint*   cellStart    [[buffer(6)]],
                    device uint*         outCount     [[buffer(7)]],
                    constant Params&     P            [[buffer(8)]],
                    uint                 i            [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    outCount[i] = (uint)sweep(i, points, radii, sortedPoints, sortedRadii,
                              sortedSet, sortedOrig, cellStart, P, nullptr);
}

// 3b. Write pass: emit packed [count, ids...] runs using the host-computed offsets.
kernel void k_write(device const float4* points       [[buffer(0)]],
                    device const float*  radii        [[buffer(1)]],
                    device const float4* sortedPoints [[buffer(2)]],
                    device const float*  sortedRadii  [[buffer(3)]],
                    device const int*    sortedSet    [[buffer(4)]],
                    device const int*    sortedOrig   [[buffer(5)]],
                    device const uint*   cellStart    [[buffer(6)]],
                    device int*          block        [[buffer(7)]],
                    device const int*    blockOffset  [[buffer(8)]],
                    constant Params&     P            [[buffer(9)]],
                    uint                 i            [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int base = blockOffset[i];
    int count = sweep(i, points, radii, sortedPoints, sortedRadii,
                      sortedSet, sortedOrig, cellStart, P, block + base + 1);
    block[base] = count;
}

// ─── Fast path: uniform radius + single point set ─────────────────────────────
// The whole neighbour payload is a single float4 per slot: xyz = position,
// w = original index (bit-cast). The sweep reads ONE array and compares against a
// uniform squared radius, roughly halving the memory traffic of the hot loop.

kernel void k_scatter_fast(device const uint*    cellId      [[buffer(0)]],
                           device atomic_uint*    cellOffset  [[buffer(1)]],
                           device const float4*   points      [[buffer(2)]],
                           device float4*         sortedPos   [[buffer(3)]],
                           constant Params&       P           [[buffer(4)]],
                           uint                   gid         [[thread_position_in_grid]])
{
    if ((int)gid >= P.totalPoints) return;
    uint lin  = cellId[gid];
    uint slot = atomic_fetch_add_explicit(&cellOffset[lin], 1u, memory_order_relaxed);
    float4 p = points[gid];
    p.w = as_type<float>((int)gid);   // stash the original index in .w
    sortedPos[slot] = p;
}

static inline int sweep_fast(uint i,
                             device const float4* points,
                             device const float4* sortedPos,
                             device const uint*   cellStart,
                             constant Params&     P,
                             device int*          out)
{
    int gid_i = P.setOffsetI + (int)i;     // setOffsetI == 0 for the single-set case
    float3 pi = points[gid_i].xyz;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    int count = 0;
    for (int dz = -P.R; dz <= P.R; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -P.R; dy <= P.R; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -P.R; dx <= P.R; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 q = sortedPos[s];
                    int gid_j = as_type<int>(q.w);
                    if (gid_j == gid_i) continue;
                    float3 e = pi - q.xyz;
                    float d2 = dot(e, e);
                    if (d2 <= P.r2) {
                        if (out != nullptr) out[count] = gid_j - P.setOffsetJ;
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

// Single-sweep search: do the distance tests ONCE, storing neighbour ids into a
// per-point scratch row (capacity TNS_MAXNB) and recording the true (uncapped)
// count. A later compaction packs the rows — avoiding the expensive second sweep.
#define TNS_MAXNB 128

kernel void k_search_fast(device const float4* points    [[buffer(0)]],
                          device const float4* sortedPos [[buffer(1)]],
                          device const uint*   cellStart [[buffer(2)]],
                          device int*          scratch   [[buffer(3)]],
                          device uint*         outCount  [[buffer(4)]],
                          constant Params&     P         [[buffer(5)]],
                          uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int gid_i = P.setOffsetI + (int)i;
    float3 pi = points[gid_i].xyz;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    device int* sc = scratch + (uint)i * TNS_MAXNB;
    int count = 0;
    for (int dz = -P.R; dz <= P.R; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -P.R; dy <= P.R; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -P.R; dx <= P.R; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 q = sortedPos[s];
                    int gid_j = as_type<int>(q.w);
                    if (gid_j == gid_i) continue;
                    float3 e = pi - q.xyz;
                    if (dot(e, e) <= P.r2) {
                        if (count < TNS_MAXNB) sc[count] = gid_j - P.setOffsetJ;
                        count++;
                    }
                }
            }
        }
    }
    outCount[i] = (uint)count;
}

// Pack scratch rows into the tight [count, ids...] output. The rare point whose
// neighbour count exceeded the scratch capacity is re-swept directly into place.
kernel void k_compact_fast(device const float4* points      [[buffer(0)]],
                           device const float4* sortedPos   [[buffer(1)]],
                           device const uint*   cellStart   [[buffer(2)]],
                           device const int*    scratch     [[buffer(3)]],
                           device const uint*   outCount    [[buffer(4)]],
                           device const int*    blockOffset [[buffer(5)]],
                           device int*          block       [[buffer(6)]],
                           constant Params&     P           [[buffer(7)]],
                           uint                 i           [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int base = blockOffset[i];
    int cnt  = (int)outCount[i];
    block[base] = cnt;
    if (cnt <= TNS_MAXNB) {
        device const int* sc = scratch + (uint)i * TNS_MAXNB;
        for (int k = 0; k < cnt; k++) block[base + 1 + k] = sc[k];
    } else {
        sweep_fast(i, points, sortedPos, cellStart, P, block + base + 1);  // overflow: re-sweep
    }
}

// ─── SPH density (CubicKernel) ────────────────────────────────────────────────
// P.r2 holds h (support radius) here; P.inv_cs = 1/cell_size = 1/h; cells swept R=1.
kernel void k_density(device const float4* points    [[buffer(0)]],
                      device const float4* sortedPos [[buffer(1)]],
                      device const float*  volume    [[buffer(2)]],
                      device const uint*   cellStart [[buffer(3)]],
                      device const float*  bVol      [[buffer(4)]],
                      device const float*  bXj       [[buffer(5)]],
                      device float*        outDensity[[buffer(6)]],
                      constant Params&     P         [[buffer(7)]],
                      uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h  = P.r2;                       // support radius
    const float k  = 8.0f / (3.14159265358979f * h * h * h);   // CubicKernel m_k
    const float W0 = k;                          // W(0) = m_k * 1

    float3 pi = points[i].xyz;
    float density = volume[i] * W0;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 q = sortedPos[s];
                    int gid_j = as_type<int>(q.w);
                    if (gid_j == (int)i) continue;
                    float3 e = pi - q.xyz;
                    float r = length(e);
                    float qd = r / h;
                    float w = 0.0f;
                    if (qd <= 1.0f) {
                        if (qd <= 0.5f) { float q2 = qd*qd; w = k * (6.0f*q2*qd - 6.0f*q2 + 1.0f); }
                        else            { float f = 1.0f - qd; w = k * (2.0f*f*f*f); }
                    }
                    density += volume[gid_j] * w;
                }
            }
        }
    }
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        if (Vb > 0.0f) {
            float3 e = pi - float3(bXj[3*i+0], bXj[3*i+1], bXj[3*i+2]);
            float r = length(e);
            float qd = r / h;
            float w = 0.0f;
            if (qd <= 1.0f) {
                if (qd <= 0.5f) { float q2 = qd*qd; w = k * (6.0f*q2*qd - 6.0f*q2 + 1.0f); }
                else            { float f = 1.0f - qd; w = k * (2.0f*f*f*f); }
            }
            density += Vb * w;
        }
    }
    outDensity[i] = density;
}

// ─── DFSPH factor alpha_i (CubicKernel gradW) ────────────────────────────────
// factor_i = 1 / sum_grad_p_k   (or 0 below eps), where
//   sum_grad_p_k = sum_j |V_j gradW_ij|^2 + | sum_j V_j gradW_ij |^2  (fluid part)
kernel void k_factor(device const float4* points    [[buffer(0)]],
                     device const float4* sortedPos [[buffer(1)]],
                     device const float*  volume    [[buffer(2)]],
                     device const uint*   cellStart [[buffer(3)]],
                     device const float*  bVol      [[buffer(4)]],
                     device const float*  bXj       [[buffer(5)]],
                     device float*        outFactor [[buffer(6)]],
                     constant Params&     P         [[buffer(7)]],
                     uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float l = 48.0f / (3.14159265358979f * h * h * h);   // CubicKernel m_l

    float3 pi = points[i].xyz;
    float3 grad_p_i = float3(0.0f);
    float  sum = 0.0f;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 qp = sortedPos[s];
                    int gid_j = as_type<int>(qp.w);
                    if (gid_j == (int)i) continue;
                    float3 rvec = pi - qp.xyz;
                    float rl = length(rvec);
                    float qd = rl / h;
                    float3 gw = float3(0.0f);
                    if (rl > 1.0e-9f && qd <= 1.0f) {
                        float3 gradq = rvec / (rl * h);
                        if (qd <= 0.5f) gw = l * qd * (3.0f*qd - 2.0f) * gradq;
                        else { float f = 1.0f - qd; gw = l * (-f*f) * gradq; }
                    }
                    float3 Vgw = volume[gid_j] * gw;
                    sum += dot(Vgw, Vgw);
                    grad_p_i += Vgw;
                }
            }
        }
    }
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        if (Vb > 0.0f) {
            float3 rvec = pi - float3(bXj[3*i+0], bXj[3*i+1], bXj[3*i+2]);
            float rl = length(rvec);
            float qd = rl / h;
            if (rl > 1.0e-9f && qd <= 1.0f) {
                float3 gradq = rvec / (rl * h);
                float3 gw = (qd <= 0.5f) ? (l * qd * (3.0f*qd - 2.0f) * gradq)
                                         : (l * (-(1.0f-qd)*(1.0f-qd)) * gradq);
                grad_p_i += Vb * gw;   // grad_p_i -= (-Vb*gradW)
            }
        }
    }
    sum += dot(grad_p_i, grad_p_i);
    outFactor[i] = (sum > 1.0e-6f) ? (1.0f / sum) : 0.0f;
}

// ─── DFSPH density-advection delta (velocity divergence) ─────────────────────
// delta_i = sum_j V_j (v_i - v_j) . gradW_ij  + static-boundary V_b (v_i).gradW
kernel void k_density_adv(device const float4* points    [[buffer(0)]],
                          device const float4* sortedPos [[buffer(1)]],
                          device const float*  volume    [[buffer(2)]],
                          device const float*  velocity  [[buffer(3)]],
                          device const uint*   cellStart [[buffer(4)]],
                          device const float*  bVol      [[buffer(5)]],
                          device const float*  bXj       [[buffer(6)]],
                          device float*        outDelta  [[buffer(7)]],
                          constant Params&     P         [[buffer(8)]],
                          uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float l = 48.0f / (3.14159265358979f * h * h * h);

    float3 pi = points[i].xyz;
    float3 vi = float3(velocity[3*i+0], velocity[3*i+1], velocity[3*i+2]);
    float delta = 0.0f;

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 qp = sortedPos[s];
                    int gid_j = as_type<int>(qp.w);
                    if (gid_j == (int)i) continue;
                    float3 rvec = pi - qp.xyz;
                    float rl = length(rvec);
                    float qd = rl / h;
                    if (rl <= 1.0e-9f || qd > 1.0f) continue;
                    float3 gradq = rvec / (rl * h);
                    float3 gw = (qd <= 0.5f) ? (l * qd * (3.0f*qd - 2.0f) * gradq)
                                             : (l * (-(1.0f-qd)*(1.0f-qd)) * gradq);
                    float3 Vgw = volume[gid_j] * gw;
                    float3 vj = float3(velocity[3*gid_j+0], velocity[3*gid_j+1], velocity[3*gid_j+2]);
                    delta += dot(vi - vj, Vgw);
                }
            }
        }
    }
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        if (Vb > 0.0f) {
            float3 rvec = pi - float3(bXj[3*i+0], bXj[3*i+1], bXj[3*i+2]);
            float rl = length(rvec);
            float qd = rl / h;
            if (rl > 1.0e-9f && qd <= 1.0f) {
                float3 gradq = rvec / (rl * h);
                float3 gw = (qd <= 0.5f) ? (l * qd * (3.0f*qd - 2.0f) * gradq)
                                         : (l * (-(1.0f-qd)*(1.0f-qd)) * gradq);
                delta += Vb * dot(vi, gw);   // static boundary: v_j = 0
            }
        }
    }
    outDelta[i] = delta;
}

// CubicKernel gradW for the pressure-solve kernels.
static inline float3 cubic_gradW(float3 rvec, float h, float l)
{
    float rl = length(rvec);
    float qd = rl / h;
    if (rl <= 1.0e-9f || qd > 1.0f) return float3(0.0f);
    float3 gradq = rvec / (rl * h);
    if (qd <= 0.5f) return l * qd * (3.0f*qd - 2.0f) * gradq;
    float f = 1.0f - qd;
    return l * (-f*f) * gradq;
}

// ─── DFSPH pressure acceleration: a_i = -sum_j (p~_i + p~_j) V_j gradW  (- p~_i V_b gradW) ──
kernel void k_pressure_accel(device const float4* points    [[buffer(0)]],
                             device const float4* sortedPos [[buffer(1)]],
                             device const float*  volume    [[buffer(2)]],
                             device const float*  pRho2     [[buffer(3)]],
                             device const uint*   cellStart [[buffer(4)]],
                             device const float*  bVol      [[buffer(5)]],
                             device const float*  bXj       [[buffer(6)]],
                             device float4*       accel     [[buffer(7)]],
                             constant Params&     P         [[buffer(8)]],
                             uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float l = 48.0f / (3.14159265358979f * h * h * h);
    const float pi_rho2 = pRho2[i];
    float3 xi = points[i].xyz;
    float3 a = float3(0.0f);

    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    for (int dz = -1; dz <= 1; dz++) { int ncz=cz+dz; if(ncz<0||ncz>=P.dz) continue;
    for (int dy = -1; dy <= 1; dy++) { int ncy=cy+dy; if(ncy<0||ncy>=P.dy) continue;
    for (int dx = -1; dx <= 1; dx++) { int ncx=cx+dx; if(ncx<0||ncx>=P.dx) continue;
        int lin = (ncz*P.dy+ncy)*P.dx+ncx;
        for (uint s = cellStart[lin]; s < cellStart[lin+1]; s++) {
            float4 qp = sortedPos[s];
            int gid_j = as_type<int>(qp.w);
            if (gid_j == (int)i) continue;
            float3 Vgw = volume[gid_j] * cubic_gradW(xi - qp.xyz, h, l);
            a -= Vgw * (pi_rho2 + pRho2[gid_j]);
        }
    }}}
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        if (Vb > 0.0f)
            a -= pi_rho2 * Vb * cubic_gradW(xi - float3(bXj[3*i+0],bXj[3*i+1],bXj[3*i+2]), h, l);
    }
    accel[i] = float4(a, 0.0f);
}

// ─── DFSPH Jacobi update: aij_pj = h^2 sum_j V_j (a_i-a_j).gradW (+ V_b a_i.gradW);
//     p~_i = max(p~_i - 0.5 (s_i - aij_pj) factor_i, 0);  err_i = -density0 * min(s_i-aij_pj,0)
kernel void k_pressure_update(device const float4* points    [[buffer(0)]],
                              device const float4* sortedPos [[buffer(1)]],
                              device const float*  volume    [[buffer(2)]],
                              device const float4* accel     [[buffer(3)]],
                              device const uint*   cellStart [[buffer(4)]],
                              device const float*  bVol      [[buffer(5)]],
                              device const float*  bXj       [[buffer(6)]],
                              device const float*  densityAdv[[buffer(7)]],
                              device const float*  factor    [[buffer(8)]],
                              device float*        pRho2     [[buffer(9)]],
                              device atomic_float* errSum    [[buffer(10)]],
                              constant Params&     P         [[buffer(11)]],
                              uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float l = 48.0f / (3.14159265358979f * h * h * h);
    float3 xi = points[i].xyz;
    float3 ai = accel[i].xyz;
    float aij_pj = 0.0f;

    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int nNeigh = 0;
    for (int dz = -1; dz <= 1; dz++) { int ncz=cz+dz; if(ncz<0||ncz>=P.dz) continue;
    for (int dy = -1; dy <= 1; dy++) { int ncy=cy+dy; if(ncy<0||ncy>=P.dy) continue;
    for (int dx = -1; dx <= 1; dx++) { int ncx=cx+dx; if(ncx<0||ncx>=P.dx) continue;
        int lin = (ncz*P.dy+ncy)*P.dx+ncx;
        for (uint s = cellStart[lin]; s < cellStart[lin+1]; s++) {
            float4 qp = sortedPos[s];
            int gid_j = as_type<int>(qp.w);
            if (gid_j == (int)i) continue;
            if (length(xi - qp.xyz) <= h) nNeigh++;
            float3 Vgw = volume[gid_j] * cubic_gradW(xi - qp.xyz, h, l);
            aij_pj += dot(ai - accel[gid_j].xyz, Vgw);
        }
    }}}
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        if (Vb > 0.0f)
            aij_pj += Vb * dot(ai, cubic_gradW(xi - float3(bXj[3*i+0],bXj[3*i+1],bXj[3*i+2]), h, l));
    }
    aij_pj *= P.aij_scale;
    float s_i = (P.divergence != 0) ? (-densityAdv[i]) : (1.0f - densityAdv[i]);
    float residuum = min(s_i - aij_pj, 0.0f);
    pRho2[i] = max(pRho2[i] - 0.5f * (s_i - aij_pj) * factor[i], 0.0f);
    // particle-deficiency: skip divergence error contribution for under-supported particles
    if (P.divergence != 0 && nNeigh < 20) residuum = 0.0f;
    // GPU reduction of the density error: one atomic add instead of an N-float readback.
    atomic_fetch_add_explicit(errSum, -P.density0 * residuum, memory_order_relaxed);
}

// ─── Optimized solve: build a flat neighbour list with precomputed V_j*gradW once
//     per step, then iterate it (no per-iteration grid sweep / kernel evaluation). ──

// Count fluid neighbours (r <= h) per particle.
kernel void k_nl_count(device const float4* points    [[buffer(0)]],
                       device const float4* sortedPos [[buffer(1)]],
                       device const uint*   cellStart [[buffer(2)]],
                       device uint*         counts    [[buffer(3)]],
                       constant Params&     P         [[buffer(4)]],
                       uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    float3 xi = points[i].xyz;
    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int n = 0;
    for (int dz=-1; dz<=1; dz++){ int ncz=cz+dz; if(ncz<0||ncz>=P.dz) continue;
    for (int dy=-1; dy<=1; dy++){ int ncy=cy+dy; if(ncy<0||ncy>=P.dy) continue;
    for (int dx=-1; dx<=1; dx++){ int ncx=cx+dx; if(ncx<0||ncx>=P.dx) continue;
        int lin=(ncz*P.dy+ncy)*P.dx+ncx;
        for (uint s=cellStart[lin]; s<cellStart[lin+1]; s++){
            float4 qp=sortedPos[s]; int j=as_type<int>(qp.w);
            if (j==(int)i) continue;
            if (length(xi-qp.xyz) <= h) n++;
        }
    }}}
    counts[i] = (uint)n;
}

// Fill the neighbour ids + precomputed V_j*gradW_ij into the CSR list, and the
// per-particle boundary term V_b*gradW(x_i - x_jb).
kernel void k_nl_fill(device const float4* points    [[buffer(0)]],
                      device const float4* sortedPos [[buffer(1)]],
                      device const float*  volume    [[buffer(2)]],
                      device const uint*   cellStart [[buffer(3)]],
                      device const uint*   nlStart   [[buffer(4)]],
                      device const float*  bVol      [[buffer(5)]],
                      device const float*  bXj       [[buffer(6)]],
                      device int*          nlId      [[buffer(7)]],
                      device float4*       nlVgw     [[buffer(8)]],
                      device float4*       bVgw      [[buffer(9)]],
                      constant Params&     P         [[buffer(10)]],
                      uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float l = 48.0f / (3.14159265358979f * h * h * h);
    float3 xi = points[i].xyz;
    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    uint base = nlStart[i];
    uint k = base;
    for (int dz=-1; dz<=1; dz++){ int ncz=cz+dz; if(ncz<0||ncz>=P.dz) continue;
    for (int dy=-1; dy<=1; dy++){ int ncy=cy+dy; if(ncy<0||ncy>=P.dy) continue;
    for (int dx=-1; dx<=1; dx++){ int ncx=cx+dx; if(ncx<0||ncx>=P.dx) continue;
        int lin=(ncz*P.dy+ncy)*P.dx+ncx;
        for (uint s=cellStart[lin]; s<cellStart[lin+1]; s++){
            float4 qp=sortedPos[s]; int j=as_type<int>(qp.w);
            if (j==(int)i) continue;
            float3 rvec = xi - qp.xyz;
            if (length(rvec) <= h) {
                nlId[k]  = j;
                nlVgw[k] = float4(volume[j] * cubic_gradW(rvec, h, l), 0.0f);
                k++;
            }
        }
    }}}
    if (P.hasBoundary != 0) {
        float Vb = bVol[i];
        bVgw[i] = (Vb > 0.0f) ? float4(Vb * cubic_gradW(xi - float3(bXj[3*i+0],bXj[3*i+1],bXj[3*i+2]), h, l), 0.0f)
                              : float4(0.0f);
    } else {
        bVgw[i] = float4(0.0f);
    }
}

// Pressure accel from the neighbour list: a_i = -sum (p~_i+p~_j) Vgw - p~_i bVgw.
kernel void k_pa_nl(device const uint*   nlStart [[buffer(0)]],
                    device const int*    nlId    [[buffer(1)]],
                    device const float4* nlVgw   [[buffer(2)]],
                    device const float4* bVgw    [[buffer(3)]],
                    device const float*  pRho2   [[buffer(4)]],
                    device float4*       accel   [[buffer(5)]],
                    constant Params&     P       [[buffer(6)]],
                    uint                 i       [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    float pri = pRho2[i];
    float3 a = float3(0.0f);
    uint b = nlStart[i], e = nlStart[i+1];
    for (uint s = b; s < e; s++) a -= nlVgw[s].xyz * (pri + pRho2[nlId[s]]);
    if (P.hasBoundary != 0) a -= pri * bVgw[i].xyz;
    accel[i] = float4(a, 0.0f);
}

// Jacobi update from the neighbour list.
kernel void k_pu_nl(device const uint*   nlStart [[buffer(0)]],
                    device const int*    nlId    [[buffer(1)]],
                    device const float4* nlVgw   [[buffer(2)]],
                    device const float4* bVgw    [[buffer(3)]],
                    device const float4* accel   [[buffer(4)]],
                    device const float*  densAdv [[buffer(5)]],
                    device const float*  factor  [[buffer(6)]],
                    device float*        pRho2   [[buffer(7)]],
                    device atomic_float* errSum  [[buffer(8)]],
                    constant Params&     P       [[buffer(9)]],
                    uint                 i       [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    float3 ai = accel[i].xyz;
    uint b = nlStart[i], e = nlStart[i+1];
    float aij = 0.0f;
    for (uint s = b; s < e; s++) aij += dot(ai - accel[nlId[s]].xyz, nlVgw[s].xyz);
    if (P.hasBoundary != 0) aij += dot(ai, bVgw[i].xyz);
    aij *= P.aij_scale;
    float s_i = (P.divergence != 0) ? (-densAdv[i]) : (1.0f - densAdv[i]);
    float residuum = min(s_i - aij, 0.0f);
    pRho2[i] = max(pRho2[i] - 0.5f * (s_i - aij) * factor[i], 0.0f);
    if (P.divergence != 0 && (int)(e - b) < 20) residuum = 0.0f;
    atomic_fetch_add_explicit(errSum, -P.density0 * residuum, memory_order_relaxed);
}

// ─── GPU-resident constant-density source term ───────────────────────────────
// densAdv_i = relative density clamped to >= 1 (only compression is corrected);
// non-finite density -> 1 (rest). Mirrors the host caller's CPU clamp so the
// resident solve is numerically identical. density holds rho_i/rho0 here.
kernel void k_densadv_clamp(device const float* density [[buffer(0)]],
                            device float*       densAdv [[buffer(1)]],
                            constant Params&    P       [[buffer(2)]],
                            uint                i       [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    float d = density[i];
    densAdv[i] = (isfinite(d) && d > 1.0f) ? d : 1.0f;
}

// ─── Pack the float4 pressure accel into the caller's tight float3 buffer ─────
kernel void k_accel_copy3(device const float4* accel [[buffer(0)]],
                          device float*        out3  [[buffer(1)]],
                          constant Params&     P     [[buffer(2)]],
                          uint                 i     [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    float3 a = accel[i].xyz;
    out3[3*i+0] = a.x; out3[3*i+1] = a.y; out3[3*i+2] = a.z;
}

// ─── XSPH velocity smoothing (viscosity) ─────────────────────────────────────
// Same grid sweep as k_density, but accumulates Sum_j V_j (v_j - v_i) W and writes
// v_i' = v_i + viscosity * Sum / rho_i. Reads velIn, writes a SEPARATE velOut so the
// sweep always sees pre-smoothing velocities. P.r2 = h, density precomputed in `density`.
kernel void k_xsph(device const float4* points    [[buffer(0)]],
                   device const float4* sortedPos [[buffer(1)]],
                   device const float*  volume    [[buffer(2)]],
                   device const uint*   cellStart [[buffer(3)]],
                   device const float*  velIn     [[buffer(4)]],  // 3/particle
                   device const float*  density   [[buffer(5)]],
                   device float*        velOut    [[buffer(6)]],  // 3/particle
                   constant Params&     P         [[buffer(7)]],
                   uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h = P.r2;
    const float k = 8.0f / (3.14159265358979f * h * h * h);  // CubicKernel m_k

    float3 pi = points[i].xyz;
    float3 vi = float3(velIn[3*i+0], velIn[3*i+1], velIn[3*i+2]);
    float3 acc = float3(0.0f);

    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);

    for (int dz = -1; dz <= 1; dz++) {
        int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = cx + dx; if (ncx < 0 || ncx >= P.dx) continue;
                int lin = (ncz * P.dy + ncy) * P.dx + ncx;
                uint begin = cellStart[lin];
                uint end   = cellStart[lin + 1];
                for (uint s = begin; s < end; s++) {
                    float4 q = sortedPos[s];
                    int gid_j = as_type<int>(q.w);
                    if (gid_j == (int)i) continue;
                    float r = length(pi - q.xyz);
                    float qd = r / h;
                    float w = 0.0f;
                    if (qd <= 1.0f) {
                        if (qd <= 0.5f) { float q2 = qd*qd; w = k * (6.0f*q2*qd - 6.0f*q2 + 1.0f); }
                        else            { float f = 1.0f - qd; w = k * (2.0f*f*f*f); }
                    }
                    float3 vj = float3(velIn[3*gid_j+0], velIn[3*gid_j+1], velIn[3*gid_j+2]);
                    acc += volume[gid_j] * (vj - vi) * w;
                }
            }
        }
    }
    float di = max(density[i], 1.0e-6f);
    float3 vs = vi + P.viscosity * acc / di;
    velOut[3*i+0] = vs.x; velOut[3*i+1] = vs.y; velOut[3*i+2] = vs.z;
}

// ─── GPU exclusive prefix-sum of per-cell counts (grid build) ────────────────
// Single-thread serial scan: writes cellStart[0..nCells] (exclusive prefix; the last
// element is the grand total) and seeds cellOffset[0..nCells) with the same offsets
// (the scatter's running cursor). O(nCells) work on one thread — the point is not to
// parallelise the scan but to keep the grid build entirely GPU-side so hash -> scan ->
// scatter ride one command buffer with no CPU readback between them. nCells here is
// ~1e3-1e4, so the serial cost (a few us) is far below the CPU-sync stall it removes.
kernel void k_scan_cells(device const uint* counts     [[buffer(0)]],
                         device uint*       cellStart   [[buffer(1)]],
                         device uint*       cellOffset  [[buffer(2)]],
                         constant Params&   P           [[buffer(3)]],
                         uint               tid         [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint acc = 0;
    int n = P.nCells;
    for (int c = 0; c < n; c++) { cellStart[c] = acc; cellOffset[c] = acc; acc += counts[c]; }
    cellStart[n] = acc;
}

// ─── GPU exclusive prefix-sum for the CSR neighbour list ─────────────────────
// Exclusive scan of the per-particle neighbour counts (length P.nSearch) into
// outStart (length nSearch+1; outStart[nSearch] = grand total = number of pairs).
// Single serial thread — keeps the CSR build GPU-side so k_nl_count -> scan ->
// k_nl_fill need no CPU readback of the per-particle counts; only the one-word grand
// total is read back afterwards, to size the pair buffers. Bit-identical to the old
// CPU scan (same left-to-right uint32 accumulation); nSearch ~ 1e3-1e4.
kernel void k_exscan(device const uint* count    [[buffer(0)]],
                     device uint*       outStart [[buffer(1)]],
                     constant Params&   P        [[buffer(2)]],
                     uint               tid      [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint acc = 0;
    int n = P.nSearch;
    for (int i = 0; i < n; i++) { outStart[i] = acc; acc += count[i]; }
    outStart[n] = acc;
}

)METAL";

// ── Cached Metal context ─────────────────────────────────────────────────────
struct MetalContext {
    id<MTLDevice>             device   = nil;
    id<MTLCommandQueue>       queue    = nil;
    id<MTLComputePipelineState> psoHash    = nil;
    id<MTLComputePipelineState> psoScatter = nil;
    id<MTLComputePipelineState> psoCount   = nil;
    id<MTLComputePipelineState> psoWrite   = nil;
    id<MTLComputePipelineState> psoScatterFast = nil;
    id<MTLComputePipelineState> psoSearchFast  = nil;
    id<MTLComputePipelineState> psoCompactFast = nil;
    id<MTLComputePipelineState> psoDensity     = nil;
    id<MTLComputePipelineState> psoFactor      = nil;
    id<MTLComputePipelineState> psoDensityAdv  = nil;
    id<MTLComputePipelineState> psoPressureAccel  = nil;
    id<MTLComputePipelineState> psoPressureUpdate = nil;
    id<MTLComputePipelineState> psoNlCount = nil;
    id<MTLComputePipelineState> psoNlFill  = nil;
    id<MTLComputePipelineState> psoPaNl    = nil;
    id<MTLComputePipelineState> psoPuNl    = nil;
    id<MTLComputePipelineState> psoDensadvClamp = nil;
    id<MTLComputePipelineState> psoAccelCopy3   = nil;
    id<MTLComputePipelineState> psoXsph         = nil;
    id<MTLComputePipelineState> psoScanCells    = nil;   // GPU grid prefix-sum
    id<MTLComputePipelineState> psoExscan       = nil;   // GPU neighbour-list prefix-sum (CSR)
    bool ok = false;
    std::string error;
};

// Optional caller-injected Metal context (the host app's own device + command queue)
// so the resident solve can bind caller buffers directly. nil => use the default device.
// Set via metal_sph_set_external_context(); a change forces context() to rebuild.
static id<MTLDevice>       g_extDevice = nil;
static id<MTLCommandQueue> g_extQueue  = nil;
static bool                g_contextNeedsRebuild = false;
static void reset_buffers();   // defined with the buffer cache below

// Build the device, queue and all compute pipelines into ctx. dev/queue may be nil,
// in which case the system default device (or first available) and a fresh queue are
// used. Leaves ctx.ok == false (with ctx.error set) on any failure.
static void build_context(MetalContext& ctx, id<MTLDevice> dev, id<MTLCommandQueue> queue)
{
    ctx = MetalContext{};
    @autoreleasepool {
        ctx.device = dev;
        if (ctx.device == nil) {
            ctx.device = MTLCreateSystemDefaultDevice();
            if (ctx.device == nil) {
                // MTLCreateSystemDefaultDevice() returns nil without a window-server
                // session (e.g. headless / over SSH); the device is still reachable here.
                NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
                if (all.count > 0) ctx.device = all.firstObject;
            }
        }
        if (ctx.device == nil) { ctx.error = "No Metal device available."; return; }

        ctx.queue = queue ? queue : [ctx.device newCommandQueue];
        if (ctx.queue == nil) { ctx.error = "Failed to create Metal command queue."; return; }

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kKernelSource];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:opts error:&err];
        if (lib == nil) {
            ctx.error = std::string("Metal shader compilation failed: ") +
                        (err ? err.localizedDescription.UTF8String : "unknown");
            return;
        }

        auto make_pso = [&](const char* name, __strong id<MTLComputePipelineState>& pso) -> bool {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (fn == nil) { ctx.error = std::string("Missing kernel: ") + name; return false; }
            NSError* e = nil;
            pso = [ctx.device newComputePipelineStateWithFunction:fn error:&e];
            if (pso == nil) {
                ctx.error = std::string("Pipeline build failed for ") + name + ": " +
                            (e ? e.localizedDescription.UTF8String : "unknown");
                return false;
            }
            return true;
        };

        if (!make_pso("k_hash",    ctx.psoHash))    return;
        if (!make_pso("k_scatter", ctx.psoScatter)) return;
        if (!make_pso("k_count",   ctx.psoCount))   return;
        if (!make_pso("k_write",   ctx.psoWrite))   return;
        if (!make_pso("k_scatter_fast", ctx.psoScatterFast)) return;
        if (!make_pso("k_search_fast",  ctx.psoSearchFast))  return;
        if (!make_pso("k_compact_fast", ctx.psoCompactFast)) return;
        if (!make_pso("k_density",      ctx.psoDensity))     return;
        if (!make_pso("k_factor",       ctx.psoFactor))      return;
        if (!make_pso("k_density_adv",  ctx.psoDensityAdv))  return;
        if (!make_pso("k_pressure_accel",  ctx.psoPressureAccel))  return;
        if (!make_pso("k_pressure_update", ctx.psoPressureUpdate)) return;
        if (!make_pso("k_nl_count", ctx.psoNlCount)) return;
        if (!make_pso("k_nl_fill",  ctx.psoNlFill))  return;
        if (!make_pso("k_pa_nl",    ctx.psoPaNl))    return;
        if (!make_pso("k_pu_nl",    ctx.psoPuNl))    return;
        if (!make_pso("k_densadv_clamp", ctx.psoDensadvClamp)) return;
        if (!make_pso("k_accel_copy3",   ctx.psoAccelCopy3))   return;
        if (!make_pso("k_xsph",          ctx.psoXsph))         return;
        if (!make_pso("k_scan_cells",      ctx.psoScanCells))     return;
        if (!make_pso("k_exscan",          ctx.psoExscan))        return;

        ctx.ok = true;
    }
}

static MetalContext& context()
{
    static MetalContext ctx;
    static bool initialized = false;
    if (initialized && !g_contextNeedsRebuild) return ctx;
    initialized = true;
    g_contextNeedsRebuild = false;
    build_context(ctx, g_extDevice, g_extQueue);
    return ctx;
}

// C++ mirror of the MSL Params struct (identical 32-bit-scalar layout).
struct ParamsC {
    float ox, oy, oz;
    float inv_cs;
    int   dx, dy, dz;
    int   R;
    int   symmetric;
    int   setOffsetI;
    int   setOffsetJ;
    int   setI, setJ;
    int   nSearch;
    int   nCells;
    int   totalPoints;
    float r2;
    int   hasBoundary;
    float dt;
    float density0;
    float aij_scale;
    int   divergence;
    float viscosity;   // mirrors Params (MSL); XSPH strength, 0 = off
};

static void dispatch1d(id<MTLComputeCommandEncoder> enc,
                       id<MTLComputePipelineState> pso, NSUInteger n)
{
    NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
    if (tpg > n) tpg = (n == 0 ? 1 : n);
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// ── Persistent, grow-only GPU buffer cache ───────────────────────────────────
// Reused across calls so the SPH loop (same point sets every step) does not
// reallocate Metal buffers each frame. Not thread-safe: assumes one search at a
// time, which is how the neighbourhood search is driven.
struct GpuBuffers {
    id<MTLBuffer> points, radii, pointSet, cellId, counts, cellStart, cellOffset;
    id<MTLBuffer> sPoints, sRadii, sSet, sOrig;
    id<MTLBuffer> count, block, blockOff, scratch;
    id<MTLBuffer> volume, density, bVol, bXj, velocity;
    id<MTLBuffer> velSmooth;                   // XSPH double-buffer (velOut)
    id<MTLBuffer> pRho2, accel, densAdv, factorBuf, err;
    id<MTLBuffer> nlStart, nlId, nlVgw, bVgw;   // precomputed neighbour list for the solve
    std::vector<uint32_t> cellStartHost, nlStartHost;
    std::vector<uint32_t> nlCountHost;
};

static GpuBuffers& buffers()
{
    static GpuBuffers b;
    return b;
}

static const int kMaxNb = 128;   // must match TNS_MAXNB in the shader


// Cached grid params from the most recent build_grid_fast(). Within one solver
// step the particle positions are fixed across density/factor/solve phases, so the
// grid (sortedPos + cellStart in the persistent buffers) can be reused instead of
// rebuilt each phase. Density rebuilds it; later phases reuse it.
static ParamsC g_gridP{};
static bool    g_gridValid = false;
// Cached neighbour list (built once per step, shared by the divergence + pressure solves).
static bool    g_nlValid = false;
static int     g_nlTotal = 0;

// Drop every cached GPU buffer — used when the Metal device changes (the old buffers
// belong to the old device). ARC releases them; the next phase reallocates on the new
// device via ensure(). Also clears the grid/neighbour-list caches.
static void reset_buffers()
{
    buffers() = GpuBuffers{};
    g_gridValid = false;
    g_nlValid   = false;
}

// Grow the buffer only when the requested size exceeds the current allocation.
static void ensure(id<MTLDevice> dev, __strong id<MTLBuffer>& buf, size_t bytes)
{
    if (bytes == 0) bytes = 4;
    if (buf == nil || (size_t)buf.length < bytes) {
        buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
}

} // anonymous namespace

namespace tns { namespace internals {

bool metal_is_available()
{
    return context().ok;
}

void metal_sph_invalidate_grid()
{
    g_gridValid = false;
    g_nlValid = false;
}

bool metal_neighbor_search(const MetalSearchRequest& req,
                           MetalSearchResult& out,
                           std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }

    const int N = req.total_points;
    const long nCellsL = (long)req.grid_dims[0] * (long)req.grid_dims[1] * (long)req.grid_dims[2];
    if (nCellsL <= 0 || nCellsL > 300000000L) {
        error = "Grid cell count out of supported range.";
        return false;
    }
    const int nCells = (int)nCellsL;

    static const bool kProfile = std::getenv("TNS_PROFILE") != nullptr;
    auto pnow = []{ return std::chrono::high_resolution_clock::now(); };
    auto pms  = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
    double tFill = 0, tHash = 0, tScan = 0, tScatter = 0, tCount = 0, tWrite = 0;
    auto pt0 = pnow();

    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        GpuBuffers& B = buffers();

        // Persistent (grow-only) buffers — no per-call allocation in steady state.
        ensure(dev, B.points,     sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.radii,      sizeof(float) * (size_t)N);
        ensure(dev, B.pointSet,   sizeof(int) * (size_t)N);
        ensure(dev, B.cellId,     sizeof(uint32_t) * (size_t)N);
        ensure(dev, B.counts,     sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.cellStart,  sizeof(uint32_t) * (size_t)(nCells + 1));
        ensure(dev, B.cellOffset, sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.sPoints,    sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.sRadii,     sizeof(float) * (size_t)N);
        ensure(dev, B.sSet,       sizeof(int) * (size_t)N);
        ensure(dev, B.sOrig,      sizeof(int) * (size_t)N);

        // Fast path: one radius for everyone + a single point set (typical SPH fluid).
        const bool fast = req.uniform_radius && req.n_sets == 1;

        // Fill inputs directly into shared (unified-memory) buffers.
        {
            float* p4 = (float*)B.points.contents;
            for (int p = 0; p < N; p++) {
                p4[4 * (size_t)p + 0] = req.points[3 * (size_t)p + 0];
                p4[4 * (size_t)p + 1] = req.points[3 * (size_t)p + 1];
                p4[4 * (size_t)p + 2] = req.points[3 * (size_t)p + 2];
                p4[4 * (size_t)p + 3] = 0.0f;
            }
            if (!fast) {   // radii / per-point set ids are only used by the general kernels
                std::memcpy(B.radii.contents, req.radii, sizeof(float) * (size_t)N);
                int* ps = (int*)B.pointSet.contents;
                for (int s = 0; s < req.n_sets; s++) {
                    for (int g = req.set_offsets[s]; g < req.set_offsets[s + 1]; g++) ps[g] = s;
                }
            }
            std::memset(B.counts.contents, 0, sizeof(uint32_t) * (size_t)nCells);
        }
        tFill = pms(pt0, pnow());

        ParamsC P{};
        P.ox = req.origin[0]; P.oy = req.origin[1]; P.oz = req.origin[2];
        P.inv_cs = 1.0f / req.cell_size;
        P.dx = req.grid_dims[0]; P.dy = req.grid_dims[1]; P.dz = req.grid_dims[2];
        P.R = req.search_range;
        P.symmetric = req.symmetric ? 1 : 0;
        P.nCells = nCells;
        P.totalPoints = N;
        P.r2 = req.radius * req.radius;

        // ── 1. hash ──────────────────────────────────────────────────────────
        auto th0 = pnow();
        {
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:B.points offset:0 atIndex:0];
            [enc setBuffer:B.cellId offset:0 atIndex:1];
            [enc setBuffer:B.counts offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            dispatch1d(enc, ctx.psoHash, (NSUInteger)N);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        tHash = pms(th0, pnow());

        // ── 2. exclusive scan of counts -> cellStart (size nCells+1) on CPU ───
        auto ts0 = pnow();
        const uint32_t* counts = (const uint32_t*)B.counts.contents;
        B.cellStartHost.resize((size_t)nCells + 1);
        uint32_t acc = 0;
        for (int c = 0; c < nCells; c++) { B.cellStartHost[c] = acc; acc += counts[c]; }
        B.cellStartHost[nCells] = acc;
        std::memcpy(B.cellStart.contents,  B.cellStartHost.data(), sizeof(uint32_t) * (size_t)(nCells + 1));
        std::memcpy(B.cellOffset.contents, B.cellStartHost.data(), sizeof(uint32_t) * (size_t)nCells); // running cursor
        tScan = pms(ts0, pnow());

        // ── 3. scatter (counting sort + payload reorder) ────────────────────────
        auto tsc0 = pnow();
        {
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (fast) {
                [enc setBuffer:B.cellId     offset:0 atIndex:0];
                [enc setBuffer:B.cellOffset offset:0 atIndex:1];
                [enc setBuffer:B.points     offset:0 atIndex:2];
                [enc setBuffer:B.sPoints    offset:0 atIndex:3];
                [enc setBytes:&P length:sizeof(P) atIndex:4];
                dispatch1d(enc, ctx.psoScatterFast, (NSUInteger)N);
            } else {
                [enc setBuffer:B.cellId     offset:0 atIndex:0];
                [enc setBuffer:B.cellOffset offset:0 atIndex:1];
                [enc setBuffer:B.points     offset:0 atIndex:2];
                [enc setBuffer:B.radii      offset:0 atIndex:3];
                [enc setBuffer:B.pointSet   offset:0 atIndex:4];
                [enc setBuffer:B.sPoints    offset:0 atIndex:5];
                [enc setBuffer:B.sRadii     offset:0 atIndex:6];
                [enc setBuffer:B.sSet       offset:0 atIndex:7];
                [enc setBuffer:B.sOrig      offset:0 atIndex:8];
                [enc setBytes:&P length:sizeof(P) atIndex:9];
                dispatch1d(enc, ctx.psoScatter, (NSUInteger)N);
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        tScatter = pms(tsc0, pnow());

        // ── 4. per-pair neighbour search (count pass, CPU scan, write pass) ──────
        out.pairs.clear();
        out.pairs.reserve(req.pairs.size());

        for (const MetalSearchPair& pr : req.pairs) {
            const int nSearch = req.set_offsets[pr.set_i + 1] - req.set_offsets[pr.set_i];

            ParamsC PP = P;
            PP.setI = pr.set_i; PP.setJ = pr.set_j;
            PP.setOffsetI = req.set_offsets[pr.set_i];
            PP.setOffsetJ = req.set_offsets[pr.set_j];
            PP.nSearch = nSearch;

            MetalPairResult res;
            res.set_i = pr.set_i; res.set_j = pr.set_j;
            res.block_offset.resize((size_t)nSearch + 1);

            if (nSearch == 0) { res.block_offset[0] = 0; out.pairs.push_back(std::move(res)); continue; }

            ensure(dev, B.count, sizeof(uint32_t) * (size_t)nSearch);
            if (fast) ensure(dev, B.scratch, sizeof(int) * (size_t)nSearch * (size_t)kMaxNb);

            // pass 1: single sweep (fast = count + store to scratch; general = count only)
            auto tc0 = pnow();
            {
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (fast) {
                    [enc setBuffer:B.points    offset:0 atIndex:0];
                    [enc setBuffer:B.sPoints   offset:0 atIndex:1];
                    [enc setBuffer:B.cellStart offset:0 atIndex:2];
                    [enc setBuffer:B.scratch   offset:0 atIndex:3];
                    [enc setBuffer:B.count     offset:0 atIndex:4];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:5];
                    dispatch1d(enc, ctx.psoSearchFast, (NSUInteger)nSearch);
                } else {
                    [enc setBuffer:B.points  offset:0 atIndex:0];
                    [enc setBuffer:B.radii   offset:0 atIndex:1];
                    [enc setBuffer:B.sPoints offset:0 atIndex:2];
                    [enc setBuffer:B.sRadii  offset:0 atIndex:3];
                    [enc setBuffer:B.sSet    offset:0 atIndex:4];
                    [enc setBuffer:B.sOrig   offset:0 atIndex:5];
                    [enc setBuffer:B.cellStart offset:0 atIndex:6];
                    [enc setBuffer:B.count   offset:0 atIndex:7];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:8];
                    dispatch1d(enc, ctx.psoCount, (NSUInteger)nSearch);
                }
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
            tCount += pms(tc0, pnow());

            // CPU prefix sum -> block offsets ( each run is [count, ids...] -> count+1 ints )
            const uint32_t* cnt = (const uint32_t*)B.count.contents;
            int blockAcc = 0;
            for (int i = 0; i < nSearch; i++) {
                res.block_offset[i] = blockAcc;
                blockAcc += (int)cnt[i] + 1;
            }
            res.block_offset[nSearch] = blockAcc;

            res.block.resize((size_t)blockAcc);
            ensure(dev, B.block,    sizeof(int) * (size_t)blockAcc);
            ensure(dev, B.blockOff, sizeof(int) * (size_t)(nSearch + 1));
            std::memcpy(B.blockOff.contents, res.block_offset.data(), sizeof(int) * (size_t)(nSearch + 1));

            // pass 2: emit packed lists (fast = compact scratch rows; general = write sweep)
            auto tw0 = pnow();
            {
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (fast) {
                    [enc setBuffer:B.points    offset:0 atIndex:0];
                    [enc setBuffer:B.sPoints   offset:0 atIndex:1];
                    [enc setBuffer:B.cellStart offset:0 atIndex:2];
                    [enc setBuffer:B.scratch   offset:0 atIndex:3];
                    [enc setBuffer:B.count     offset:0 atIndex:4];
                    [enc setBuffer:B.blockOff  offset:0 atIndex:5];
                    [enc setBuffer:B.block     offset:0 atIndex:6];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:7];
                    dispatch1d(enc, ctx.psoCompactFast, (NSUInteger)nSearch);
                } else {
                    [enc setBuffer:B.points   offset:0 atIndex:0];
                    [enc setBuffer:B.radii    offset:0 atIndex:1];
                    [enc setBuffer:B.sPoints  offset:0 atIndex:2];
                    [enc setBuffer:B.sRadii   offset:0 atIndex:3];
                    [enc setBuffer:B.sSet     offset:0 atIndex:4];
                    [enc setBuffer:B.sOrig    offset:0 atIndex:5];
                    [enc setBuffer:B.cellStart offset:0 atIndex:6];
                    [enc setBuffer:B.block    offset:0 atIndex:7];
                    [enc setBuffer:B.blockOff offset:0 atIndex:8];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:9];
                    dispatch1d(enc, ctx.psoWrite, (NSUInteger)nSearch);
                }
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            tWrite += pms(tw0, pnow());
            std::memcpy(res.block.data(), B.block.contents, sizeof(int) * (size_t)blockAcc);
            out.pairs.push_back(std::move(res));
        }
    }

    if (kProfile) {
        std::fprintf(stderr,
            "[TNS_PROFILE] N=%d nCells=%d  fill=%.2f hash=%.2f scan=%.2f scatter=%.2f count=%.2f write=%.2f  total=%.2f ms\n",
            N, nCells, tFill, tHash, tScan, tScatter, tCount, tWrite,
            tFill + tHash + tScan + tScatter + tCount + tWrite);
    }
    return true;
}

// Builds the uniform grid (hash + CPU scan + scatter into sPoints with packed
// index) into the persistent buffers and fills `P`. Shared by the SPH phases.
static bool build_grid_fast(MetalContext& ctx, GpuBuffers& B, ParamsC& P,
                            const float* points3, int N,
                            const float origin[3], const int dims[3], float h)
{
    const long nCellsL = (long)dims[0] * (long)dims[1] * (long)dims[2];
    if (N <= 0 || nCellsL <= 0 || nCellsL > 300000000L) return false;
    const int nCells = (int)nCellsL;
    id<MTLDevice> dev = ctx.device;

    ensure(dev, B.points,     sizeof(float) * 4 * (size_t)N);
    ensure(dev, B.cellId,     sizeof(uint32_t) * (size_t)N);
    ensure(dev, B.counts,     sizeof(uint32_t) * (size_t)nCells);
    ensure(dev, B.cellStart,  sizeof(uint32_t) * (size_t)(nCells + 1));
    ensure(dev, B.cellOffset, sizeof(uint32_t) * (size_t)nCells);
    ensure(dev, B.sPoints,    sizeof(float) * 4 * (size_t)N);

    float* p4 = (float*)B.points.contents;
    for (int p = 0; p < N; p++) {
        p4[4*(size_t)p+0] = points3[3*(size_t)p+0];
        p4[4*(size_t)p+1] = points3[3*(size_t)p+1];
        p4[4*(size_t)p+2] = points3[3*(size_t)p+2];
        p4[4*(size_t)p+3] = 0.0f;
    }
    std::memset(B.counts.contents, 0, sizeof(uint32_t) * (size_t)nCells);

    P = ParamsC{};
    P.ox = origin[0]; P.oy = origin[1]; P.oz = origin[2];
    P.inv_cs = 1.0f / h;
    P.dx = dims[0]; P.dy = dims[1]; P.dz = dims[2];
    P.R = 1; P.nCells = nCells; P.totalPoints = N; P.nSearch = N; P.r2 = h;

    { // hash
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:B.points offset:0 atIndex:0];
        [enc setBuffer:B.cellId offset:0 atIndex:1];
        [enc setBuffer:B.counts offset:0 atIndex:2];
        [enc setBytes:&P length:sizeof(P) atIndex:3];
        dispatch1d(enc, ctx.psoHash, (NSUInteger)N);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    }
    const uint32_t* counts = (const uint32_t*)B.counts.contents;
    B.cellStartHost.resize((size_t)nCells + 1);
    uint32_t acc = 0;
    for (int c = 0; c < nCells; c++) { B.cellStartHost[c] = acc; acc += counts[c]; }
    B.cellStartHost[nCells] = acc;
    std::memcpy(B.cellStart.contents,  B.cellStartHost.data(), sizeof(uint32_t)*(size_t)(nCells+1));
    std::memcpy(B.cellOffset.contents, B.cellStartHost.data(), sizeof(uint32_t)*(size_t)nCells);
    { // scatter
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:B.cellId offset:0 atIndex:0];
        [enc setBuffer:B.cellOffset offset:0 atIndex:1];
        [enc setBuffer:B.points offset:0 atIndex:2];
        [enc setBuffer:B.sPoints offset:0 atIndex:3];
        [enc setBytes:&P length:sizeof(P) atIndex:4];
        dispatch1d(enc, ctx.psoScatterFast, (NSUInteger)N);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    }
    g_gridP = P; g_gridValid = true;   // cache for same-step reuse
    return true;
}

// Either build the grid (and cache it) or, when reuse is requested and the cached
// grid matches the point count, reuse the resident sortedPos/cellStart by copying
// the cached grid params into P. Returns false only on a genuine build failure.
static bool ensure_grid(MetalContext& ctx, GpuBuffers& B, ParamsC& P,
                        const float* points3, int N,
                        const float origin[3], const int dims[3], float h, bool reuse)
{
    if (reuse && g_gridValid && g_gridP.totalPoints == N) { P = g_gridP; return true; }
    return build_grid_fast(ctx, B, P, points3, N, origin, dims, h);
}

// Runs one per-particle kernel that reads (points, sortedPos, volume, cellStart)
// and writes one float per particle. Used by both density and factor.
static bool run_per_particle(MetalContext& ctx, GpuBuffers& B, ParamsC& P,
                             id<MTLComputePipelineState> pso,
                             const float* volume, int N, float* out,
                             const float* boundary_volume, const float* boundary_xj)
{
    id<MTLDevice> dev = ctx.device;
    ensure(dev, B.volume,  sizeof(float) * (size_t)N);
    ensure(dev, B.density, sizeof(float) * (size_t)N);   // reused as the output slot
    ensure(dev, B.bVol,    sizeof(float) * (size_t)N);
    ensure(dev, B.bXj,     sizeof(float) * 3 * (size_t)N);
    std::memcpy(B.volume.contents, volume, sizeof(float) * (size_t)N);

    P.hasBoundary = (boundary_volume != nullptr && boundary_xj != nullptr) ? 1 : 0;
    if (P.hasBoundary) {
        std::memcpy(B.bVol.contents, boundary_volume, sizeof(float) * (size_t)N);
        std::memcpy(B.bXj.contents,  boundary_xj,     sizeof(float) * 3 * (size_t)N);
    }
    {
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setBuffer:B.points    offset:0 atIndex:0];
        [enc setBuffer:B.sPoints   offset:0 atIndex:1];
        [enc setBuffer:B.volume    offset:0 atIndex:2];
        [enc setBuffer:B.cellStart offset:0 atIndex:3];
        [enc setBuffer:B.bVol      offset:0 atIndex:4];
        [enc setBuffer:B.bXj       offset:0 atIndex:5];
        [enc setBuffer:B.density   offset:0 atIndex:6];
        [enc setBytes:&P length:sizeof(P) atIndex:7];
        dispatch1d(enc, pso, (NSUInteger)N);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    }
    std::memcpy(out, B.density.contents, sizeof(float) * (size_t)N);
    return true;
}

bool metal_compute_density(const MetalDensityRequest& req, float* out_density, std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }
    @autoreleasepool {
        GpuBuffers& B = buffers();
        ParamsC P{};
        if (!build_grid_fast(ctx, B, P, req.points, req.n, req.origin, req.grid_dims, req.h)) { error = "density: bad size"; return false; }
        run_per_particle(ctx, B, P, ctx.psoDensity, req.volume, req.n, out_density, req.boundary_volume, req.boundary_xj);
    }
    return true;
}

bool metal_compute_factor(const MetalDensityRequest& req, float* out_factor, std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }
    @autoreleasepool {
        GpuBuffers& B = buffers();
        ParamsC P{};
        if (!ensure_grid(ctx, B, P, req.points, req.n, req.origin, req.grid_dims, req.h, req.reuse_grid)) { error = "factor: bad size"; return false; }
        run_per_particle(ctx, B, P, ctx.psoFactor, req.volume, req.n, out_factor, req.boundary_volume, req.boundary_xj);
    }
    return true;
}

bool metal_compute_density_adv(const MetalDensityRequest& req, float* out_delta, std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }
    const int N = req.n;
    @autoreleasepool {
        GpuBuffers& B = buffers();
        ParamsC P{};
        if (!build_grid_fast(ctx, B, P, req.points, N, req.origin, req.grid_dims, req.h)) { error = "density_adv: bad size"; return false; }
        id<MTLDevice> dev = ctx.device;
        ensure(dev, B.volume,   sizeof(float) * (size_t)N);
        ensure(dev, B.velocity, sizeof(float) * 3 * (size_t)N);
        ensure(dev, B.density,  sizeof(float) * (size_t)N);   // output slot
        ensure(dev, B.bVol,     sizeof(float) * (size_t)N);
        ensure(dev, B.bXj,      sizeof(float) * 3 * (size_t)N);
        std::memcpy(B.volume.contents,   req.volume,   sizeof(float) * (size_t)N);
        std::memcpy(B.velocity.contents, req.velocity, sizeof(float) * 3 * (size_t)N);
        P.hasBoundary = (req.boundary_volume && req.boundary_xj) ? 1 : 0;
        if (P.hasBoundary) {
            std::memcpy(B.bVol.contents, req.boundary_volume, sizeof(float) * (size_t)N);
            std::memcpy(B.bXj.contents,  req.boundary_xj,     sizeof(float) * 3 * (size_t)N);
        }
        {
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:B.points    offset:0 atIndex:0];
            [enc setBuffer:B.sPoints   offset:0 atIndex:1];
            [enc setBuffer:B.volume    offset:0 atIndex:2];
            [enc setBuffer:B.velocity  offset:0 atIndex:3];
            [enc setBuffer:B.cellStart offset:0 atIndex:4];
            [enc setBuffer:B.bVol      offset:0 atIndex:5];
            [enc setBuffer:B.bXj       offset:0 atIndex:6];
            [enc setBuffer:B.density   offset:0 atIndex:7];
            [enc setBytes:&P length:sizeof(P) atIndex:8];
            dispatch1d(enc, ctx.psoDensityAdv, (NSUInteger)N);
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        }
        std::memcpy(out_delta, B.density.contents, sizeof(float) * (size_t)N);
    }
    return true;
}

bool metal_pressure_solve(const MetalPressureRequest& req, float* out_accel, int* out_iterations, std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }
    const int N = req.n;
    @autoreleasepool {
        GpuBuffers& B = buffers();
        ParamsC P{};
        if (!ensure_grid(ctx, B, P, req.points, N, req.origin, req.grid_dims, req.h, req.reuse_grid)) { error = "pressure: bad size"; return false; }
        id<MTLDevice> dev = ctx.device;
        ensure(dev, B.volume,    sizeof(float) * (size_t)N);
        ensure(dev, B.pRho2,     sizeof(float) * (size_t)N);
        ensure(dev, B.accel,     sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.densAdv,   sizeof(float) * (size_t)N);
        ensure(dev, B.factorBuf, sizeof(float) * (size_t)N);
        ensure(dev, B.err,       sizeof(float) * (size_t)N);
        ensure(dev, B.bVol,      sizeof(float) * (size_t)N);
        ensure(dev, B.bXj,       sizeof(float) * 3 * (size_t)N);
        std::memcpy(B.volume.contents,    req.volume,        sizeof(float) * (size_t)N);
        std::memcpy(B.pRho2.contents,     req.pressure_rho2, sizeof(float) * (size_t)N);
        std::memcpy(B.densAdv.contents,   req.densityAdv,    sizeof(float) * (size_t)N);
        std::memcpy(B.factorBuf.contents, req.factor,        sizeof(float) * (size_t)N);
        P.dt = req.dt; P.density0 = req.density0;
        P.divergence = req.divergence ? 1 : 0;
        P.aij_scale = req.divergence ? req.dt : (req.dt * req.dt);
        P.hasBoundary = (req.boundary_volume && req.boundary_xj) ? 1 : 0;
        if (P.hasBoundary) {
            std::memcpy(B.bVol.contents, req.boundary_volume, sizeof(float) * (size_t)N);
            std::memcpy(B.bXj.contents,  req.boundary_xj,     sizeof(float) * 3 * (size_t)N);
        }

        // Build the flat neighbour list with precomputed V_j*gradW once per step;
        // the divergence and pressure solves share it (positions fixed across both).
        if (!(g_nlValid && g_gridP.totalPoints == N)) {
            ensure(dev, B.count, sizeof(uint32_t) * (size_t)N);
            {
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setBuffer:B.points offset:0 atIndex:0];
                [enc setBuffer:B.sPoints offset:0 atIndex:1];
                [enc setBuffer:B.cellStart offset:0 atIndex:2];
                [enc setBuffer:B.count offset:0 atIndex:3];
                [enc setBytes:&P length:sizeof(P) atIndex:4];
                dispatch1d(enc, ctx.psoNlCount, (NSUInteger)N);
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            }
            const uint32_t* cnt = (const uint32_t*)B.count.contents;
            B.nlStartHost.resize((size_t)N + 1);
            uint32_t acc = 0;
            for (int p = 0; p < N; p++) { B.nlStartHost[p] = acc; acc += cnt[p]; }
            B.nlStartHost[N] = acc;
            g_nlTotal = (int)acc;
            const int tot = g_nlTotal > 0 ? g_nlTotal : 1;
            ensure(dev, B.nlStart, sizeof(uint32_t) * (size_t)(N + 1));
            std::memcpy(B.nlStart.contents, B.nlStartHost.data(), sizeof(uint32_t) * (size_t)(N + 1));
            ensure(dev, B.nlId,  sizeof(int)   * (size_t)tot);
            ensure(dev, B.nlVgw, sizeof(float) * 4 * (size_t)tot);
            ensure(dev, B.bVgw,  sizeof(float) * 4 * (size_t)N);
            {
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setBuffer:B.points offset:0 atIndex:0];
                [enc setBuffer:B.sPoints offset:0 atIndex:1];
                [enc setBuffer:B.volume offset:0 atIndex:2];
                [enc setBuffer:B.cellStart offset:0 atIndex:3];
                [enc setBuffer:B.nlStart offset:0 atIndex:4];
                [enc setBuffer:B.bVol offset:0 atIndex:5];
                [enc setBuffer:B.bXj offset:0 atIndex:6];
                [enc setBuffer:B.nlId offset:0 atIndex:7];
                [enc setBuffer:B.nlVgw offset:0 atIndex:8];
                [enc setBuffer:B.bVgw offset:0 atIndex:9];
                [enc setBytes:&P length:sizeof(P) atIndex:10];
                dispatch1d(enc, ctx.psoNlFill, (NSUInteger)N);
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            }
            g_nlValid = true;
        }

        auto encode_accel = [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:B.nlStart offset:0 atIndex:0];
            [enc setBuffer:B.nlId    offset:0 atIndex:1];
            [enc setBuffer:B.nlVgw   offset:0 atIndex:2];
            [enc setBuffer:B.bVgw    offset:0 atIndex:3];
            [enc setBuffer:B.pRho2   offset:0 atIndex:4];
            [enc setBuffer:B.accel   offset:0 atIndex:5];
            [enc setBytes:&P length:sizeof(P) atIndex:6];
            dispatch1d(enc, ctx.psoPaNl, (NSUInteger)N);
        };
        auto encode_update = [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:B.nlStart   offset:0 atIndex:0];
            [enc setBuffer:B.nlId      offset:0 atIndex:1];
            [enc setBuffer:B.nlVgw     offset:0 atIndex:2];
            [enc setBuffer:B.bVgw      offset:0 atIndex:3];
            [enc setBuffer:B.accel     offset:0 atIndex:4];
            [enc setBuffer:B.densAdv   offset:0 atIndex:5];
            [enc setBuffer:B.factorBuf offset:0 atIndex:6];
            [enc setBuffer:B.pRho2     offset:0 atIndex:7];
            [enc setBuffer:B.err       offset:0 atIndex:8];
            [enc setBytes:&P length:sizeof(P) atIndex:9];
            dispatch1d(enc, ctx.psoPuNl, (NSUInteger)N);
        };
        // One iteration = accel + update in a single command buffer (one CPU<->GPU
        // sync), with the density error reduced on the GPU into B.err[0].
        auto iterate = [&]()->double {
            *(float*)B.err.contents = 0.0f;   // shared buffer: reset the atomic accumulator
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> e1 = [cmd computeCommandEncoder]; encode_accel(e1);  [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cmd computeCommandEncoder]; encode_update(e2); [e2 endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
            return (double)(*(const float*)B.err.contents) / (double)N;
        };

        int iters = 0;
        if (req.max_iterations > 0) {
            bool chk = false;
            while ((!chk || iters < req.min_iterations) && iters < req.max_iterations) {
                chk = (iterate() <= (double)req.eta);
                iters++;
            }
        } else {
            for (int it = 0; it < req.iterations; it++) { iterate(); iters++; }
        }
        // Final pressure accel from the converged pressure, for the velocity update.
        if (out_accel) {
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; encode_accel(enc); [enc endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
            const float* a = (const float*)B.accel.contents;   // float4 per particle
            for (int i = 0; i < N; i++) { out_accel[3*i+0]=a[4*i+0]; out_accel[3*i+1]=a[4*i+1]; out_accel[3*i+2]=a[4*i+2]; }
        }
        if (out_iterations) *out_iterations = iters;
        std::memcpy(req.pressure_rho2, B.pRho2.contents, sizeof(float) * (size_t)N);
    }
    return true;
}

// ── GPU-resident SPH solve ───────────────────────────────────────────────────

void metal_sph_set_external_context(void* mtlDevice, void* mtlQueue)
{
    id<MTLDevice>       dev = (__bridge id<MTLDevice>)mtlDevice;
    id<MTLCommandQueue> q   = (__bridge id<MTLCommandQueue>)mtlQueue;
    if (dev == g_extDevice && q == g_extQueue) return;   // unchanged
    g_extDevice = dev;
    g_extQueue  = q;
    g_contextNeedsRebuild = true;   // context() recompiles pipelines on the new device
    reset_buffers();                // cached buffers belonged to the previous device
}

bool metal_sph_solve_gpu(const MetalSphGpuRequest& req, std::string& error)
{
    MetalContext& ctx = context();
    if (!ctx.ok) { error = ctx.error; return false; }
    const int N = req.n;
    if (N <= 0) { error = "solve_gpu: n <= 0"; return false; }
    if (!req.points || !req.volume || !req.outAccel) { error = "solve_gpu: null buffer handle"; return false; }

    const long nCellsL = (long)req.grid_dims[0] * (long)req.grid_dims[1] * (long)req.grid_dims[2];
    if (nCellsL <= 0 || nCellsL > 300000000L) { error = "solve_gpu: bad grid size"; return false; }
    const int nCells = (int)nCellsL;

    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        GpuBuffers& B = buffers();
        id<MTLBuffer> pts = (__bridge id<MTLBuffer>)req.points;
        id<MTLBuffer> vol = (__bridge id<MTLBuffer>)req.volume;
        id<MTLBuffer> out = (__bridge id<MTLBuffer>)req.outAccel;
        id<MTLBuffer> vel = req.velocity ? (__bridge id<MTLBuffer>)req.velocity : nil;
        const bool doXsph = (vel != nil && req.viscosity > 0.0f);

        // ── Grid params, shared by every phase ───────────────────────────────────
        ParamsC P{};
        P.ox = req.origin[0]; P.oy = req.origin[1]; P.oz = req.origin[2];
        P.inv_cs = 1.0f / req.h;
        P.dx = req.grid_dims[0]; P.dy = req.grid_dims[1]; P.dz = req.grid_dims[2];
        P.R = 1; P.nCells = nCells; P.totalPoints = N; P.nSearch = N; P.r2 = req.h;

        // ── Scratch. The grid prefix-sum moves onto the GPU (k_scan_cells; cellStart
        //    length nCells+1 is known a priori, so the whole grid build stays GPU-side).
        //    The neighbour-list prefix-sum also moves onto the GPU (k_exscan), keeping the
        //    EXACT CSR layout — the only remaining host dependency is one word (the pair
        //    total nlStart[N]) read back to size nlId/nlVgw, which splits the solve into
        //    two command buffers (two waits) instead of the old eight. No neighbour cap. ──
        ensure(dev, B.cellId,     sizeof(uint32_t) * (size_t)N);
        ensure(dev, B.counts,     sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.cellStart,  sizeof(uint32_t) * (size_t)(nCells + 1));
        ensure(dev, B.cellOffset, sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.sPoints,    sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.density,    sizeof(float) * (size_t)N);
        ensure(dev, B.factorBuf,  sizeof(float) * (size_t)N);
        ensure(dev, B.densAdv,    sizeof(float) * (size_t)N);
        ensure(dev, B.pRho2,      sizeof(float) * (size_t)N);
        ensure(dev, B.accel,      sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.err,        sizeof(float) * (size_t)N);
        ensure(dev, B.bVol,       sizeof(float) * (size_t)N);           // unused (no boundary)
        ensure(dev, B.bXj,        sizeof(float) * 3 * (size_t)N);
        ensure(dev, B.count,      sizeof(uint32_t) * (size_t)N);
        ensure(dev, B.nlStart,    sizeof(uint32_t) * (size_t)(N + 1));
        ensure(dev, B.bVgw,       sizeof(float) * 4 * (size_t)N);
        // B.nlId / B.nlVgw are sized to the EXACT pair total (nlStart[N]) after CB1's
        // neighbour-count exclusive scan — a one-word readback, exact CSR, no cap.
        if (doXsph) ensure(dev, B.velSmooth, sizeof(float) * 3 * (size_t)N);

        // Host-side buffer prep. These writes precede [cmd commit], so the GPU sees them
        // (shared storage): zero the per-cell counts (hash accumulates), the pressure
        // (starts at rest) and the density-error accumulator (atomic reduction target).
        std::memset(B.counts.contents, 0, sizeof(uint32_t) * (size_t)nCells);
        std::memset(B.pRho2.contents,  0, sizeof(float) * (size_t)N);
        *(float*)B.err.contents = 0.0f;

        ParamsC Pp = P; Pp.hasBoundary = 0; Pp.viscosity = req.viscosity;   // grid/density/NL build
        ParamsC Ps = P; Ps.hasBoundary = 0; Ps.dt = req.dt; Ps.density0 = req.density0;
        Ps.divergence = 0; Ps.aij_scale = req.dt * req.dt;                   // pressure solve (h^2)

        // ── Phase encoders. Each appends encoders to a caller-supplied command buffer;
        //    Metal serialises compute encoders within one buffer and auto-tracks buffer
        //    hazards between them, so back-to-back phases need no CPU sync. ──────────────

        // grid: hash (counts pre-zeroed) -> GPU exclusive scan -> scatter into sPoints.
        auto enc_grid = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> eh = [cmd computeCommandEncoder];
            [eh setBuffer:pts offset:0 atIndex:0];
            [eh setBuffer:B.cellId offset:0 atIndex:1];
            [eh setBuffer:B.counts offset:0 atIndex:2];
            [eh setBytes:&P length:sizeof(P) atIndex:3];
            dispatch1d(eh, ctx.psoHash, (NSUInteger)N);
            [eh endEncoding];
            id<MTLComputeCommandEncoder> es = [cmd computeCommandEncoder];   // counts -> cellStart(+seed cellOffset)
            [es setBuffer:B.counts offset:0 atIndex:0];
            [es setBuffer:B.cellStart offset:0 atIndex:1];
            [es setBuffer:B.cellOffset offset:0 atIndex:2];
            [es setBytes:&P length:sizeof(P) atIndex:3];
            dispatch1d(es, ctx.psoScanCells, 1);   // single serial thread; replaces the CPU scan+readback
            [es endEncoding];
            id<MTLComputeCommandEncoder> esc = [cmd computeCommandEncoder];
            [esc setBuffer:B.cellId offset:0 atIndex:0];
            [esc setBuffer:B.cellOffset offset:0 atIndex:1];
            [esc setBuffer:pts offset:0 atIndex:2];
            [esc setBuffer:B.sPoints offset:0 atIndex:3];
            [esc setBytes:&P length:sizeof(P) atIndex:4];
            dispatch1d(esc, ctx.psoScatterFast, (NSUInteger)N);
            [esc endEncoding];
        };

        // density -> factor -> constant-density source clamp (B.density hazard tracked).
        auto enc_dfac = [&](id<MTLCommandBuffer> cmd){
            auto bind_pp = [&](id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pso, id<MTLBuffer> dstOut){
                [enc setBuffer:pts offset:0 atIndex:0];
                [enc setBuffer:B.sPoints offset:0 atIndex:1];
                [enc setBuffer:vol offset:0 atIndex:2];
                [enc setBuffer:B.cellStart offset:0 atIndex:3];
                [enc setBuffer:B.bVol offset:0 atIndex:4];
                [enc setBuffer:B.bXj offset:0 atIndex:5];
                [enc setBuffer:dstOut offset:0 atIndex:6];
                [enc setBytes:&Pp length:sizeof(Pp) atIndex:7];
                dispatch1d(enc, pso, (NSUInteger)N);
            };
            id<MTLComputeCommandEncoder> e1 = [cmd computeCommandEncoder]; bind_pp(e1, ctx.psoDensity, B.density);   [e1 endEncoding];
            id<MTLComputeCommandEncoder> e2 = [cmd computeCommandEncoder]; bind_pp(e2, ctx.psoFactor,  B.factorBuf); [e2 endEncoding];
            id<MTLComputeCommandEncoder> e3 = [cmd computeCommandEncoder];
            [e3 setBuffer:B.density offset:0 atIndex:0];
            [e3 setBuffer:B.densAdv offset:0 atIndex:1];
            [e3 setBytes:&Pp length:sizeof(Pp) atIndex:2];
            dispatch1d(e3, ctx.psoDensadvClamp, (NSUInteger)N);
            [e3 endEncoding];
        };

        // optional XSPH: sweep into velSmooth (reads pre-smoothing vel), blit back into vel.
        auto enc_xsph = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:pts offset:0 atIndex:0];
            [enc setBuffer:B.sPoints offset:0 atIndex:1];
            [enc setBuffer:vol offset:0 atIndex:2];
            [enc setBuffer:B.cellStart offset:0 atIndex:3];
            [enc setBuffer:vel offset:0 atIndex:4];   // velIn
            [enc setBuffer:B.density offset:0 atIndex:5];
            [enc setBuffer:B.velSmooth offset:0 atIndex:6];   // velOut
            [enc setBytes:&Pp length:sizeof(Pp) atIndex:7];
            dispatch1d(enc, ctx.psoXsph, (NSUInteger)N);
            [enc endEncoding];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:B.velSmooth sourceOffset:0 toBuffer:vel destinationOffset:0
                            size:sizeof(float) * 3 * (size_t)N];
            [blit endEncoding];
        };

        // neighbour list (exact CSR): per-particle count -> GPU exclusive-scan into nlStart.
        // The one-word total nlStart[N] is read after this command buffer to size nlId/nlVgw;
        // the fill (enc_nlfill) then runs in the next command buffer.
        auto enc_nlcount = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> ec = [cmd computeCommandEncoder];
            [ec setBuffer:pts offset:0 atIndex:0];
            [ec setBuffer:B.sPoints offset:0 atIndex:1];
            [ec setBuffer:B.cellStart offset:0 atIndex:2];
            [ec setBuffer:B.count offset:0 atIndex:3];
            [ec setBytes:&Pp length:sizeof(Pp) atIndex:4];
            dispatch1d(ec, ctx.psoNlCount, (NSUInteger)N);
            [ec endEncoding];
            id<MTLComputeCommandEncoder> es = [cmd computeCommandEncoder];   // counts -> nlStart (exclusive)
            [es setBuffer:B.count offset:0 atIndex:0];
            [es setBuffer:B.nlStart offset:0 atIndex:1];
            [es setBytes:&Pp length:sizeof(Pp) atIndex:2];
            dispatch1d(es, ctx.psoExscan, 1);
            [es endEncoding];
        };
        auto enc_nlfill = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> ef = [cmd computeCommandEncoder];
            [ef setBuffer:pts offset:0 atIndex:0];
            [ef setBuffer:B.sPoints offset:0 atIndex:1];
            [ef setBuffer:vol offset:0 atIndex:2];
            [ef setBuffer:B.cellStart offset:0 atIndex:3];
            [ef setBuffer:B.nlStart offset:0 atIndex:4];
            [ef setBuffer:B.bVol offset:0 atIndex:5];
            [ef setBuffer:B.bXj offset:0 atIndex:6];
            [ef setBuffer:B.nlId offset:0 atIndex:7];
            [ef setBuffer:B.nlVgw offset:0 atIndex:8];
            [ef setBuffer:B.bVgw offset:0 atIndex:9];
            [ef setBytes:&Pp length:sizeof(Pp) atIndex:10];
            dispatch1d(ef, ctx.psoNlFill, (NSUInteger)N);
            [ef endEncoding];
        };

        // CSR pressure-accel / Jacobi-update encoders (read nlStart[i]..nlStart[i+1]).
        auto enc_accel = [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:B.nlStart offset:0 atIndex:0];
            [enc setBuffer:B.nlId    offset:0 atIndex:1];
            [enc setBuffer:B.nlVgw   offset:0 atIndex:2];
            [enc setBuffer:B.bVgw    offset:0 atIndex:3];
            [enc setBuffer:B.pRho2   offset:0 atIndex:4];
            [enc setBuffer:B.accel   offset:0 atIndex:5];
            [enc setBytes:&Ps length:sizeof(Ps) atIndex:6];
            dispatch1d(enc, ctx.psoPaNl, (NSUInteger)N);
        };
        auto enc_update = [&](id<MTLComputeCommandEncoder> enc){
            [enc setBuffer:B.nlStart   offset:0 atIndex:0];
            [enc setBuffer:B.nlId      offset:0 atIndex:1];
            [enc setBuffer:B.nlVgw     offset:0 atIndex:2];
            [enc setBuffer:B.bVgw      offset:0 atIndex:3];
            [enc setBuffer:B.accel     offset:0 atIndex:4];
            [enc setBuffer:B.densAdv   offset:0 atIndex:5];
            [enc setBuffer:B.factorBuf offset:0 atIndex:6];
            [enc setBuffer:B.pRho2     offset:0 atIndex:7];
            [enc setBuffer:B.err       offset:0 atIndex:8];
            [enc setBytes:&Ps length:sizeof(Ps) atIndex:9];
            dispatch1d(enc, ctx.psoPuNl, (NSUInteger)N);
        };
        // g Jacobi iterations (accel -> update). When zeroErr, B.err is cleared right
        // before the final update so a post-wait read is that iteration's density error.
        auto enc_jacobi = [&](id<MTLCommandBuffer> cmd, int g, bool zeroErr){
            for (int k = 0; k < g; k++) {
                id<MTLComputeCommandEncoder> ea = [cmd computeCommandEncoder]; enc_accel(ea); [ea endEncoding];
                if (zeroErr && k == g - 1) {
                    id<MTLBlitCommandEncoder> bz = [cmd blitCommandEncoder];
                    [bz fillBuffer:B.err range:NSMakeRange(0, sizeof(float)) value:0];
                    [bz endEncoding];
                }
                id<MTLComputeCommandEncoder> eu = [cmd computeCommandEncoder]; enc_update(eu); [eu endEncoding];
            }
        };
        // final accel from the converged p~, packed straight into the caller's float3 buffer.
        auto enc_final = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> ea = [cmd computeCommandEncoder]; enc_accel(ea); [ea endEncoding];
            id<MTLComputeCommandEncoder> ecp = [cmd computeCommandEncoder];
            [ecp setBuffer:B.accel offset:0 atIndex:0];
            [ecp setBuffer:out     offset:0 atIndex:1];
            [ecp setBytes:&Ps length:sizeof(Ps) atIndex:2];
            dispatch1d(ecp, ctx.psoAccelCopy3, (NSUInteger)N);
            [ecp endEncoding];
        };

        int maxIt = req.max_iterations > 0 ? req.max_iterations : 1;
        int minIt = req.min_iterations < 0 ? 0 : req.min_iterations;
        if (minIt > maxIt) minIt = maxIt;
        const bool earlyOut = (req.eta > 0.0f && minIt < maxIt);

        // Size the exact CSR pair buffers from the neighbour-count scan. Shared by both
        // paths: run CB1 (grid -> density/factor/clamp -> [XSPH] -> nl_count -> scan),
        // wait once, then read the single word nlStart[N] (the pair total) and grow
        // nlId/nlVgw to exactly that. CB1 has drained, so the realloc cannot race.
        auto build_and_size_nl = [&]{
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            enc_grid(cmd);
            enc_dfac(cmd);
            if (doXsph) enc_xsph(cmd);
            enc_nlcount(cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
            const uint32_t tot = ((const uint32_t*)B.nlStart.contents)[N];
            const size_t p = (size_t)(tot > 0 ? tot : 1);
            ensure(dev, B.nlId,  sizeof(int)   * p);
            ensure(dev, B.nlVgw, sizeof(float) * 4 * p);
        };

        if (!earlyOut) {
            // Two command buffers, two waits: CB1 (grid..nl_count..scan) then, after the
            // one-word size readback, CB2 = nl_fill -> all Jacobi iterations -> final pack.
            build_and_size_nl();
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            enc_nlfill(cmd);
            enc_jacobi(cmd, maxIt, /*zeroErr=*/false);
            enc_final(cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
        } else {
            // Early-out requested: the convergence readback forces splitting the Jacobi
            // loop. CB1 builds+sizes the list; the fill rides the first chunk's buffer;
            // then chunked Jacobi (one wait per chunk) and the final accel pack.
            build_and_size_nl();
            const int kSolveChunk = 4;
            int done = 0;
            bool filled = false;
            while (done < maxIt) {
                int g = (maxIt - done < kSolveChunk) ? (maxIt - done) : kSolveChunk;
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                if (!filled) { enc_nlfill(cmd); filled = true; }
                enc_jacobi(cmd, g, /*zeroErr=*/true);
                [cmd commit];
                [cmd waitUntilCompleted];
                done += g;
                double err = (double)(*(const float*)B.err.contents) / (double)N;
                if (done >= minIt && err <= (double)req.eta) break;
            }
            {
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                enc_final(cmd);
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }

        // Leave no reusable cache behind: the resident grid/NL scratch is fluid-only and
        // must not be picked up by a later host-pointer solve (which may add boundary).
        g_gridValid = false;
        g_nlValid   = false;
    }
    return true;
}

}} // namespace tns::internals
