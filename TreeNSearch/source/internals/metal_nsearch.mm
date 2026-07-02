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

// Single-sweep search (k_search_fast_sorted, further below): distance tests run
// ONCE, neighbour ids go to a per-point scratch row (capacity TNS_MAXNB) with the
// true (uncapped) count recorded; a later compaction packs the rows — avoiding an
// expensive second sweep. Overflowing rows (rare) are re-swept into place.
#define TNS_MAXNB 128

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

// ─── Parallel exclusive prefix-sum (three dispatches, one command buffer) ─────
// Replaces the single-thread scans for anything that grows with N: k_scan_partial
// scans each 256-element block in-threadgroup (simdgroup scans + a block combine)
// and emits per-block totals; k_scan_blocks turns the block totals into exclusive
// block offsets with one looping threadgroup; k_scan_fixup adds the block offset
// back, writes the grand total into out[n], and (for the grid) seeds the scatter
// cursor with the same offsets.
struct ScanArgs {
    uint n;          // number of input elements
    uint numBlocks;  // ceil(n / 256)
    uint seedCopy;   // 1 -> also write the offsets into out2 (grid scatter cursor)
    uint addPer;     // added to every element (1 -> scan of count+1: packed [count, ids...] offsets)
};

kernel void k_scan_partial(device const uint* in       [[buffer(0)]],
                           device uint*       partial  [[buffer(1)]],
                           device uint*       blockSum [[buffer(2)]],
                           constant ScanArgs& A        [[buffer(3)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lid  [[thread_index_in_threadgroup]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simdSums[8];   // 256 threads / 32 lanes
    uint v = (gid < A.n) ? (in[gid] + A.addPer) : 0u;
    uint p = simd_prefix_exclusive_sum(v);
    if (lane == 31) simdSums[sg] = p + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint s  = (lane < 8) ? simdSums[lane] : 0u;
        uint sp = simd_prefix_exclusive_sum(s);
        if (lane < 8) simdSums[lane] = sp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p += simdSums[sg];
    if (gid < A.n) partial[gid] = p;
    if (lid == 255) blockSum[tg] = p + v;   // block total (padding lanes carry v = 0)
}

kernel void k_scan_blocks(device uint*       blockSum [[buffer(0)]],
                          constant ScanArgs& A        [[buffer(1)]],
                          uint lid  [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]])
{
    // One 256-wide threadgroup scans all block totals, 256 at a time.
    threadgroup uint simdSums[8];
    threadgroup uint carry, carryNext;
    if (lid == 0) carry = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < A.numBlocks; base += 256) {
        uint i = base + lid;
        uint v = (i < A.numBlocks) ? blockSum[i] : 0u;
        uint p = simd_prefix_exclusive_sum(v);
        if (lane == 31) simdSums[sg] = p + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            uint s  = (lane < 8) ? simdSums[lane] : 0u;
            uint sp = simd_prefix_exclusive_sum(s);
            if (lane < 8) simdSums[lane] = sp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        p += simdSums[sg] + carry;
        if (i < A.numBlocks) blockSum[i] = p;
        if (lid == 255) carryNext = p + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) carry = carryNext;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void k_scan_fixup(device const uint* partial  [[buffer(0)]],
                         device const uint* blockSum [[buffer(1)]],   // exclusive block offsets
                         device uint*       out      [[buffer(2)]],
                         device uint*       out2     [[buffer(3)]],   // seeded copy (bind out when unused)
                         device const uint* in       [[buffer(4)]],
                         constant ScanArgs& A        [[buffer(5)]],
                         uint gid [[thread_position_in_grid]])
{
    if (gid >= A.n) return;
    uint val = partial[gid] + blockSum[gid / 256];
    out[gid] = val;
    if (A.seedCopy != 0) out2[gid] = val;
    if (gid == A.n - 1) out[A.n] = val + in[gid] + A.addPer;   // grand total
}

// ─── SPH resident path, sorted-space kernels ─────────────────────────────────
// Everything below runs over particles in CELL (sorted) order: thread i handles
// sorted slot i, so adjacent threads sweep adjacent cells and the CSR arrays are
// written/read contiguously. Original ids live in sortedOrig; results are
// scattered back to caller order only at the boundaries (velocity, accel).

// Counting-sort scatter that also reorders the per-particle volume and keeps the
// original index both in sortedPos.w (bit-cast) and as a plain int array.
kernel void k_scatter_sph(device const uint*   cellId     [[buffer(0)]],
                          device atomic_uint*  cellOffset [[buffer(1)]],
                          device const float4* points     [[buffer(2)]],
                          device const float*  volume     [[buffer(3)]],
                          device float4*       sortedPos  [[buffer(4)]],
                          device float*        sortedVol  [[buffer(5)]],
                          device int*          sortedOrig [[buffer(6)]],
                          constant Params&     P          [[buffer(7)]],
                          uint                 gid        [[thread_position_in_grid]])
{
    if ((int)gid >= P.totalPoints) return;
    uint lin  = cellId[gid];
    uint slot = atomic_fetch_add_explicit(&cellOffset[lin], 1u, memory_order_relaxed);
    float4 p = points[gid];
    p.w = as_type<float>((int)gid);
    sortedPos[slot]  = p;
    sortedVol[slot]  = volume[gid];
    sortedOrig[slot] = (int)gid;
}

// Neighbour count per sorted slot. Cells along x are adjacent in the grid
// linearisation, so the 3x3x3 block collapses into 9 contiguous slot ranges
// (one per (y,z) row) — 9 range lookups instead of 27 cell lookups, and each
// range is one contiguous, coalesced run of sortedPos.
kernel void k_nl_count_sorted(device const float4* sortedPos [[buffer(0)]],
                              device const uint*   cellStart [[buffer(1)]],
                              device uint*         counts    [[buffer(2)]],
                              constant Params&     P         [[buffer(3)]],
                              uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h2 = P.r2 * P.r2;
    float3 xi = sortedPos[i].xyz;
    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int xlo = max(cx - 1, 0), xhi = min(cx + 1, P.dx - 1);
    int n = 0;
    for (int dz = -1; dz <= 1; dz++) { int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
    for (int dy = -1; dy <= 1; dy++) { int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
        int row = (ncz * P.dy + ncy) * P.dx;
        uint begin = cellStart[row + xlo];
        uint end   = cellStart[row + xhi + 1];
        for (uint s = begin; s < end; s++) {
            if (s == i) continue;
            float3 e = xi - sortedPos[s].xyz;
            if (dot(e, e) <= h2) n++;
        }
    }}
    counts[i] = (uint)n;
}

// Fused build: ONE traversal emits the CSR neighbour list (ids + V_j*gradW with
// V_j*W in .w) AND the SPH density, DFSPH factor and clamped constant-density
// source. Replaces four separate traversals (density, factor, nl-count, nl-fill)
// of the previous design; only the count pass above remains separate (the CSR
// layout needs the totals first).
kernel void k_sph_build(device const float4* sortedPos  [[buffer(0)]],
                        device const float*  sortedVol  [[buffer(1)]],
                        device const uint*   cellStart  [[buffer(2)]],
                        device const uint*   nlStart    [[buffer(3)]],
                        device int*          nlId       [[buffer(4)]],
                        device float4*       nlVgw      [[buffer(5)]],
                        device float*        outDensity [[buffer(6)]],
                        device float*        outFactor  [[buffer(7)]],
                        device float*        outDensAdv [[buffer(8)]],
                        constant Params&     P          [[buffer(9)]],
                        uint                 i          [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    const float h  = P.r2;
    const float h2 = h * h;
    const float kW = 8.0f  / (3.14159265358979f * h * h * h);   // CubicKernel m_k
    const float l  = 48.0f / (3.14159265358979f * h * h * h);   // CubicKernel m_l

    float3 xi = sortedPos[i].xyz;
    float  density = sortedVol[i] * kW;   // self: V_i * W(0)
    float3 gpi = float3(0.0f);
    float  sum = 0.0f;

    int cx = clamp((int)floor((xi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((xi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((xi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int xlo = max(cx - 1, 0), xhi = min(cx + 1, P.dx - 1);
    uint k = nlStart[i];
    for (int dz = -1; dz <= 1; dz++) { int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
    for (int dy = -1; dy <= 1; dy++) { int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
        int row = (ncz * P.dy + ncy) * P.dx;
        uint begin = cellStart[row + xlo];
        uint end   = cellStart[row + xhi + 1];
        for (uint s = begin; s < end; s++) {
            if (s == i) continue;
            float3 rvec = xi - sortedPos[s].xyz;
            float d2 = dot(rvec, rvec);
            if (d2 > h2) continue;
            float rl = sqrt(d2);
            float qd = rl / h;
            float w;   // CubicKernel W
            if (qd <= 0.5f) { float q2 = qd * qd; w = kW * (6.0f*q2*qd - 6.0f*q2 + 1.0f); }
            else            { float f = 1.0f - qd; w = kW * (2.0f*f*f*f); }
            float3 gw = float3(0.0f);   // CubicKernel gradW
            if (rl > 1.0e-9f) {
                float3 gradq = rvec / (rl * h);
                if (qd <= 0.5f) gw = l * qd * (3.0f*qd - 2.0f) * gradq;
                else { float f = 1.0f - qd; gw = l * (-f*f) * gradq; }
            }
            float Vj = sortedVol[s];
            float3 Vgw = Vj * gw;
            density += Vj * w;
            sum += dot(Vgw, Vgw);
            gpi += Vgw;
            nlId[k]  = (int)s;
            nlVgw[k] = float4(Vgw, Vj * w);
            k++;
        }
    }}
    sum += dot(gpi, gpi);
    outDensity[i] = density;
    outFactor[i]  = (sum > 1.0e-6f) ? (1.0f / sum) : 0.0f;
    outDensAdv[i] = (isfinite(density) && density > 1.0f) ? density : 1.0f;
}

// XSPH velocity smoothing from the neighbour list: V_j*W_ij was precomputed into
// nlVgw.w by k_sph_build, so this is a multiply-accumulate over the CSR list — no
// grid traversal, no kernel evaluation. Velocities live in caller (original)
// order; velOut is a separate buffer so the pass reads pre-smoothing values.
kernel void k_xsph_nl(device const uint*   nlStart    [[buffer(0)]],
                      device const int*    nlId       [[buffer(1)]],
                      device const float4* nlVgw      [[buffer(2)]],
                      device const int*    sortedOrig [[buffer(3)]],
                      device const float*  velIn      [[buffer(4)]],
                      device const float*  density    [[buffer(5)]],   // sorted order
                      device float*        velOut     [[buffer(6)]],
                      constant Params&     P          [[buffer(7)]],
                      uint                 i          [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int oi = sortedOrig[i];
    float3 vi = float3(velIn[3*oi+0], velIn[3*oi+1], velIn[3*oi+2]);
    float3 acc = float3(0.0f);
    uint b = nlStart[i], e = nlStart[i+1];
    for (uint s = b; s < e; s++) {
        int oj = sortedOrig[nlId[s]];
        float3 vj = float3(velIn[3*oj+0], velIn[3*oj+1], velIn[3*oj+2]);
        acc += nlVgw[s].w * (vj - vi);   // V_j * W_ij * (v_j - v_i)
    }
    float di = max(density[i], 1.0e-6f);
    float3 vs = vi + P.viscosity * acc / di;
    velOut[3*oi+0] = vs.x; velOut[3*oi+1] = vs.y; velOut[3*oi+2] = vs.z;
}

// Scatter the (sorted-order) pressure accel into the caller's tight float3 buffer.
kernel void k_accel_scatter3(device const float4* accel      [[buffer(0)]],
                             device const int*    sortedOrig [[buffer(1)]],
                             device float*        out3       [[buffer(2)]],
                             constant Params&     P          [[buffer(3)]],
                             uint                 i          [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int oi = sortedOrig[i];
    float3 a = accel[i].xyz;
    out3[3*oi+0] = a.x; out3[3*oi+1] = a.y; out3[3*oi+2] = a.z;
}

// ─── Sorted-order fast search (uniform radius, single set) ────────────────────
// Same idea as k_search_fast / k_compact_fast, but thread i handles SORTED slot i:
// the query position is one coalesced sortedPos load, adjacent threads sweep
// adjacent cells (SIMD-coherent trip counts and cache-shared candidate reads),
// and each 3x3x3 block collapses into 9 contiguous slot ranges. Counts and the
// packed output stay keyed by ORIGINAL point index (the API's layout), via the
// original id stashed in sortedPos.w.
static inline int sweep_fast_sorted(uint i,
                                    device const float4* sortedPos,
                                    device const uint*   cellStart,
                                    constant Params&     P,
                                    device int*          out)
{
    float3 pi = sortedPos[i].xyz;
    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int xlo = max(cx - P.R, 0), xhi = min(cx + P.R, P.dx - 1);
    int count = 0;
    for (int dz = -P.R; dz <= P.R; dz++) { int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
    for (int dy = -P.R; dy <= P.R; dy++) { int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
        int row = (ncz * P.dy + ncy) * P.dx;
        uint begin = cellStart[row + xlo];
        uint end   = cellStart[row + xhi + 1];
        for (uint s = begin; s < end; s++) {
            if (s == i) continue;
            float4 q = sortedPos[s];
            float3 e = pi - q.xyz;
            if (dot(e, e) <= P.r2) {
                if (out != nullptr) out[count] = as_type<int>(q.w) - P.setOffsetJ;
                count++;
            }
        }
    }}
    return count;
}

kernel void k_search_fast_sorted(device const float4* sortedPos [[buffer(0)]],
                                 device const uint*   cellStart [[buffer(1)]],
                                 device int*          scratch   [[buffer(2)]],
                                 device uint*         outCount  [[buffer(3)]],
                                 constant Params&     P         [[buffer(4)]],
                                 uint                 i         [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    float3 pi = sortedPos[i].xyz;
    int cx = clamp((int)floor((pi.x - P.ox) * P.inv_cs), 0, P.dx - 1);
    int cy = clamp((int)floor((pi.y - P.oy) * P.inv_cs), 0, P.dy - 1);
    int cz = clamp((int)floor((pi.z - P.oz) * P.inv_cs), 0, P.dz - 1);
    int xlo = max(cx - P.R, 0), xhi = min(cx + P.R, P.dx - 1);
    device int* sc = scratch + (uint)i * TNS_MAXNB;   // scratch row by SLOT
    int count = 0;
    for (int dz = -P.R; dz <= P.R; dz++) { int ncz = cz + dz; if (ncz < 0 || ncz >= P.dz) continue;
    for (int dy = -P.R; dy <= P.R; dy++) { int ncy = cy + dy; if (ncy < 0 || ncy >= P.dy) continue;
        int row = (ncz * P.dy + ncy) * P.dx;
        uint begin = cellStart[row + xlo];
        uint end   = cellStart[row + xhi + 1];
        for (uint s = begin; s < end; s++) {
            if (s == i) continue;
            float4 q = sortedPos[s];
            float3 e = pi - q.xyz;
            if (dot(e, e) <= P.r2) {
                if (count < TNS_MAXNB) sc[count] = as_type<int>(q.w) - P.setOffsetJ;
                count++;
            }
        }
    }}
    outCount[as_type<int>(sortedPos[i].w) - P.setOffsetI] = (uint)count;   // count by ORIGINAL index
}

// Pack the slot-order scratch rows into the original-index-ordered [count, ids...]
// runs, using the GPU-scanned offsets. Overflowed rows are re-swept into place.
kernel void k_compact_fast_sorted(device const float4* sortedPos   [[buffer(0)]],
                                  device const uint*   cellStart   [[buffer(1)]],
                                  device const int*    scratch     [[buffer(2)]],
                                  device const uint*   outCount    [[buffer(3)]],
                                  device const int*    blockOffset [[buffer(4)]],
                                  device int*          block       [[buffer(5)]],
                                  constant Params&     P           [[buffer(6)]],
                                  uint                 i           [[thread_position_in_grid]])
{
    if ((int)i >= P.nSearch) return;
    int oi   = as_type<int>(sortedPos[i].w) - P.setOffsetI;
    int base = blockOffset[oi];
    int cnt  = (int)outCount[oi];
    block[base] = cnt;
    if (cnt <= TNS_MAXNB) {
        device const int* sc = scratch + (uint)i * TNS_MAXNB;
        for (int k = 0; k < cnt; k++) block[base + 1 + k] = sc[k];
    } else {
        sweep_fast_sorted(i, sortedPos, cellStart, P, block + base + 1);
    }
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
    id<MTLComputePipelineState> psoDensity     = nil;
    id<MTLComputePipelineState> psoFactor      = nil;
    id<MTLComputePipelineState> psoDensityAdv  = nil;
    id<MTLComputePipelineState> psoPressureAccel  = nil;
    id<MTLComputePipelineState> psoPressureUpdate = nil;
    id<MTLComputePipelineState> psoNlCount = nil;
    id<MTLComputePipelineState> psoNlFill  = nil;
    id<MTLComputePipelineState> psoPaNl    = nil;
    id<MTLComputePipelineState> psoPuNl    = nil;
    // Parallel scan + sorted-space SPH kernels (resident solve).
    id<MTLComputePipelineState> psoScanPartial  = nil;
    id<MTLComputePipelineState> psoScanBlocks   = nil;
    id<MTLComputePipelineState> psoScanFixup    = nil;
    id<MTLComputePipelineState> psoScatterSph   = nil;
    id<MTLComputePipelineState> psoNlCountSorted = nil;
    id<MTLComputePipelineState> psoSphBuild     = nil;
    id<MTLComputePipelineState> psoXsphNl       = nil;
    id<MTLComputePipelineState> psoAccelScatter3 = nil;
    id<MTLComputePipelineState> psoSearchFastSorted  = nil;
    id<MTLComputePipelineState> psoCompactFastSorted = nil;
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
        if (!make_pso("k_density",      ctx.psoDensity))     return;
        if (!make_pso("k_factor",       ctx.psoFactor))      return;
        if (!make_pso("k_density_adv",  ctx.psoDensityAdv))  return;
        if (!make_pso("k_pressure_accel",  ctx.psoPressureAccel))  return;
        if (!make_pso("k_pressure_update", ctx.psoPressureUpdate)) return;
        if (!make_pso("k_nl_count", ctx.psoNlCount)) return;
        if (!make_pso("k_nl_fill",  ctx.psoNlFill))  return;
        if (!make_pso("k_pa_nl",    ctx.psoPaNl))    return;
        if (!make_pso("k_pu_nl",    ctx.psoPuNl))    return;
        if (!make_pso("k_scan_partial",    ctx.psoScanPartial))   return;
        if (!make_pso("k_scan_blocks",     ctx.psoScanBlocks))    return;
        if (!make_pso("k_scan_fixup",      ctx.psoScanFixup))     return;
        if (!make_pso("k_scatter_sph",     ctx.psoScatterSph))    return;
        if (!make_pso("k_nl_count_sorted", ctx.psoNlCountSorted)) return;
        if (!make_pso("k_sph_build",       ctx.psoSphBuild))      return;
        if (!make_pso("k_xsph_nl",         ctx.psoXsphNl))        return;
        if (!make_pso("k_accel_scatter3",  ctx.psoAccelScatter3)) return;
        if (!make_pso("k_search_fast_sorted",  ctx.psoSearchFastSorted))  return;
        if (!make_pso("k_compact_fast_sorted", ctx.psoCompactFastSorted)) return;

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
    // 256-wide threadgroups: full SIMD occupancy without starving the register
    // file the way max-width (1024) groups can for the heavier sweep kernels.
    NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
    if (tpg > 256) tpg = 256;
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
    id<MTLBuffer> count, scratch;
    id<MTLBuffer> volume, density, bVol, bXj, velocity;
    id<MTLBuffer> velSmooth;                   // XSPH double-buffer (velOut)
    id<MTLBuffer> pRho2, accel, densAdv, factorBuf, err;
    id<MTLBuffer> nlStart, nlId, nlVgw, bVgw;   // precomputed neighbour list for the solve
    id<MTLBuffer> sVol, blockSums;              // sorted volume + parallel-scan scratch
    // Zero-copy neighbour-search results: one packed-block + offsets buffer per
    // search pair, handed to the caller as pointers into unified memory (valid
    // until the next metal_neighbor_search call in the process).
    std::vector<id<MTLBuffer>> blockPair, blockOffPair;
    std::vector<uint32_t> cellStartHost, nlStartHost;
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

// Mirrors the MSL ScanArgs struct.
struct ScanArgsC { uint32_t n, numBlocks, seedCopy, addPer; };

// Encode a parallel exclusive prefix-sum of `in` (n uints) into `out` (n+1 uints;
// out[n] = grand total) onto `cmd`. If `seed` is non-nil the offsets are also
// written there (the grid scatter's running cursor). addPer is added to every
// element (1 -> offsets for packed [count, ids...] runs). Three dispatches, no
// CPU sync; B.blockSums is (re)used as scratch.
static void encode_scan(id<MTLCommandBuffer> cmd, MetalContext& ctx, GpuBuffers& B,
                        id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> seed, uint32_t n,
                        uint32_t addPer = 0)
{
    const uint32_t numBlocks = (n + 255u) / 256u;
    ensure(ctx.device, B.blockSums, sizeof(uint32_t) * (size_t)numBlocks);
    ScanArgsC A{ n, numBlocks, seed != nil ? 1u : 0u, addPer };

    id<MTLComputeCommandEncoder> e1 = [cmd computeCommandEncoder];
    [e1 setComputePipelineState:ctx.psoScanPartial];
    [e1 setBuffer:in offset:0 atIndex:0];
    [e1 setBuffer:out offset:0 atIndex:1];        // reused as per-element partials
    [e1 setBuffer:B.blockSums offset:0 atIndex:2];
    [e1 setBytes:&A length:sizeof(A) atIndex:3];
    [e1 dispatchThreadgroups:MTLSizeMake(numBlocks, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e1 endEncoding];

    id<MTLComputeCommandEncoder> e2 = [cmd computeCommandEncoder];
    [e2 setComputePipelineState:ctx.psoScanBlocks];
    [e2 setBuffer:B.blockSums offset:0 atIndex:0];
    [e2 setBytes:&A length:sizeof(A) atIndex:1];
    [e2 dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e2 endEncoding];

    id<MTLComputeCommandEncoder> e3 = [cmd computeCommandEncoder];
    [e3 setComputePipelineState:ctx.psoScanFixup];
    [e3 setBuffer:out offset:0 atIndex:0];        // partials in place
    [e3 setBuffer:B.blockSums offset:0 atIndex:1];
    [e3 setBuffer:out offset:0 atIndex:2];
    [e3 setBuffer:(seed != nil ? seed : out) offset:0 atIndex:3];
    [e3 setBuffer:in offset:0 atIndex:4];
    [e3 setBytes:&A length:sizeof(A) atIndex:5];
    [e3 dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e3 endEncoding];
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
    double tFill = 0, tGrid = 0, tCount = 0, tWrite = 0;
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

        // ── 1. grid build: hash -> parallel GPU scan -> scatter, ONE command buffer ──
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
            encode_scan(cmd, ctx, B, B.counts, B.cellStart, B.cellOffset, (uint32_t)nCells);
            id<MTLComputeCommandEncoder> esc = [cmd computeCommandEncoder];
            if (fast) {
                [esc setBuffer:B.cellId     offset:0 atIndex:0];
                [esc setBuffer:B.cellOffset offset:0 atIndex:1];
                [esc setBuffer:B.points     offset:0 atIndex:2];
                [esc setBuffer:B.sPoints    offset:0 atIndex:3];
                [esc setBytes:&P length:sizeof(P) atIndex:4];
                dispatch1d(esc, ctx.psoScatterFast, (NSUInteger)N);
            } else {
                [esc setBuffer:B.cellId     offset:0 atIndex:0];
                [esc setBuffer:B.cellOffset offset:0 atIndex:1];
                [esc setBuffer:B.points     offset:0 atIndex:2];
                [esc setBuffer:B.radii      offset:0 atIndex:3];
                [esc setBuffer:B.pointSet   offset:0 atIndex:4];
                [esc setBuffer:B.sPoints    offset:0 atIndex:5];
                [esc setBuffer:B.sRadii     offset:0 atIndex:6];
                [esc setBuffer:B.sSet       offset:0 atIndex:7];
                [esc setBuffer:B.sOrig      offset:0 atIndex:8];
                [esc setBytes:&P length:sizeof(P) atIndex:9];
                dispatch1d(esc, ctx.psoScatter, (NSUInteger)N);
            }
            [esc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        tGrid = pms(th0, pnow());

        // ── 2. per-pair neighbour search ─────────────────────────────────────────
        // Results are ZERO-COPY: the packed [count, ids...] runs and their offsets
        // live in per-pair unified-memory buffers; MetalPairResult carries pointers
        // into them (valid until the next call — the buffers are reused).
        if (B.blockPair.size() < req.pairs.size()) {
            B.blockPair.resize(req.pairs.size(), nil);
            B.blockOffPair.resize(req.pairs.size(), nil);
        }
        out.pairs.clear();
        out.pairs.reserve(req.pairs.size());

        for (size_t pi = 0; pi < req.pairs.size(); pi++) {
            const MetalSearchPair& pr = req.pairs[pi];
            const int nSearch = req.set_offsets[pr.set_i + 1] - req.set_offsets[pr.set_i];

            ParamsC PP = P;
            PP.setI = pr.set_i; PP.setJ = pr.set_j;
            PP.setOffsetI = req.set_offsets[pr.set_i];
            PP.setOffsetJ = req.set_offsets[pr.set_j];
            PP.nSearch = nSearch;

            MetalPairResult res;
            res.set_i = pr.set_i; res.set_j = pr.set_j;

            if (nSearch == 0) { out.pairs.push_back(std::move(res)); continue; }

            ensure(dev, B.count, sizeof(uint32_t) * (size_t)nSearch);
            ensure(dev, B.blockOffPair[pi], sizeof(int) * (size_t)(nSearch + 1));
            id<MTLBuffer> blockOff = B.blockOffPair[pi];

            int blockAcc = 0;
            auto tc0 = pnow();
            if (fast) {
                // Sorted sweep: thread = sorted slot (SIMD-coherent, coalesced),
                // counts keyed by original index; the [count, ids...] offsets come
                // from the GPU scan (count+1 per point) in the same command buffer.
                PP.nSearch = N;   // every slot is a query in the single-set case
                ensure(dev, B.scratch, sizeof(int) * (size_t)N * (size_t)kMaxNb);
                {
                    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setBuffer:B.sPoints   offset:0 atIndex:0];
                    [enc setBuffer:B.cellStart offset:0 atIndex:1];
                    [enc setBuffer:B.scratch   offset:0 atIndex:2];
                    [enc setBuffer:B.count     offset:0 atIndex:3];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:4];
                    dispatch1d(enc, ctx.psoSearchFastSorted, (NSUInteger)N);
                    [enc endEncoding];
                    encode_scan(cmd, ctx, B, B.count, blockOff, nil, (uint32_t)nSearch, /*addPer=*/1);
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
                tCount += pms(tc0, pnow());
                blockAcc = (int)((const uint32_t*)blockOff.contents)[nSearch];

                ensure(dev, B.blockPair[pi], sizeof(int) * (size_t)blockAcc);
                auto tw0 = pnow();
                {
                    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setBuffer:B.sPoints        offset:0 atIndex:0];
                    [enc setBuffer:B.cellStart      offset:0 atIndex:1];
                    [enc setBuffer:B.scratch        offset:0 atIndex:2];
                    [enc setBuffer:B.count          offset:0 atIndex:3];
                    [enc setBuffer:blockOff         offset:0 atIndex:4];
                    [enc setBuffer:B.blockPair[pi]  offset:0 atIndex:5];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:6];
                    dispatch1d(enc, ctx.psoCompactFastSorted, (NSUInteger)N);
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
                tWrite += pms(tw0, pnow());
            } else {
                // General path (multi-set / variable radius): count sweep, CPU scan
                // of the offsets (written straight into the shared buffer), write sweep.
                {
                    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
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
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
                tCount += pms(tc0, pnow());

                const uint32_t* cnt = (const uint32_t*)B.count.contents;
                int* off = (int*)blockOff.contents;
                for (int i = 0; i < nSearch; i++) {
                    off[i] = blockAcc;
                    blockAcc += (int)cnt[i] + 1;
                }
                off[nSearch] = blockAcc;

                ensure(dev, B.blockPair[pi], sizeof(int) * (size_t)blockAcc);
                auto tw0 = pnow();
                {
                    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setBuffer:B.points   offset:0 atIndex:0];
                    [enc setBuffer:B.radii    offset:0 atIndex:1];
                    [enc setBuffer:B.sPoints  offset:0 atIndex:2];
                    [enc setBuffer:B.sRadii   offset:0 atIndex:3];
                    [enc setBuffer:B.sSet     offset:0 atIndex:4];
                    [enc setBuffer:B.sOrig    offset:0 atIndex:5];
                    [enc setBuffer:B.cellStart offset:0 atIndex:6];
                    [enc setBuffer:B.blockPair[pi] offset:0 atIndex:7];
                    [enc setBuffer:blockOff   offset:0 atIndex:8];
                    [enc setBytes:&PP length:sizeof(PP) atIndex:9];
                    dispatch1d(enc, ctx.psoWrite, (NSUInteger)nSearch);
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
                tWrite += pms(tw0, pnow());
            }

            res.block_ptr        = (const int*)B.blockPair[pi].contents;
            res.block_offset_ptr = (const int*)blockOff.contents;
            out.pairs.push_back(std::move(res));
        }
    }

    if (kProfile) {
        std::fprintf(stderr,
            "[TNS_PROFILE] N=%d nCells=%d  fill=%.2f grid=%.2f count=%.2f write=%.2f  total=%.2f ms\n",
            N, nCells, tFill, tGrid, tCount, tWrite,
            tFill + tGrid + tCount + tWrite);
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

    static const bool kProfile = std::getenv("TNS_PROFILE") != nullptr;
    auto pnow = []{ return std::chrono::high_resolution_clock::now(); };
    auto pms  = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto pt0 = pnow();
    double tBuild = 0, tSolve = 0;

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

        // ── Scratch. Everything internal lives in SORTED (cell) order: the scatter
        //    reorders position + volume + original id, the fused build emits the CSR
        //    list, density, factor and the clamped source in ONE traversal, and the
        //    Jacobi state (pRho2 / accel / densAdv / factor) is indexed by sorted slot
        //    so neighbour gathers stay cache-local. Both prefix-sums are parallel GPU
        //    scans. The only host dependency is the one-word pair total nlStart[N],
        //    read back to size nlId/nlVgw (exact CSR, no cap) — hence two command
        //    buffers / two waits. Results are scattered to caller order at the end. ──
        ensure(dev, B.cellId,     sizeof(uint32_t) * (size_t)N);
        ensure(dev, B.counts,     sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.cellStart,  sizeof(uint32_t) * (size_t)(nCells + 1));
        ensure(dev, B.cellOffset, sizeof(uint32_t) * (size_t)nCells);
        ensure(dev, B.sPoints,    sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.sVol,       sizeof(float) * (size_t)N);
        ensure(dev, B.sOrig,      sizeof(int) * (size_t)N);
        ensure(dev, B.density,    sizeof(float) * (size_t)N);
        ensure(dev, B.factorBuf,  sizeof(float) * (size_t)N);
        ensure(dev, B.densAdv,    sizeof(float) * (size_t)N);
        ensure(dev, B.pRho2,      sizeof(float) * (size_t)N);
        ensure(dev, B.accel,      sizeof(float) * 4 * (size_t)N);
        ensure(dev, B.err,        sizeof(float));
        ensure(dev, B.count,      sizeof(uint32_t) * (size_t)N);
        ensure(dev, B.nlStart,    sizeof(uint32_t) * (size_t)(N + 1));
        ensure(dev, B.bVgw,       sizeof(float) * 4);   // bound but unread (no boundary)
        if (doXsph) ensure(dev, B.velSmooth, sizeof(float) * 3 * (size_t)N);

        // Host-side buffer prep. These writes precede [cmd commit], so the GPU sees them
        // (shared storage): zero the per-cell counts (hash accumulates), the pressure
        // (starts at rest) and the density-error accumulator (atomic reduction target).
        std::memset(B.counts.contents, 0, sizeof(uint32_t) * (size_t)nCells);
        std::memset(B.pRho2.contents,  0, sizeof(float) * (size_t)N);
        *(float*)B.err.contents = 0.0f;

        ParamsC Pp = P; Pp.hasBoundary = 0; Pp.viscosity = req.viscosity;   // grid + fused build
        ParamsC Ps = P; Ps.hasBoundary = 0; Ps.dt = req.dt; Ps.density0 = req.density0;
        Ps.divergence = 0; Ps.aij_scale = req.dt * req.dt;                   // pressure solve (h^2)

        // ── Phase encoders. Each appends encoders to a caller-supplied command buffer;
        //    Metal serialises compute encoders within one buffer and auto-tracks buffer
        //    hazards between them, so back-to-back phases need no CPU sync. ──────────────

        // grid: hash (counts pre-zeroed) -> parallel scan -> scatter pos/vol/orig into
        // cell order; then per-slot neighbour count -> parallel scan into nlStart.
        auto enc_grid_count = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> eh = [cmd computeCommandEncoder];
            [eh setBuffer:pts offset:0 atIndex:0];
            [eh setBuffer:B.cellId offset:0 atIndex:1];
            [eh setBuffer:B.counts offset:0 atIndex:2];
            [eh setBytes:&P length:sizeof(P) atIndex:3];
            dispatch1d(eh, ctx.psoHash, (NSUInteger)N);
            [eh endEncoding];
            encode_scan(cmd, ctx, B, B.counts, B.cellStart, B.cellOffset, (uint32_t)nCells);
            id<MTLComputeCommandEncoder> esc = [cmd computeCommandEncoder];
            [esc setBuffer:B.cellId offset:0 atIndex:0];
            [esc setBuffer:B.cellOffset offset:0 atIndex:1];
            [esc setBuffer:pts offset:0 atIndex:2];
            [esc setBuffer:vol offset:0 atIndex:3];
            [esc setBuffer:B.sPoints offset:0 atIndex:4];
            [esc setBuffer:B.sVol offset:0 atIndex:5];
            [esc setBuffer:B.sOrig offset:0 atIndex:6];
            [esc setBytes:&P length:sizeof(P) atIndex:7];
            dispatch1d(esc, ctx.psoScatterSph, (NSUInteger)N);
            [esc endEncoding];
            id<MTLComputeCommandEncoder> ec = [cmd computeCommandEncoder];
            [ec setBuffer:B.sPoints offset:0 atIndex:0];
            [ec setBuffer:B.cellStart offset:0 atIndex:1];
            [ec setBuffer:B.count offset:0 atIndex:2];
            [ec setBytes:&Pp length:sizeof(Pp) atIndex:3];
            dispatch1d(ec, ctx.psoNlCountSorted, (NSUInteger)N);
            [ec endEncoding];
            encode_scan(cmd, ctx, B, B.count, B.nlStart, nil, (uint32_t)N);
        };

        // Fused build: CSR list + density + factor + clamped source, one traversal.
        auto enc_build = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> eb = [cmd computeCommandEncoder];
            [eb setBuffer:B.sPoints offset:0 atIndex:0];
            [eb setBuffer:B.sVol offset:0 atIndex:1];
            [eb setBuffer:B.cellStart offset:0 atIndex:2];
            [eb setBuffer:B.nlStart offset:0 atIndex:3];
            [eb setBuffer:B.nlId offset:0 atIndex:4];
            [eb setBuffer:B.nlVgw offset:0 atIndex:5];
            [eb setBuffer:B.density offset:0 atIndex:6];
            [eb setBuffer:B.factorBuf offset:0 atIndex:7];
            [eb setBuffer:B.densAdv offset:0 atIndex:8];
            [eb setBytes:&Pp length:sizeof(Pp) atIndex:9];
            dispatch1d(eb, ctx.psoSphBuild, (NSUInteger)N);
            [eb endEncoding];
        };

        // optional XSPH from the CSR list (V_j*W precomputed in nlVgw.w): smooth into
        // velSmooth (reads pre-smoothing vel), blit back into the caller's vel.
        auto enc_xsph = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:B.nlStart offset:0 atIndex:0];
            [enc setBuffer:B.nlId offset:0 atIndex:1];
            [enc setBuffer:B.nlVgw offset:0 atIndex:2];
            [enc setBuffer:B.sOrig offset:0 atIndex:3];
            [enc setBuffer:vel offset:0 atIndex:4];   // velIn (caller order)
            [enc setBuffer:B.density offset:0 atIndex:5];
            [enc setBuffer:B.velSmooth offset:0 atIndex:6];   // velOut (caller order)
            [enc setBytes:&Pp length:sizeof(Pp) atIndex:7];
            dispatch1d(enc, ctx.psoXsphNl, (NSUInteger)N);
            [enc endEncoding];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:B.velSmooth sourceOffset:0 toBuffer:vel destinationOffset:0
                            size:sizeof(float) * 3 * (size_t)N];
            [blit endEncoding];
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
        // final accel from the converged p~, scattered into the caller's float3 buffer.
        auto enc_final = [&](id<MTLCommandBuffer> cmd){
            id<MTLComputeCommandEncoder> ea = [cmd computeCommandEncoder]; enc_accel(ea); [ea endEncoding];
            id<MTLComputeCommandEncoder> ecp = [cmd computeCommandEncoder];
            [ecp setBuffer:B.accel offset:0 atIndex:0];
            [ecp setBuffer:B.sOrig offset:0 atIndex:1];
            [ecp setBuffer:out     offset:0 atIndex:2];
            [ecp setBytes:&Ps length:sizeof(Ps) atIndex:3];
            dispatch1d(ecp, ctx.psoAccelScatter3, (NSUInteger)N);
            [ecp endEncoding];
        };

        int maxIt = req.max_iterations > 0 ? req.max_iterations : 1;
        int minIt = req.min_iterations < 0 ? 0 : req.min_iterations;
        if (minIt > maxIt) minIt = maxIt;
        const bool earlyOut = (req.eta > 0.0f && minIt < maxIt);

        // CB1: grid -> scatter -> neighbour count -> scan. One wait, then read the
        // single word nlStart[N] (the pair total) and grow nlId/nlVgw to exactly
        // that. CB1 has drained, so the realloc cannot race.
        auto build_and_size_nl = [&]{
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            enc_grid_count(cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
            const uint32_t tot = ((const uint32_t*)B.nlStart.contents)[N];
            const size_t p = (size_t)(tot > 0 ? tot : 1);
            ensure(dev, B.nlId,  sizeof(int)   * p);
            ensure(dev, B.nlVgw, sizeof(float) * 4 * p);
        };

        if (!earlyOut) {
            // Two command buffers, two waits: CB1 sizes the CSR list; CB2 = fused
            // build -> [XSPH] -> all Jacobi iterations -> final accel scatter.
            build_and_size_nl();
            tBuild = pms(pt0, pnow());
            id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
            enc_build(cmd);
            if (doXsph) enc_xsph(cmd);
            enc_jacobi(cmd, maxIt, /*zeroErr=*/false);
            enc_final(cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
            tSolve = pms(pt0, pnow()) - tBuild;
        } else {
            // Early-out requested: the convergence readback forces splitting the Jacobi
            // loop. CB1 sizes the list; the fused build (+XSPH) rides the first chunk's
            // buffer; then chunked Jacobi (one wait per chunk) and the final scatter.
            build_and_size_nl();
            tBuild = pms(pt0, pnow());
            const int kSolveChunk = 4;
            int done = 0;
            bool filled = false;
            while (done < maxIt) {
                int g = (maxIt - done < kSolveChunk) ? (maxIt - done) : kSolveChunk;
                id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
                if (!filled) { enc_build(cmd); if (doXsph) enc_xsph(cmd); filled = true; }
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
            tSolve = pms(pt0, pnow()) - tBuild;
        }

        // Leave no reusable cache behind: the resident grid/NL scratch is fluid-only,
        // sorted-space, and must not be picked up by a later host-pointer solve.
        g_gridValid = false;
        g_nlValid   = false;
    }
    if (kProfile) {
        std::fprintf(stderr, "[TNS_PROFILE] solve_gpu N=%d  build=%.2f solve=%.2f total=%.2f ms\n",
                     N, tBuild, tSolve, tBuild + tSolve);
    }
    return true;
}

}} // namespace tns::internals
