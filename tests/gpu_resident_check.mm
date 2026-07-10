// ─────────────────────────────────────────────────────────────────────────────
// Validation for the GPU-resident SPH solve (tns::internals::metal_sph_solve_gpu).
//
// Creates a caller-owned Metal device + command queue, injects it via
// metal_sph_set_external_context(), keeps particle state in caller id<MTLBuffer>s,
// runs the resident solve, and checks the pressure acceleration against the proven
// host-pointer path (metal_compute_density -> clamp -> metal_compute_factor ->
// metal_pressure_solve) on identical inputs and an identical iteration count. Same
// kernels + same math, so the two must agree to floating-point tolerance.
//
// Compiled as Objective-C++ with ARC (see tests/CMakeLists.txt). Entry point
// run_resident_checks() is called from gpu_tests.cpp.
// ─────────────────────────────────────────────────────────────────────────────
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "internals/metal_nsearch.h"

namespace {

std::vector<float> rand_vec(int count, float lo, float hi, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(count);
    for (auto& x : v) x = d(rng);
    return v;
}

// One resident-vs-host comparison at n particles. Returns the failure count (0 = pass).
int check_resident(id<MTLDevice> dev, id<MTLCommandQueue> queue, int n)
{
    using namespace tns::internals;
    const float h = 0.6f, dt = 0.0025f, density0 = 1000.0f;
    const int   ITERS = 6;
    std::string err;

    // Random particles + per-particle volume (same recipe as gpu_tests density_check).
    std::vector<float> pts3 = rand_vec(3 * n, 0.0f, 10.0f, 101);
    std::vector<float> vol  = rand_vec(n, 0.8f, 1.2f, 102);
    for (auto& x : vol) x *= 0.02f;

    // Background grid: bbox with cell = h. Identical origin/dims feed BOTH paths.
    float mn[3] = { 1e30f,  1e30f,  1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) { float v = pts3[3*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
    float origin[3]; int dims[3];
    for (int d = 0; d < 3; d++) { origin[d] = mn[d]; dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1; }

    // ── Resident path: caller buffers on the injected device ─────────────────────
    metal_sph_set_external_context((__bridge void*)dev, (__bridge void*)queue);

    id<MTLBuffer> pBuf = [dev newBufferWithLength:sizeof(float) * 4 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    float* p4 = (float*)pBuf.contents;
    for (int i = 0; i < n; i++) {
        p4[4*i+0] = pts3[3*i+0]; p4[4*i+1] = pts3[3*i+1]; p4[4*i+2] = pts3[3*i+2]; p4[4*i+3] = 1.0f;
    }
    std::memcpy(vBuf.contents, vol.data(), sizeof(float) * n);
    std::memset(aBuf.contents, 0, sizeof(float) * 3 * n);

    MetalSphGpuRequest req;
    req.points   = (__bridge void*)pBuf;
    req.volume   = (__bridge void*)vBuf;
    req.outAccel = (__bridge void*)aBuf;
    req.n = n; req.h = h; req.dt = dt; req.density0 = density0;
    for (int d = 0; d < 3; d++) { req.origin[d] = origin[d]; req.grid_dims[d] = dims[d]; }
    req.min_iterations = ITERS; req.max_iterations = ITERS; req.eta = 0.0f;   // fixed-count solve
    if (!metal_sph_solve_gpu(req, err)) {
        std::printf("  n=%d: resident solve FAILED: %s\n", n, err.c_str());
        return 1;
    }
    const float* aRes = (const float*)aBuf.contents;

    // ── Host-pointer path on identical inputs (the reference) ────────────────────
    metal_sph_invalidate_grid();
    std::vector<float> density(n), factor(n), densAdv(n), pRho2(n, 0.0f), aHost(3 * n, 0.0f);

    MetalDensityRequest dreq;
    dreq.points = pts3.data(); dreq.volume = vol.data(); dreq.n = n; dreq.h = h;
    for (int d = 0; d < 3; d++) { dreq.origin[d] = origin[d]; dreq.grid_dims[d] = dims[d]; }
    if (!metal_compute_density(dreq, density.data(), err)) { std::printf("  n=%d: host density FAILED: %s\n", n, err.c_str()); return 1; }
    dreq.reuse_grid = true;
    if (!metal_compute_factor(dreq, factor.data(), err)) { std::printf("  n=%d: host factor FAILED: %s\n", n, err.c_str()); return 1; }
    for (int i = 0; i < n; i++) { float d = density[i]; densAdv[i] = (std::isfinite(d) && d > 1.0f) ? d : 1.0f; }

    MetalPressureRequest preq;
    preq.points = pts3.data(); preq.volume = vol.data();
    preq.densityAdv = densAdv.data(); preq.factor = factor.data(); preq.pressure_rho2 = pRho2.data();
    preq.n = n; preq.h = h; preq.dt = dt; preq.density0 = density0;
    for (int d = 0; d < 3; d++) { preq.origin[d] = origin[d]; preq.grid_dims[d] = dims[d]; }
    preq.iterations = ITERS;   // min/max 0 -> exactly ITERS fixed iterations
    preq.reuse_grid = true;
    if (!metal_pressure_solve(preq, aHost.data(), nullptr, err)) { std::printf("  n=%d: host pressure FAILED: %s\n", n, err.c_str()); return 1; }

    // ── Compare the pressure accel ───────────────────────────────────────────────
    double maxAbs = 0.0, maxRel = 0.0, refMag = 0.0;
    for (int i = 0; i < 3 * n; i++) {
        double r = aHost[i], g = aRes[i];
        maxAbs = std::max(maxAbs, std::abs(g - r));
        maxRel = std::max(maxRel, std::abs(g - r) / (std::abs(r) + 1e-6));
        refMag = std::max(refMag, std::abs(r));
    }
    const bool pass = (maxRel < 1e-3) || (maxAbs < 1e-4 * (refMag + 1e-9));
    std::printf("  n=%d: accel max abs err %.3e, max rel err %.3e (|accel|max %.3e) ... %s\n",
                n, maxAbs, maxRel, refMag, pass ? "passed!" : "FAILED!");

    // The host reference path above leaves the cached neighbour list valid (g_nlValid).
    // Clear it so a later independent solve (e.g. gpu_tests' density_check, which relies
    // on the cache starting empty) rebuilds its own list instead of reusing ours.
    metal_sph_invalidate_grid();
    return pass ? 0 : 1;
}

// Pressure warm-start: two consecutive solves on a slightly-advected dense
// cloud, with the chunked early-out enabled (min < max, eta > 0 — this is also
// the only check that drives that branch of metal_sph_solve_gpu). The second
// solve, seeded with the first solve's converged pressure via req.pressure,
// must (a) converge in no more iterations than the same solve run cold, and
// (b) land on the same pressure accel — a Jacobi seed cannot change the fixed
// point, only how fast it is reached.
int check_warmstart(id<MTLDevice> dev, id<MTLCommandQueue> queue, int n)
{
    using namespace tns::internals;
    // SPH-like block, not a random cloud: a jittered unit lattice compressed a
    // few percent (h = 2*spacing, ~33 neighbours, density ~1.09) is a state the
    // Jacobi solve actually converges from — a random cloud is 300%+ compressed
    // and hits max_iterations from any seed, which makes warm-vs-cold moot.
    // dt is in lattice units and picked so the solve CONVERGES in tens of
    // iterations (this formulation's Jacobi contraction goes as ~0.5*dt^2 per
    // iteration in these units — measured: dt 0.25/0.5/1.0 -> 404/104/28
    // iterations to 0.1%); a converged cold reference is what makes the
    // warm-vs-cold comparison meaningful.
    const float h = 2.0f, dt = 1.0f, density0 = 1000.0f, compress = 0.97f;
    const float eta = 1.0e-3f * density0;   // 0.1% avg density error (engine value)
    std::string err;

    const int side = (int)std::ceil(std::cbrt((double)n));
    std::vector<float> pts3(3 * (size_t)n);
    {
        std::mt19937 rng(401);
        std::uniform_real_distribution<float> jit(-0.2f, 0.2f);
        int p = 0;
        for (int z = 0; z < side && p < n; z++)
            for (int y = 0; y < side && p < n; y++)
                for (int x = 0; x < side && p < n; x++, p++) {
                    pts3[3*p+0] = ((float)x + jit(rng)) * compress;
                    pts3[3*p+1] = ((float)y + jit(rng)) * compress;
                    pts3[3*p+2] = ((float)z + jit(rng)) * compress;
                }
    }
    std::vector<float> vol(n, 1.0f);   // spacing^3 -> rest-normalised density ~1
    // One substep of drift, a few percent of the particle spacing.
    std::vector<float> drift = rand_vec(3 * n, -0.03f, 0.03f, 403);

    metal_sph_set_external_context((__bridge void*)dev, (__bridge void*)queue);
    id<MTLBuffer> pBuf  = [dev newBufferWithLength:sizeof(float) * 4 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf  = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf  = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> prBuf = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    std::memcpy(vBuf.contents, vol.data(), sizeof(float) * n);

    auto load_positions = [&](bool advected){
        float* p4 = (float*)pBuf.contents;
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 3; d++)
                p4[4*i+d] = pts3[3*i+d] + (advected ? drift[3*i+d] : 0.0f);
            p4[4*i+3] = 1.0f;
        }
    };
    auto solve = [&](bool warm, int& iters_out) -> bool {
        std::memset(aBuf.contents, 0, sizeof(float) * 3 * n);
        // Grid over the advected bbox (covers both position sets).
        float mn[3] = { 1e30f,  1e30f,  1e30f};
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        const float* p4 = (const float*)pBuf.contents;
        for (int i = 0; i < n; i++)
            for (int d = 0; d < 3; d++) { float v = p4[4*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
        MetalSphGpuRequest req;
        req.points   = (__bridge void*)pBuf;
        req.volume   = (__bridge void*)vBuf;
        req.outAccel = (__bridge void*)aBuf;
        req.pressure = warm ? (__bridge void*)prBuf : nullptr;
        req.n = n; req.h = h; req.dt = dt; req.density0 = density0;
        for (int d = 0; d < 3; d++) {
            req.origin[d] = mn[d];
            req.grid_dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1;
        }
        // Room for the cold solve to converge (~28 iterations at dt=1); the warm
        // seed already contains the solved global compression mode, which is
        // precisely the effect under test.
        req.min_iterations = 4; req.max_iterations = 100; req.eta = eta;
        req.iterations_out = &iters_out;
        return metal_sph_solve_gpu(req, err);
    };

    int itSeed = 0, itCold = 0, itWarm = 0;
    // Solve 1: original positions, cold, converged pressure lands in prBuf.
    std::memset(prBuf.contents, 0, sizeof(float) * n);
    load_positions(false);
    if (!solve(true, itSeed)) { std::printf("  warmstart n=%d: seed solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }
    // Solve 2a: advected positions, cold reference.
    load_positions(true);
    if (!solve(false, itCold)) { std::printf("  warmstart n=%d: cold solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }
    std::vector<float> aCold((const float*)aBuf.contents, (const float*)aBuf.contents + 3 * n);
    // Solve 2b: same advected positions, warm-started from solve 1's pressure.
    if (!solve(true, itWarm)) { std::printf("  warmstart n=%d: warm solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }
    const float* aWarm = (const float*)aBuf.contents;

    double maxAbs = 0.0, refMag = 0.0;
    for (int i = 0; i < 3 * n; i++) {
        maxAbs = std::max(maxAbs, (double)std::abs(aWarm[i] - aCold[i]));
        refMag = std::max(refMag, (double)std::abs(aCold[i]));
    }
    // Both runs converged to avg density error <= eta; they are different
    // iterates of the same fixed point, so demand agreement at the few-percent
    // level of the accel scale (and that the test isn't vacuous: refMag > 0).
    const bool pass = (refMag > 0.0) && (itWarm <= itCold) && (maxAbs < 0.05 * refMag);
    std::printf("  warmstart n=%d: iters cold %d -> warm %d, accel max abs diff %.3e (|accel|max %.3e) ... %s\n",
                n, itCold, itWarm, maxAbs, refMag, pass ? "passed!" : "FAILED!");
    metal_sph_invalidate_grid();
    return pass ? 0 : 1;
}

// Fast-path overflow retry: solve a sparse cloud (primes the CSR cache at 1.5x
// its pair total, so the next call takes the single-command-buffer fast path by
// construction), then compress the SAME particles to ~15x the pair density. The
// fast path's capacity clamp must overflow, and the transparent retry must
// deliver exactly what a from-scratch solve would: accel checked against the
// host-pointer reference, and the XSPH-smoothed velocity against a CPU
// brute-force reference (a broken retry double-smooths the caller's velocity —
// the accel check alone would not catch that). Runs the chunked early-out
// config (eta tiny, never met -> deterministic ITERS iterations, comparable to
// the fixed-count host reference) so the overflow lands in the chunked branch.
double cubicW(double r, double h);   // defined below
int check_overflow_retry(id<MTLDevice> dev, id<MTLCommandQueue> queue, int n)
{
    using namespace tns::internals;
    const float h = 0.6f, dt = 0.0025f, density0 = 1000.0f, visc = 0.2f;
    const float shrink = 0.4f;   // [0,10]^3 -> [0,4]^3: ~15x the pair total
    const int   ITERS = 6;
    std::string err;

    std::vector<float> pts3 = rand_vec(3 * n, 0.0f, 10.0f, 501);
    std::vector<float> vol  = rand_vec(n, 0.8f, 1.2f, 502);
    for (auto& x : vol) x *= 0.02f;
    std::vector<float> vel0 = rand_vec(3 * n, -1.0f, 1.0f, 503);

    metal_sph_set_external_context((__bridge void*)dev, (__bridge void*)queue);
    id<MTLBuffer> pBuf   = [dev newBufferWithLength:sizeof(float) * 4 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf   = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    id<MTLBuffer> velBuf = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf   = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> prBuf  = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    std::memcpy(vBuf.contents, vol.data(), sizeof(float) * n);

    auto reset_state = [&]{
        float* p4 = (float*)pBuf.contents;
        for (int i = 0; i < n; i++) {
            p4[4*i+0] = pts3[3*i+0]; p4[4*i+1] = pts3[3*i+1]; p4[4*i+2] = pts3[3*i+2]; p4[4*i+3] = 1.0f;
        }
        std::memcpy(velBuf.contents, vel0.data(), sizeof(float) * 3 * n);
        std::memset(aBuf.contents, 0, sizeof(float) * 3 * n);
        std::memset(prBuf.contents, 0, sizeof(float) * n);
    };
    auto solve = [&]() -> bool {
        float mn[3] = { 1e30f,  1e30f,  1e30f};
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < n; i++)
            for (int d = 0; d < 3; d++) { float v = pts3[3*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
        MetalSphGpuRequest req;
        req.points   = (__bridge void*)pBuf;
        req.volume   = (__bridge void*)vBuf;
        req.velocity = (__bridge void*)velBuf;
        req.outAccel = (__bridge void*)aBuf;
        req.pressure = (__bridge void*)prBuf;
        req.n = n; req.h = h; req.dt = dt; req.density0 = density0;
        req.viscosity = visc;
        for (int d = 0; d < 3; d++) {
            req.origin[d] = mn[d];
            req.grid_dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1;
        }
        req.min_iterations = 1; req.max_iterations = ITERS; req.eta = 1.0e-9f;
        return metal_sph_solve_gpu(req, err);
    };

    reset_state();
    if (!solve()) { std::printf("  overflow n=%d: priming solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }

    for (auto& x : pts3) x *= shrink;
    reset_state();
    if (!solve()) { std::printf("  overflow n=%d: compressed solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }
    std::vector<float> aRes((const float*)aBuf.contents, (const float*)aBuf.contents + 3 * n);
    std::vector<float> vRes((const float*)velBuf.contents, (const float*)velBuf.contents + 3 * n);

    // Host-pointer reference on the compressed cloud (accel is XSPH-independent).
    metal_sph_invalidate_grid();
    float origin[3]; int dims[3];
    {
        float mn[3] = { 1e30f,  1e30f,  1e30f};
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < n; i++)
            for (int d = 0; d < 3; d++) { float v = pts3[3*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
        for (int d = 0; d < 3; d++) { origin[d] = mn[d]; dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1; }
    }
    std::vector<float> density(n), factor(n), densAdv(n), pRho2(n, 0.0f), aHost(3 * n, 0.0f);
    MetalDensityRequest dreq;
    dreq.points = pts3.data(); dreq.volume = vol.data(); dreq.n = n; dreq.h = h;
    for (int d = 0; d < 3; d++) { dreq.origin[d] = origin[d]; dreq.grid_dims[d] = dims[d]; }
    if (!metal_compute_density(dreq, density.data(), err)) { std::printf("  overflow n=%d: host density FAILED\n", n); return 1; }
    dreq.reuse_grid = true;
    if (!metal_compute_factor(dreq, factor.data(), err)) { std::printf("  overflow n=%d: host factor FAILED\n", n); return 1; }
    for (int i = 0; i < n; i++) { float d = density[i]; densAdv[i] = (std::isfinite(d) && d > 1.0f) ? d : 1.0f; }
    MetalPressureRequest preq;
    preq.points = pts3.data(); preq.volume = vol.data();
    preq.densityAdv = densAdv.data(); preq.factor = factor.data(); preq.pressure_rho2 = pRho2.data();
    preq.n = n; preq.h = h; preq.dt = dt; preq.density0 = density0;
    for (int d = 0; d < 3; d++) { preq.origin[d] = origin[d]; preq.grid_dims[d] = dims[d]; }
    preq.iterations = ITERS;
    preq.reuse_grid = true;
    if (!metal_pressure_solve(preq, aHost.data(), nullptr, err)) { std::printf("  overflow n=%d: host pressure FAILED\n", n); return 1; }

    double maxAbsA = 0.0, refMagA = 0.0;
    for (int i = 0; i < 3 * n; i++) {
        maxAbsA = std::max(maxAbsA, (double)std::abs(aRes[i] - aHost[i]));
        refMagA = std::max(refMagA, (double)std::abs(aHost[i]));
    }
    const bool passA = (refMagA > 0.0) && (maxAbsA < 1e-3 * refMagA);

    // Brute-force XSPH reference from the original vel0 on compressed positions.
    const double W0 = cubicW(0.0, h);
    double maxAbsV = 0.0, refMagV = 0.0;
    for (int i = 0; i < n; i++) {
        double di = (double)vol[i] * W0;
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double dx=pts3[3*i+0]-pts3[3*j+0], dy=pts3[3*i+1]-pts3[3*j+1], dz=pts3[3*i+2]-pts3[3*j+2];
            double r = std::sqrt(dx*dx+dy*dy+dz*dz);
            if (r > h) continue;
            double w = cubicW(r, h);
            di += (double)vol[j] * w;
            ax += (double)vol[j] * (vel0[3*j+0]-vel0[3*i+0]) * w;
            ay += (double)vol[j] * (vel0[3*j+1]-vel0[3*i+1]) * w;
            az += (double)vol[j] * (vel0[3*j+2]-vel0[3*i+2]) * w;
        }
        const double inv = 1.0 / std::max(di, 1e-6);
        const double ref[3] = { vel0[3*i+0] + visc*ax*inv, vel0[3*i+1] + visc*ay*inv, vel0[3*i+2] + visc*az*inv };
        for (int d = 0; d < 3; d++) {
            maxAbsV = std::max(maxAbsV, (double)std::abs(vRes[3*i+d] - ref[d]));
            refMagV = std::max(refMagV, std::abs(ref[d]));
        }
    }
    const bool passV = maxAbsV < 1e-3 * refMagV;

    const bool pass = passA && passV;
    std::printf("  overflow-retry n=%d (~15x pair growth): accel max abs diff %.3e (|a|max %.3e), "
                "vel max abs err %.3e ... %s\n",
                n, maxAbsA, refMagA, maxAbsV, pass ? "passed!" : "FAILED!");
    metal_sph_invalidate_grid();
    return pass ? 0 : 1;
}

// CubicKernel W (matches k_density / SPlisHSPlasH SPHKernels.h).
double cubicW(double r, double h)
{
    const double kk = 8.0 / (3.14159265358979 * h * h * h);
    const double q = r / h;
    if (q > 1.0) return 0.0;
    if (q <= 0.5) { const double q2 = q*q; return kk * (6.0*q2*q - 6.0*q2 + 1.0); }
    const double f = 1.0 - q;
    return kk * (2.0*f*f*f);
}

// XSPH viscosity check: run the resident solve with viscosity>0 + a velocity buffer,
// then compare the smoothed velocity against a CPU brute-force reference:
//   v_i' = v_i + visc * [Sum_j V_j (v_j - v_i) W(|x_i-x_j|)] / rho_i   (rho_i = rel. density)
int check_xsph(id<MTLDevice> dev, id<MTLCommandQueue> queue, int n)
{
    using namespace tns::internals;
    const float h = 0.6f, dt = 0.0025f, density0 = 1000.0f, visc = 0.2f;
    std::string err;

    std::vector<float> pts3 = rand_vec(3 * n, 0.0f, 10.0f, 201);
    std::vector<float> vol  = rand_vec(n, 0.8f, 1.2f, 202);
    for (auto& x : vol) x *= 0.02f;
    std::vector<float> vel0 = rand_vec(3 * n, -1.0f, 1.0f, 203);   // pre-smoothing velocities

    float mn[3] = { 1e30f,  1e30f,  1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) { float v = pts3[3*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }
    float origin[3]; int dims[3];
    for (int d = 0; d < 3; d++) { origin[d] = mn[d]; dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1; }

    metal_sph_set_external_context((__bridge void*)dev, (__bridge void*)queue);
    id<MTLBuffer> pBuf   = [dev newBufferWithLength:sizeof(float) * 4 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf   = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    id<MTLBuffer> velBuf = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf   = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    float* p4 = (float*)pBuf.contents;
    for (int i = 0; i < n; i++) { p4[4*i+0]=pts3[3*i+0]; p4[4*i+1]=pts3[3*i+1]; p4[4*i+2]=pts3[3*i+2]; p4[4*i+3]=1.0f; }
    std::memcpy(vBuf.contents,   vol.data(),  sizeof(float) * n);
    std::memcpy(velBuf.contents, vel0.data(), sizeof(float) * 3 * n);
    std::memset(aBuf.contents, 0, sizeof(float) * 3 * n);

    MetalSphGpuRequest req;
    req.points = (__bridge void*)pBuf; req.volume = (__bridge void*)vBuf;
    req.velocity = (__bridge void*)velBuf; req.outAccel = (__bridge void*)aBuf;
    req.n = n; req.h = h; req.dt = dt; req.density0 = density0;
    for (int d = 0; d < 3; d++) { req.origin[d] = origin[d]; req.grid_dims[d] = dims[d]; }
    req.min_iterations = 1; req.max_iterations = 4; req.eta = 0.0f;
    req.viscosity = visc;
    if (!metal_sph_solve_gpu(req, err)) { std::printf("  xsph n=%d: solve FAILED: %s\n", n, err.c_str()); metal_sph_invalidate_grid(); return 1; }
    const float* vGpu = (const float*)velBuf.contents;   // smoothed in place

    const double W0 = cubicW(0.0, h);
    double maxAbs = 0.0, maxRel = 0.0, refMag = 0.0;
    for (int i = 0; i < n; i++) {
        double di = (double)vol[i] * W0;
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double dx=pts3[3*i+0]-pts3[3*j+0], dy=pts3[3*i+1]-pts3[3*j+1], dz=pts3[3*i+2]-pts3[3*j+2];
            double r = std::sqrt(dx*dx+dy*dy+dz*dz);
            if (r > h) continue;
            double w = cubicW(r, h);
            di += (double)vol[j] * w;
            ax += (double)vol[j] * (vel0[3*j+0]-vel0[3*i+0]) * w;
            ay += (double)vol[j] * (vel0[3*j+1]-vel0[3*i+1]) * w;
            az += (double)vol[j] * (vel0[3*j+2]-vel0[3*i+2]) * w;
        }
        const double inv = 1.0 / std::max(di, 1e-6);
        const double ref[3] = { vel0[3*i+0] + visc*ax*inv, vel0[3*i+1] + visc*ay*inv, vel0[3*i+2] + visc*az*inv };
        for (int d = 0; d < 3; d++) {
            double g = vGpu[3*i+d], rr = ref[d];
            maxAbs = std::max(maxAbs, std::abs(g - rr));
            maxRel = std::max(maxRel, std::abs(g - rr) / (std::abs(rr) + 1e-6));
            refMag = std::max(refMag, std::abs(rr));
        }
    }
    const bool pass = (maxRel < 1e-3) || (maxAbs < 1e-4 * (refMag + 1e-9));
    std::printf("  xsph n=%d (visc %.2f): vel max abs err %.3e, max rel err %.3e ... %s\n",
                n, visc, maxAbs, maxRel, pass ? "passed!" : "FAILED!");
    metal_sph_invalidate_grid();
    return pass ? 0 : 1;
}

}  // namespace

extern "C" int run_resident_checks();
int run_resident_checks()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
            if (all.count > 0) dev = all.firstObject;
        }
        if (!dev) { std::printf("  no Metal device; resident checks skipped\n"); return 0; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        std::printf("\n=== GPU-resident solve vs host-pointer path ===\n");
        int fails = 0;
        fails += check_resident(dev, queue, 3000);
        fails += check_resident(dev, queue, 20000);
        fails += check_warmstart(dev, queue, 3000);
        fails += check_overflow_retry(dev, queue, 3000);
        fails += check_xsph(dev, queue, 3000);
        return fails;
    }
}
