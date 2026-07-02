// ─────────────────────────────────────────────────────────────────────────────
// Timing harness for the GPU-resident SPH solve (tns::internals::metal_sph_solve_gpu).
//
// Mimics the real per-substep call: an SPH-like particle block (jittered cubic
// lattice, h = 2 * spacing, ~30-40 neighbours/particle) resident in caller
// MTLBuffers, solved with a fixed Jacobi iteration count. Reports the median and
// best wall time per call at several N and iteration counts, so the fixed cost
// (grid + density/factor + neighbour-list build) and the per-iteration cost can
// be separated: cost(iters) ~ fixed + iters * slope.
//
// Build: APPLE-only target `gpu_bench` (see tests/CMakeLists.txt).
// Env: TNS_PROFILE=1 additionally prints the solve's internal phase timings.
// ─────────────────────────────────────────────────────────────────────────────
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "internals/metal_nsearch.h"

namespace {

double now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// SPH-like block: n particles on a jittered cubic lattice with unit spacing,
// support radius h = 2 (lattice units) -> ~33 neighbours per interior particle.
void make_block(int n, std::vector<float>& pts3, float& h, float& particleVolume)
{
    const int side = (int)std::ceil(std::cbrt((double)n));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> jit(-0.2f, 0.2f);
    pts3.resize(3 * (size_t)n);
    int p = 0;
    for (int z = 0; z < side && p < n; z++)
        for (int y = 0; y < side && p < n; y++)
            for (int x = 0; x < side && p < n; x++, p++) {
                pts3[3 * (size_t)p + 0] = (float)x + jit(rng);
                pts3[3 * (size_t)p + 1] = (float)y + jit(rng);
                pts3[3 * (size_t)p + 2] = (float)z + jit(rng);
            }
    h = 2.0f;              // support radius = 2 * spacing (typical SPH)
    particleVolume = 1.0f; // spacing^3 -> rest-normalised density ~ 1
}

struct BenchResult { double median_ms = 0, best_ms = 0; };

BenchResult bench_solve(id<MTLDevice> dev, int n, int iters, int reps)
{
    using namespace tns::internals;
    std::vector<float> pts3; float h = 0, vol0 = 0;
    make_block(n, pts3, h, vol0);

    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++)
        for (int d = 0; d < 3; d++) { float v = pts3[3*i+d]; mn[d] = std::min(mn[d], v); mx[d] = std::max(mx[d], v); }

    id<MTLBuffer> pBuf = [dev newBufferWithLength:sizeof(float) * 4 * n options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf = [dev newBufferWithLength:sizeof(float) * n     options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf = [dev newBufferWithLength:sizeof(float) * 3 * n options:MTLResourceStorageModeShared];
    float* p4 = (float*)pBuf.contents;
    for (int i = 0; i < n; i++) { p4[4*i+0]=pts3[3*i+0]; p4[4*i+1]=pts3[3*i+1]; p4[4*i+2]=pts3[3*i+2]; p4[4*i+3]=0.0f; }
    float* vv = (float*)vBuf.contents;
    for (int i = 0; i < n; i++) vv[i] = vol0;

    MetalSphGpuRequest req;
    req.points = (__bridge void*)pBuf; req.volume = (__bridge void*)vBuf; req.outAccel = (__bridge void*)aBuf;
    req.n = n; req.h = h; req.dt = 0.0025f; req.density0 = 1000.0f;
    for (int d = 0; d < 3; d++) {
        req.origin[d] = mn[d];
        req.grid_dims[d] = (int)std::floor((mx[d] - mn[d]) / h) + 1;
    }
    req.min_iterations = iters; req.max_iterations = iters; req.eta = 0.0f;   // fixed-count

    std::string err;
    if (!metal_sph_solve_gpu(req, err)) {   // warm-up (also sizes the buffer cache)
        std::printf("  n=%d: solve FAILED: %s\n", n, err.c_str());
        return {};
    }

    std::vector<double> times;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_ms();
        metal_sph_solve_gpu(req, err);
        times.push_back(now_ms() - t0);
    }
    std::sort(times.begin(), times.end());
    return { times[times.size() / 2], times.front() };
}

} // namespace

int main()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
            if (all.count > 0) dev = all.firstObject;
        }
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        tns::internals::metal_sph_set_external_context((__bridge void*)dev, (__bridge void*)queue);

        std::printf("GPU-resident solve wall time per call (SPH-like block, ~33 nbrs/particle)\n");
        std::printf("%10s %6s %12s %12s\n", "N", "iters", "median(ms)", "best(ms)");
        for (int n : {3000, 20000, 65536, 131072, 262144}) {
            for (int iters : {2, 6, 12}) {
                auto r = bench_solve(dev, n, iters, 15);
                std::printf("%10d %6d %12.3f %12.3f\n", n, iters, r.median_ms, r.best_ms);
            }
        }
    }
    return 0;
}
