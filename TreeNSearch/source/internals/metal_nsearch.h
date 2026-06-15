#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Metal GPU backend for TreeNSearch (Apple Silicon).
//
// This header is PURE C++ on purpose: it is included by TreeNSearch.cpp and the
// only thing it exposes is a plain function operating on POD / std::vector data.
// All Objective-C / Metal code lives in metal_nsearch.mm so that the rest of the
// codebase never needs an Objective-C++ compiler.
//
// Algorithm: classic GPU uniform-grid neighbour search.
//   1. hash every point into a background grid cell                (GPU)
//   2. counting-sort point indices by cell                         (GPU + 1 CPU scan)
//   3. for each searching point, sweep the (2R+1)^3 surrounding
//      cells and test the squared-distance / radius predicate      (GPU)
//   The search runs in two passes (count, then write) so the output
//   neighbour lists can be tightly packed without GPU-side allocation.
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <string>

namespace tns
{
namespace internals
{
	// One active "set_i searches set_j" query.
	struct MetalSearchPair {
		int set_i = -1;
		int set_j = -1;
	};

	// Everything the GPU needs, expressed as plain data.
	struct MetalSearchRequest {
		// Geometry. Points of all sets are concatenated (xyzxyz...). A point's
		// global index g belongs to set s where set_offsets[s] <= g < set_offsets[s+1].
		const float* points = nullptr;    // length 3 * total_points
		const float* radii  = nullptr;    // length total_points (per-point radius; fixed-radius callers fill with the global radius)
		int          total_points = 0;
		int          n_sets = 0;
		std::vector<int> set_offsets;      // size n_sets + 1

		// Predicate.
		bool symmetric = true;             // include j if d^2 <= r_i^2 OR (symmetric && d^2 <= r_j^2)

		// Fast path: when the radius is the same for every point AND there is a single
		// point set (the typical SPH fluid case), a specialised kernel is used that
		// reads one float4 per candidate instead of four separate arrays.
		bool  uniform_radius = false;
		float radius = 0.0f;               // the shared radius when uniform_radius is true

		// Background grid.
		float origin[3] = {0, 0, 0};       // min corner used for cell = floor((p-origin)/cell_size)
		int   grid_dims[3] = {1, 1, 1};
		float cell_size = 1.0f;
		int   search_range = 1;            // sweep cells in [-R, R] per axis

		// Queries.
		std::vector<MetalSearchPair> pairs;
	};

	// Result for one pair, laid out exactly as TreeNSearch::solution_ptr expects:
	// block holds, for every point i of set_i, the contiguous run [count, id0, id1, ...]
	// starting at block_offset[i]. Neighbour ids are set_j-local indices.
	struct MetalPairResult {
		int set_i = -1;
		int set_j = -1;
		std::vector<int> block;          // packed [count, ids...] for every searching point
		std::vector<int> block_offset;   // size n_points(set_i) + 1; block_offset[i] = start of point i's run
	};

	struct MetalSearchResult {
		std::vector<MetalPairResult> pairs;  // same order as request.pairs
	};

	// Runs the search on the default Metal device.
	// Returns false (and leaves `out` untouched) if Metal is unavailable or the
	// request is out of supported bounds, so the caller can fall back to the CPU path.
	// `error` receives a human-readable reason on failure.
	bool metal_neighbor_search(const MetalSearchRequest& req,
	                           MetalSearchResult& out,
	                           std::string& error);

	// True if a usable Metal device exists on this machine.
	bool metal_is_available();

	// Invalidate the cached uniform grid. Call once per simulation step (before the
	// first GPU phase) so reuse_grid only reuses a grid built this step.
	void metal_sph_invalidate_grid();

	// ── SPH solver phase 1: density (CubicKernel) on the GPU ─────────────────────
	// Computes, for a single point set with a uniform support radius `h`:
	//   density_i = volume_i*W(0) + sum_j volume_j*W(|x_i-x_j|)   (fluid neighbours)
	// Boundary contributions are added by the caller (kept on CPU for now).
	// Reuses the same uniform-grid build as the neighbour search.
	struct MetalDensityRequest {
		const float* points = nullptr;   // 3*n
		const float* volume = nullptr;   // n  (per-particle volume = mass/density0)
		int   n = 0;
		float h = 0.0f;                  // support radius (CubicKernel radius)
		float origin[3] = {0,0,0};
		int   grid_dims[3] = {1,1,1};
		// Optional Bender2019 boundary contribution, precomputed on the CPU
		// (per fluid particle). Both null -> fluid-only.
		const float* boundary_volume = nullptr;  // n
		const float* boundary_xj     = nullptr;  // 3*n
		const float* velocity        = nullptr;  // 3*n (density-advection only)
		bool reuse_grid = false;     // reuse the grid built by a prior same-step phase (positions unchanged)
	};
	// Writes n densities into out_density. Returns false on failure (caller falls back to CPU).
	bool metal_compute_density(const MetalDensityRequest& req,
	                           float* out_density,
	                           std::string& error);

	// SPH solver phase 2: DFSPH factor alpha_i (fluid part) on the GPU.
	// Uses the same request fields (points/volume/n/h/grid). Writes n factors.
	bool metal_compute_factor(const MetalDensityRequest& req,
	                          float* out_factor,
	                          std::string& error);

	// SPH solver phase 4a: velocity-divergence delta_i = sum_j V_j (v_i-v_j).gradW_ij
	// (+ static-boundary term). Requires req.velocity. Writes n deltas.
	bool metal_compute_density_adv(const MetalDensityRequest& req,
	                               float* out_delta,
	                               std::string& error);

	// SPH solver phase 4b: DFSPH Jacobi pressure-solve iterations on the GPU.
	// Runs `iterations` fixed Jacobi sweeps (accel -> update). densityAdv and factor
	// are inputs; pressure_rho2 is in/out. avg_density_err[k] gets the average error
	// after iteration k (caller decides convergence). Boundary optional.
	struct MetalPressureRequest {
		const float* points = nullptr;   // 3n
		const float* volume = nullptr;   // n
		const float* densityAdv = nullptr; // n  (precomputed)
		const float* factor = nullptr;     // n  (precomputed, already * 1/h^2)
		float* pressure_rho2 = nullptr;    // n  in/out
		int   n = 0;
		float h = 0.0f;                    // support radius
		float dt = 0.0f;                   // timestep
		float density0 = 1000.0f;
		float origin[3] = {0,0,0};
		int   grid_dims[3] = {1,1,1};
		const float* boundary_volume = nullptr; // n
		const float* boundary_xj     = nullptr; // 3n
		int   iterations = 1;          // used when min/max are 0 (fixed-count, for validation)
		int   min_iterations = 0;
		int   max_iterations = 0;
		float eta = 0.0f;              // convergence threshold on avg density error
		bool  divergence = false;      // false: pressure solve (s=1-rhoAdv, aij*=h^2); true: divergence (s=-rhoAdv, aij*=h)
		bool  reuse_grid = false;      // reuse grid built by a prior same-step phase (positions unchanged)
	};
	// Runs the Jacobi solve. If max_iterations>0, loops until (iter>=min_iterations &&
	// avg_err<=eta) or iter==max_iterations; else runs exactly `iterations`. Writes the
	// final pressure accel (3n) into out_accel and the iteration count into out_iterations.
	bool metal_pressure_solve(const MetalPressureRequest& req,
	                          float* out_accel,        // 3n  (may be null)
	                          int*   out_iterations,   // may be null
	                          std::string& error);
}
}
