#ifndef CONFIG_H
#define CONFIG_H

#include "core/Types.h"

// ---------------------------
// Fluid config
// ---------------------------
struct FluidConfig {
	// time step
	F32 dt = 1.0f / 60.0f;						// macro timestep: 60 FPS
	PVec3 gravity = { 0.0f, -3.81f, 0.0f };

	// PBF / SPH kernels
	F32 h = 0.15f;								// smoothing radius: h/spacing = 2.0 (minimum stable PBF ratio)
	F32 rho0 = 1200.0f;						// rest density for m=1, h=0.2, spacing=0.1
	U32 solverIterations = 3;					// increased from 2: cohesion needs more iterations for stability
	U32 substepIterations = 2;					// increased: smaller effective dt reduces ceiling sticking from cohesion overshoot
	F32 eps = 0.0001f;							// increased from 0.00001: cohesion needs stronger regularization

	// Spawn configuration — 65K particles, dam-break block
	U32  particleCount = 1 << 15;				// 65536 particles
	PVec3 spawnMin = { 2.f, 0.2f, 2.f };		// lower corner of fluid block
	PVec3 spawnMax = { 5.f, 5.f, 5.f };		// ~41^3 = 68921 grid slots > 65536
	bool spawnRandom = false;					// grid spawn: uniform spacing, no overlaps
	F32  spacing = 0.1f;						// particle spacing = h/2 (stable packing)
	PVec3 initialVelocity = { 0.0f, 0.0f, 0.0f };

	// Boundary handling
	PVec3 boundsMin = { 0.0f, 0.0f, 0.0f };
	PVec3 boundsMax = { 6.0f, 6.0f, 6.0f };
	F32 boundDamping = 0.5f;					// reduced: less energy retained at walls, particles fall away from ceiling

	// Neighbor search hash grid
	U32 hashSize = 1 << 17;					// 262144 cells: 4:1 ratio reduces hash collisions

	// Surface tension correction (sCorr)
	bool enableSCorr = true;					// suppress particle clustering at surface
	F32 kCorr = 0.001f;						// tensile correction strength
	F32 nCorr = 4.0f;							// restored to 4.0 for surface tension stability with cohesion
	F32 deltaQ = 0.3f;							// reference distance ratio (0.3 * h)

	// Cohesion (soft constraint on low-density regions for droplet formation)
	F32 cohesionStrength = 0.001f;				// reduced: weaker cohesion so gravity wins over wall attraction

	// XSPH viscosity
	bool enableViscosity = true;				// energy dissipation for stability
	F32 viscosity = 0.1f;						// light viscosity: damps jitter without killing flow

	// Mouse interaction
	F32 interactionRadius = 1.5f;				// world-space radius of influence
	F32 interactionStrength = 25.0f;				// force magnitude (negative = push, positive = pull)

	// Vorticity confinement
	bool enableVorticity = true;
	F32 vorticityEpsilon = 0.05f;				// re-injects rotational energy lost to damping

	// APBF: Adaptive solver iterations (Köster & Krüger, 2016)
	bool enableAPBF = false;
	U32  minLOD = 2;          // solver iters for far/interior particles
	U32  maxLOD = 7;          // solver iters for near/surface particles
	F32  lodMaxDist = 15.0f;  // camera distance at which LOD clamps to minLOD
};

// ---------------------------
// Deformation config (later)
// ---------------------------
struct DeformConfig {
	bool enabled = false;
	F32 stiffness = 1.0f;  // placeholder
	// add Laplacian/constraints params later
};

// ---------------------------
// Viewer config
// ---------------------------
struct ViewerConfig {
	bool showParticles = true;
	F32 pointRadius = 0.02f;
};

// ---------------------------
// Global config: combines all sub-configs
// ---------------------------
struct Config {
	FluidConfig fluid;
	DeformConfig deform;
	ViewerConfig viewer;
};

#endif