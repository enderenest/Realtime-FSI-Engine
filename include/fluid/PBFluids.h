#ifndef PBFLUIDS_H
#define PBFLUIDS_H

#include "core/config.h"
#include "core/Types.h"
#include "fluid/Particle.h"
#include "fluid/NeighborSearch.h"
#include "fluid/Kernels.h"   
#include "opengl/SSBO.h"
#include "opengl/UBO.h"
#include "opengl/ComputeShader.h"

#include <vector>
#include <memory>


// ============================================================
// UBO Memory Layout (std140 padded)
// ============================================================
struct FluidConfigUBO {
    // 16 byte Vec4 parameters
    PVec4 boundsMin;        // xyz: min bound, w: padding
    PVec4 boundsMax;        // xyz: max bound, w: padding
    PVec4 gravity_dt;       // xyz: gravity, w: subDt

    // 16 byte Float parameters
    F32 h;
    F32 rho0;
    F32 eps;
    F32 wq;

    // 16 byte Float parameters
    F32 kCorr;
    F32 nCorr;
    F32 viscosity;
    F32 boundDamping;

    // 16 byte UInt parameters
    U32 hashSize;
    U32 particleCount;
    U32 enableSCorr;
    U32 enableViscosity;

    F32 cohesionStrength;
    F32 interactionRadius;
    F32 interactionStrength;
    F32 w0_self;  // Pre-computed: calcPoly6Kernel(vec3(0.0), h)

    // Pre-computed kernel coefficients (avoid per-neighbor recomputation)
    U32 hashMask;      // hashSize - 1, for bitwise AND instead of modulo
    F32 poly6Coeff;    // 315.0 / (64.0 * PI * h^9)
    F32 spikyCoeff;    // 45.0 / (PI * h^6)
    F32 invRho0;       // 1.0 / rho0

    // APBF: adaptive solver iteration params
    U32 minLOD;
    U32 maxLOD;
    F32 lodMaxDist;
    U32 enableAPBF;
};


// ============================================================
// PBFluids
// ------------------------------------------------------------
// CPU prototype of Position-Based Fluids (PBF)
// Step structure follows the standard PBF loop:
//
// 1) for all i: v_i += dt * fext;  x*_i = x_i + dt * v_i
// 2) for all i: find neighbors N_i(x*) -> it will be implemented in NeighborSearch.h / .cpp
// 3) repeat solverIterations:
//      a) for all i: compute lambda_i
//      b) for all i: compute deltaP_i
//      c) collisions (on x*)
//      d) for all i: x*_i += deltaP_i
// 4) for all i: v_i = (x*_i - x_i)/dt;  x_i = x*_i
// ============================================================

class PBFluids {
public:
    // ----------------------------
    // Construction
    // ----------------------------
    PBFluids() = default;
    explicit PBFluids(const FluidConfig& p);

    void setParams(const FluidConfig& p);
    const FluidConfig& params() const { return _params; }

    // Must be called each frame before step() for APBF LOD computation
    void setCameraPos(const PVec3& pos) { _cameraPos = pos; }

    // Particles
    void setParticles(const std::vector<Particle>& particles);
    std::vector<Particle>& particles() { return _particles; }
    const std::vector<Particle>& particles() const { return _particles; }

	// visual radius for collisions (early CPU prototype, Akinci boundary problem exists here)
    void setCollisionPadding(F32 p) { _collisionPadding = p; }
    F32 _collisionPadding = 0.0f;

    // simple AABB bounds for collisions (early CPU prototype, Akinci boundary problem exists here)
    void setBounds(const PVec3& minBound, const PVec3& maxBound, F32 damping = 0.0f);

    // ----------------------------
    // Mouse interaction
    // ----------------------------
    void setInteraction(bool active, const PVec3& pos);

    // ----------------------------
    // Simulation step
    // ----------------------------
    void step();

    // Bind particle SSBO to binding 0 for rendering
    void bindParticlesForRendering() const { _ssboParticles.bindTo(0); }

private:
    // apply forces + predict x*
    void applyForcesAndPredict();

	// the neighbor search will be implemented in NeighborSearch.h / .cpp

    // solver iterations
    void solverIterationLoop();
    void computeLambdas();  
    void computeDeltaP();    
    void handleCollisions(); 
    void applyDeltaP();     

    // update v and commit positions
    void updateVelocityFromPred();
	void applyViscosityXSPH();
    void commitPositions();

    // ----------------------------
    // Math Helpers
    // ----------------------------
    F32  computeDensity(I32 i) const;
    F32  computeSCorr(const PVec3& dpos) const; // returns 0 if disabled

private:
	// Simulation parameters
    FluidConfig _params;

    // Particle storage
    std::vector<Particle> _particles;

    // Per-step / per-iteration buffers
    std::vector<F32>   _lambda;     // size = N
    std::vector<PVec3> _deltaP;     // size = N

	// Neighbor search structure
    NeighborSearch _neighborSearch{ /*cellSize*/ 0.08f, /*hashSize*/ 1u << 16 };

    // AABB collision
    PVec3 _minBound{ 0,0,0 };   // _minBound = (xmin, ymin, zmin)
    PVec3 _maxBound{ 0,0,0 };   // _maxBound = (xmax, ymax, zmax)
    F32 _boundDamping = 0.0f;

    // Solution to density problem on boundaries
    std::vector<PVec3> getGhostRelativeVectors(const PVec3& pos) const;

    // Derived precomputed constants
    F32 _wq;

    // ----------------------------
    // GPU MEMORY BUFFERS
    // ----------------------------
    UBO<FluidConfigUBO> _uboConfig;

    SSBO<Particle> _ssboParticles;      // Binding 0
    SSBO<PVec4> _ssboSolver;            // Binding 1 (PredPos_Lambda, DeltaP_Rho)
    SSBO<UVec2> _ssboHashGrid;          // Binding 2 (Hash pairs - PRIMARY)
    SSBO<IVec2> _ssboOffsets;           // Binding 3 (Cell start/end indices)

    // Radix Sort Buffers
    SSBO<UVec2> _ssboHashGridAlt; // Binding 4 (Hash pairs - PING-PONG)
    SSBO<U32> _ssboHistogram;   // Binding 5 (Radix block counts / prefix sums)
    U32 _radixSortGroups; // Tracks the number of workgroups for the histogram

    // APBF: per-particle LOD (solver iterations assigned by camera distance)
    SSBO<U32> _ssboLOD;        // Binding 7
    PVec3 _cameraPos{ 0, 0, 0 };
    void initLODBuffer();

    // ----------------------------
    // COMPUTE SHADER PIPELINE
    // ----------------------------
    ComputeShader _csPredictAndHash;
	ComputeShader _csRadixHistogram;
    ComputeShader _csRadixPrefixSum;
    ComputeShader _csRadixScatter;
    ComputeShader _csBuildOffsets;
    ComputeShader _csComputeLambdas;
    ComputeShader _csComputeDeltaP;
    ComputeShader _csApplyDeltaP;
    ComputeShader _csIntegrate;
    ComputeShader _csVorticity;
    ComputeShader _csComputeLOD;
};

#endif