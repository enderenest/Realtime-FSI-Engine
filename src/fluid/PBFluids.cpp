#include "fluid/PBFluids.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
// Construction / setup
// ------------------------------------------------------------

PBFluids::PBFluids(const FluidConfig& p)
    : _uboConfig(GL_DYNAMIC_DRAW)
    , _ssboParticles(0)
    , _ssboSolver(0)
    , _ssboHashGrid(0)
    , _ssboOffsets(0)
    , _ssboHashGridAlt(0)
    , _ssboHistogram(0)
    , _ssboLOD(0)
    , _csPredictAndHash(RESOURCES_PATH "shaders/compute/CS_PredictAndHash.glsl")
	, _csRadixHistogram(RESOURCES_PATH "shaders/compute/CS_RadixHistogram.glsl")
	, _csRadixPrefixSum(RESOURCES_PATH "shaders/compute/CS_RadixPrefixSum.glsl")
    , _csRadixScatter(RESOURCES_PATH "shaders/compute/CS_RadixScatter.glsl")
    , _csBuildOffsets(RESOURCES_PATH "shaders/compute/CS_BuildOffsets.glsl")
    , _csComputeLambdas(RESOURCES_PATH "shaders/compute/CS_ComputeLambdas.glsl")
    , _csComputeDeltaP(RESOURCES_PATH "shaders/compute/CS_ComputeDeltaP.glsl")
    , _csApplyDeltaP(RESOURCES_PATH "shaders/compute/CS_ApplyDeltaP.glsl")
    , _csIntegrate(RESOURCES_PATH "shaders/compute/CS_Integrate.glsl")
    , _csVorticity(RESOURCES_PATH "shaders/compute/CS_VorticityConfinement.glsl")
    , _csComputeLOD(RESOURCES_PATH "shaders/compute/CS_ComputeLOD.glsl")
{
    setParams(p);
}

void PBFluids::setParams(const FluidConfig& p)
{
    _params = p;
    setBounds(_params.boundsMin, _params.boundsMax, _params.boundDamping);

    // Precompute wq exactly once
    const F32 dq = _params.deltaQ * _params.h;
    _wq = calcSCorrKernel(make_pvec3(dq, 0.0f, 0.0f), _params.h);

    // Pack and upload UBO
    FluidConfigUBO uboData;
    uboData.boundsMin = { _minBound.x, _minBound.y, _minBound.z, 0.0f };
    uboData.boundsMax = { _maxBound.x, _maxBound.y, _maxBound.z, 0.0f };
    uboData.gravity_dt = { _params.gravity.x, _params.gravity.y, _params.gravity.z, _params.dt / (F32)_params.substepIterations };

    uboData.h = _params.h;
    uboData.rho0 = _params.rho0;
    uboData.eps = _params.eps;
    uboData.wq = _wq;

    uboData.kCorr = _params.kCorr;
    uboData.nCorr = _params.nCorr;
    uboData.viscosity = _params.viscosity;
    uboData.boundDamping = _boundDamping;

    uboData.hashSize = _params.hashSize;
    uboData.particleCount = _params.particleCount;
    uboData.enableSCorr = _params.enableSCorr ? 1u : 0u;
    uboData.enableViscosity = _params.enableViscosity ? 1u : 0u;

    uboData.cohesionStrength = _params.cohesionStrength;
    uboData.interactionRadius = _params.interactionRadius;
    uboData.interactionStrength = _params.interactionStrength;

    // Pre-compute constants that every shader recomputes per-neighbor
    constexpr F32 PI = 3.14159265358979323846f;
    F32 h  = _params.h;
    F32 h2 = h * h;
    F32 h3 = h2 * h;
    F32 h4 = h2 * h2;
    F32 h6 = h3 * h3;
    F32 h9 = h6 * h3;

    // w0_self = Poly6(r=0, h) = (315/(64*PI)) * (h²)³ / h⁹ = (315/(64*PI)) / h³
    uboData.w0_self = (315.0f / (64.0f * PI)) / h3;

    // poly6Coeff = 315 / (64 * PI * h^9)  — multiply by (h²-r²)³ to get W
    uboData.poly6Coeff = 315.0f / (64.0f * PI * h9);

    // spikyCoeff = 45 / (PI * h^6)  — multiply by (h-dist)² to get |∇W|
    uboData.spikyCoeff = 45.0f / (PI * h6);

    // invRho0 = 1 / rho0
    uboData.invRho0 = 1.0f / _params.rho0;

    // hashMask = hashSize - 1  (all hashSizes are powers of 2)
    uboData.hashMask = _params.hashSize - 1;

    // APBF adaptive iteration params
    uboData.minLOD     = _params.enableAPBF ? _params.minLOD : _params.solverIterations;
    uboData.maxLOD     = _params.enableAPBF ? _params.maxLOD : _params.solverIterations;
    uboData.lodMaxDist = _params.lodMaxDist;
    uboData.enableAPBF = _params.enableAPBF ? 1u : 0u;

    _uboConfig.upload(uboData);

    // When APBF is disabled, ensure LOD = solverIterations so all
    // particles are always active (lod[id] >= currentIter for every iter).
    if (!_params.enableAPBF && _ssboLOD.count() > 0)
        initLODBuffer();
}

void PBFluids::setParticles(const std::vector<Particle>& particles)
{
    _particles = particles;
    const size_t N = _particles.size();

    // Update particle count in params and UBO
    _params.particleCount = (U32)N;
    setParams(_params);

    _ssboParticles.upload(_particles);

    // Allocate volatile solver buffers
    std::vector<PVec4> dummySolver(N * 2); // 2 vec4s per particle (predPos_lambda, deltaP_rho)
    _ssboSolver.upload(dummySolver);

    std::vector<UVec2> dummyHash(N); // 1 uvec2 per particle (hash, original_id)
    _ssboHashGrid.upload(dummyHash);

    // Radix sort ping-pong buffer
    _ssboHashGridAlt.upload(dummyHash);

    // CRITICAL FIX: Radix sort histogram (8-bit = 256 bins per workgroup)
    // We use 256 threads per workgroup for the Scatter/Histogram shaders
    _radixSortGroups = ((U32)N + 256 - 1) / 256;

    // Allocate enough space for 256 bins per group
    std::vector<U32> dummyHist(_radixSortGroups * 256, 0u);
    _ssboHistogram.upload(dummyHist);

    // 1 ivec2 per cell (start, end). Initialize both to -1 (empty)
    std::vector<IVec2> dummyOffsets(_params.hashSize, { -1, -1 });
    _ssboOffsets.upload(dummyOffsets);

    // APBF LOD buffer — 1 uint per particle
    initLODBuffer();
}

void PBFluids::readbackParticles(std::vector<Particle>& out)
{
    const size_t n = _ssboParticles.count();
    out.resize(n);
    if (n == 0) return;

    // Make the compute shaders' writes visible to the upcoming client-side map.
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    const Particle* src = _ssboParticles.map(GL_READ_ONLY);
    if (src) {
        std::copy(src, src + n, out.data());
        _ssboParticles.unmap();
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PBFluids::initLODBuffer()
{
    // Default LOD = solverIterations so all particles always active when APBF off.
    // When APBF on, the CS_ComputeLOD shader overwrites this each frame.
    const U32 defaultLOD = _params.enableAPBF ? _params.maxLOD : _params.solverIterations;
    std::vector<U32> lodData(_params.particleCount, defaultLOD);
    _ssboLOD.upload(lodData);
}

// ------------------------------------------------------------
// Bounds
// ------------------------------------------------------------

void PBFluids::setBounds(const PVec3& minBound, const PVec3& maxBound, F32 damping)
{
    const PVec3 pad = make_pvec3(_collisionPadding, _collisionPadding, _collisionPadding);

    _minBound = minBound + pad;
    _maxBound = maxBound - pad;
    _boundDamping = std::clamp(damping, 0.0f, 1.0f);
}

// ------------------------------------------------------------
// Mouse interaction
// ------------------------------------------------------------

void PBFluids::setInteraction(bool active, const PVec3& pos)
{
    _csPredictAndHash.use();
    _csPredictAndHash.setUint("isInteracting", active ? 1u : 0u);
    _csPredictAndHash.setVec3("interactionPos", pos.x, pos.y, pos.z);
}

// ------------------------------------------------------------
// Main step (GPU Compute Pipeline)
// ------------------------------------------------------------

void PBFluids::step()
{
    if (_particles.empty()) return;

    const U32 N = _params.particleCount;

    // Standard compute shader workgroup size
    const U32 workgroupSize = 256;
    const U32 numGroups = (N + workgroupSize - 1) / workgroupSize;
    const U32 gridGroups = (_params.hashSize + workgroupSize - 1) / workgroupSize;

    // Bind all buffers to their respective binding points
    _uboConfig.bindTo(0);

    _ssboParticles.bindTo(0);
    _ssboSolver.bindTo(1);
    _ssboHashGrid.bindTo(2);
    _ssboOffsets.bindTo(3);
    _ssboLOD.bindTo(7);

    // XPBD Substepping Loop
    for (U32 s = 0; s < _params.substepIterations; ++s) {

        // 1. Predict & Hash
        _csPredictAndHash.use();
        _csPredictAndHash.dispatch(numGroups);
        _csPredictAndHash.wait();

        // 2. Radix Sort (Sorting the HashGridBuffer)
        // (8-bit radix, 4 passes for 32-bit keys)
        // Optimized for RTX 4060: 256 workgroup size, efficient dispatch
        {
            const U32 numPasses = 4; // CRITICAL: Even number prevents binding traps
            const U32 bins = 256;    // 8-bit radix uses 256 bins
            const U32 histSize = _radixSortGroups * bins;
            bool sourceIsPrimary = true;

            for (U32 pass = 0; pass < numPasses; ++pass) {
                U32 bitOffset = pass * 8;

                // Ping-pong bindings
                if (sourceIsPrimary) {
                    _ssboHashGrid.bindTo(2);       // Read from Grid
                    _ssboHashGridAlt.bindTo(4);    // Write to Alt
                }
                else {
                    _ssboHashGridAlt.bindTo(2);    // Read from Alt
                    _ssboHashGrid.bindTo(4);       // Write to Grid
                }
                _ssboHistogram.bindTo(5);

                // Pass A: Build per-workgroup histograms
                _csRadixHistogram.use();
                _csRadixHistogram.setUint("bitOffset", bitOffset);
                _csRadixHistogram.setUint("particleCount", N);
                _csRadixHistogram.setUint("numWorkgroups", _radixSortGroups);
                _csRadixHistogram.dispatch(_radixSortGroups);
                _csRadixHistogram.wait();

                // Pass B: Exclusive prefix sum over histogram
                _csRadixPrefixSum.use();
                _csRadixPrefixSum.setUint("totalCount", histSize);
                _csRadixPrefixSum.dispatch(1);
                _csRadixPrefixSum.wait();

                // Pass C: Scatter elements to destination
                _csRadixScatter.use();
                _csRadixScatter.setUint("bitOffset", bitOffset);
                _csRadixScatter.setUint("particleCount", N);
                _csRadixScatter.setUint("numWorkgroups", _radixSortGroups);
                _csRadixScatter.dispatch(_radixSortGroups);
                _csRadixScatter.wait();

                sourceIsPrimary = !sourceIsPrimary;
            }

            // Because we did exactly 4 passes, sourceIsPrimary is true again.
            // The final pass read from Alt (binding 2) and wrote to Grid (binding 4).
            // This means the perfectly sorted data safely landed back in _ssboHashGrid!

            // Lock the sorted buffer into binding 2 for the Constraint Solver
            _ssboHashGrid.bindTo(2);
        }

        // 3. Build Grid Offsets
        _csBuildOffsets.use();

        // Clear offsets first
        _csBuildOffsets.setUint("clearMode", 1u);
        _csBuildOffsets.dispatch(gridGroups);
        _csBuildOffsets.wait();

        // Build offsets
        _csBuildOffsets.setUint("clearMode", 0u);
        _csBuildOffsets.dispatch(numGroups);
        _csBuildOffsets.wait();

        // 4a. APBF: assign per-particle LOD from camera distance
        if (_params.enableAPBF) {
            _csComputeLOD.use();
            _csComputeLOD.setVec3("cameraPos", _cameraPos.x, _cameraPos.y, _cameraPos.z);
            _csComputeLOD.dispatch(numGroups);
            _csComputeLOD.wait();
        }

        // 4b. Constraint Solving
        // iter is 1-based: particle is active if lod[id] >= iter (eq. 9 in APBF paper)
        const U32 maxIter = _params.enableAPBF ? _params.maxLOD : _params.solverIterations;
        for (U32 iter = 1; iter <= maxIter; ++iter) {

            _csComputeLambdas.use();
            _csComputeLambdas.setUint("currentIter", iter);
            _csComputeLambdas.dispatch(numGroups);
            _csComputeLambdas.wait();

            _csComputeDeltaP.use();
            _csComputeDeltaP.setUint("currentIter", iter);
            _csComputeDeltaP.dispatch(numGroups);
            _csComputeDeltaP.wait();

            _csApplyDeltaP.use();
            _csApplyDeltaP.setUint("currentIter", iter);
            _csApplyDeltaP.dispatch(numGroups);
            _csApplyDeltaP.wait();
        }

        // 5. Integrate & Handle Collisions (also computes curl if vorticity enabled)
        _csIntegrate.use();
        _csIntegrate.setUint("enableVorticity", _params.enableVorticity ? 1u : 0u);

        // SDF solid boundary: bind the 3D texture and hand the shader the grid
        // metadata it needs to map world space -> texture coordinates.
        if (_sdf && _sdf->valid()) {
            _sdf->bindForQuery(kSDFTextureUnit);
            const Eigen::Vector3d& o   = _sdf->origin();
            const Eigen::Vector3i& dim = _sdf->dimensions();
            _csIntegrate.setUint("enableSDF", 1u);
            _csIntegrate.setUint("sdfMode", _sdfContainment ? 1u : 0u);
            _csIntegrate.setVec4("sdfOrigin", (F32)o.x(), (F32)o.y(), (F32)o.z(), 0.0f);
            _csIntegrate.setFloat("sdfCellSize", (F32)_sdf->cellSize());
            _csIntegrate.setVec4("sdfResolution", (F32)dim.x(), (F32)dim.y(), (F32)dim.z(), 0.0f);
            _csIntegrate.setFloat("sdfPadding", _sdfPadding);
        } else {
            _csIntegrate.setUint("enableSDF", 0u);
        }

        _csIntegrate.dispatch(numGroups);
        _csIntegrate.wait();
    }

    // 6. Vorticity Confinement — pass 1 only (curl already computed in CS_Integrate)
    if (_params.enableVorticity) {
        _csVorticity.use();
        _csVorticity.setFloat("vorticityEpsilon", _params.vorticityEpsilon);
        _csVorticity.setUint("pass", 1u);
        _csVorticity.dispatch(numGroups);
        _csVorticity.wait();
    }

    // SSBO stays on GPU — vertex shader reads directly from binding 0.
    // No CPU readback needed.
}