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
    , _ssboContactCount(0)
    , _ssboContactCandidates(0)
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
    , _csContactFilter(RESOURCES_PATH "shaders/compute/CS_ContactFilter.glsl")
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

    // Contact filter buffers: count (1 uint, cleared) + candidate list (N slots worst-case)
    std::vector<U32> zeroCount(1, 0u);
    _ssboContactCount.upload(zeroCount);
    std::vector<ContactCandidate> emptyCandidates(N);
    _ssboContactCandidates.upload(emptyCandidates);
}

void PBFluids::readbackParticles(std::vector<Particle>& out)
{
    const size_t n = _ssboParticles.count();
    out.resize(n);
    if (n == 0) return;

    // Non-stalling readback. Called at the TOP of the frame, before the next
    // step(), so the only GPU work that has written the particle buffer is the
    // PREVIOUS frame's step. We wait on that step's fence; a full frame of
    // render/swap has elapsed, so it is already signalled and the wait returns
    // immediately instead of draining the in-flight pipeline. With completion
    // guaranteed we map UNSYNCHRONIZED, which also skips waiting on last frame's
    // particle draw (a read — no hazard with our read). If the fence somehow has
    // not signalled yet, we fall back to a synchronized map so data stays valid.
    bool writesDone = false;
    if (_readbackFence) {
        const GLenum r = glClientWaitSync(_readbackFence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
        writesDone = (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
        glDeleteSync(_readbackFence);
        _readbackFence = nullptr;
    }
    if (!writesDone) {
        // First readback (no fence yet) or rare not-ready case: make the writes
        // visible and let the synchronized map below block as needed.
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }

    const GLbitfield flags = GL_MAP_READ_BIT | (writesDone ? GL_MAP_UNSYNCHRONIZED_BIT : 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboParticles.getID());
    const Particle* src = static_cast<const Particle*>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                         static_cast<GLsizeiptr>(n * sizeof(Particle)), flags));
    if (src) {
        std::copy(src, src + n, out.data());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PBFluids::filterContactCandidates(float contactBand)
{
    if (!_sdf || !_sdf->valid() || contactBand <= 0.0f) return;
    if (_ssboContactCandidates.count() == 0) return; // setParticles not called yet

    const U32 N = _params.particleCount;

    // Reset the atomic counter to 0.  A GL_BUFFER_UPDATE_BARRIER_BIT barrier
    // makes this write visible to the subsequent compute dispatch.
    U32 zero = 0u;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboContactCount.getID());
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(U32), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Bind all resources needed by the filter shader
    _uboConfig.bindTo(0);
    _ssboParticles.bindTo(0);
    _ssboContactCount.bindTo(6);
    _ssboContactCandidates.bindTo(8);

    _sdf->bindForQuery(kSDFTextureUnit);
    const Eigen::Vector3d& o   = _sdf->origin();
    const Eigen::Vector3i& dim = _sdf->dimensions();

    _csContactFilter.use();
    _csContactFilter.setVec4("sdfOrigin",     (F32)o.x(), (F32)o.y(), (F32)o.z(), 0.0f);
    _csContactFilter.setFloat("sdfCellSize",  (F32)_sdf->cellSize());
    _csContactFilter.setVec4("sdfResolution", (F32)dim.x(), (F32)dim.y(), (F32)dim.z(), 0.0f);
    _csContactFilter.setFloat("contactBand",  contactBand);

    _csContactFilter.dispatch((N + 255u) / 256u);
    // No wait() here: GL_SHADER_STORAGE_BARRIER_BIT orders GPU→GPU accesses on
    // the same buffer, but nothing in this frame reads bindings 6 or 8 after the
    // filter.  The fence below uses GL_SYNC_GPU_COMMANDS_COMPLETE, which signals
    // only after ALL preceding GPU commands have finished and their effects are
    // "fully realized" — that covers the candidate writes for the CPU map.

    // Supersede any previous readback fence with one that also covers this filter
    // dispatch.  readbackContactCandidates() (called at the TOP of the next frame)
    // waits on it — by then the filter is already done, so the wait returns
    // immediately and the map can be UNSYNCHRONIZED.
    if (_readbackFence) glDeleteSync(_readbackFence);
    _readbackFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void PBFluids::readbackContactCandidates(std::vector<ContactCandidate>& out)
{
    out.clear();
    if (_ssboContactCandidates.count() == 0) return;

    // Same deferred fence pattern as readbackParticles: called at the TOP of the
    // frame before step(), so the fence from last frame's filterContactCandidates()
    // is already signalled and the maps are UNSYNCHRONIZED (no CPU stall).
    bool writesDone = false;
    if (_readbackFence) {
        const GLenum r = glClientWaitSync(_readbackFence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
        writesDone = (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
        glDeleteSync(_readbackFence);
        _readbackFence = nullptr;
    }
    // No glMemoryBarrier in the fallback path: GL_SHADER_STORAGE_BARRIER_BIT and
    // GL_BUFFER_UPDATE_BARRIER_BIT order GPU→GPU commands inside the GL stream —
    // they do not affect CPU visibility.  A synchronized map (no
    // GL_MAP_UNSYNCHRONIZED_BIT) already stalls the CPU until all pending GPU
    // operations on the buffer complete and the data is coherent.

    const GLbitfield flags = GL_MAP_READ_BIT | (writesDone ? GL_MAP_UNSYNCHRONIZED_BIT : 0);

    // Read the candidate count (1 uint)
    U32 count = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboContactCount.getID());
    const U32* pCount = static_cast<const U32*>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(U32), flags));
    if (pCount) { count = *pCount; glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); }

    // Clamp to the allocated buffer size — the shader guards this too, but belt + suspenders.
    count = std::min(count, (U32)_ssboContactCandidates.count());

    if (count == 0) { glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); return; }

    // Read the compact candidate list (positions + velocities only, no full Particle)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboContactCandidates.getID());
    const ContactCandidate* src = static_cast<const ContactCandidate*>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                         static_cast<GLsizeiptr>(count * sizeof(ContactCandidate)), flags));
    if (src) {
        out.assign(src, src + count);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
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
    // No CPU readback needed for rendering.

    // Fallback fence: covers this step's GPU work.  When filterContactCandidates()
    // is called right after (the normal case), it supersedes this fence with one
    // that also covers the filter dispatch — so readbackContactCandidates() waits
    // on the later fence.  When the filter is not dispatched (paused sim), this
    // fence remains and readbackContactCandidates() uses it as-is.
    if (_readbackFence) glDeleteSync(_readbackFence);
    _readbackFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}