#version 430 core

layout(local_size_x = 256) in;

layout(std140, binding = 0) uniform FluidConfig {
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 gravity_dt;
    float h;
    float rho0;
    float eps;
    float wq;
    float kCorr;
    float nCorr;
    float viscosity;
    float boundDamping;
    uint hashSize;
    uint particleCount;
    uint enableSCorr;
    uint enableViscosity;

    float cohesionStrength;
    float interactionRadius;
    float interactionStrength;
    float w0_self;

    uint  hashMask;
    float poly6Coeff;
    float spikyCoeff;
    float invRho0;
    // APBF params
    uint minLOD;
    uint maxLOD;
    float lodMaxDist;
    uint enableAPBF;
} ubo;

// Per-iteration uniform for APBF
uniform uint currentIter;

struct SolverData {
    vec4 predPos_lambda;
    vec4 deltaP_rho;
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 7) buffer LODBuffer {
    uint lod[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    // APBF: particle is active in iteration l only if lod[id] >= l (eq. 9)
    if (ubo.enableAPBF != 0u && lod[id] < currentIter) return;

    vec3 dp = solver[id].deltaP_rho.xyz;

    // Clamp deltaP to half the smoothing radius.
    // No single iteration should move a particle more than this — if it would,
    // it means lambda exploded (isolated particle, near-zero denominator).
    float maxDp = 0.5 * ubo.h;
    float dpLenSq = dot(dp, dp);
    float maxDpSq = maxDp * maxDp;

    if (dpLenSq > maxDpSq) {
        // Only compute sqrt if needed
        float dpLen = sqrt(dpLenSq);
        dp *= (maxDp / dpLen);
    }

    solver[id].predPos_lambda.xyz += dp;
}