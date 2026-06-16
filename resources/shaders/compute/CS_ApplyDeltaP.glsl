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

// --- SDF boundary (same layout/binding as CS_Integrate) ---
// Projecting predicted positions onto the boundary inside the solver loop lets the
// density constraint and the boundary constraint converge together, instead of the
// solver pushing particles through the wall and Integrate snapping them back.
uniform uint      enableSDF;
uniform uint      sdfMode;     // 0: obstacle (fluid outside), 1: container (fluid inside)
layout(binding = 1) uniform sampler3D sdfTexture;
uniform vec4      sdfOrigin;
uniform float     sdfCellSize;
uniform vec4      sdfResolution;
uniform float     sdfPadding;

float querySDF(vec3 worldPos) {
    vec3 tc = ((worldPos - sdfOrigin.xyz) / sdfCellSize + 0.5) / sdfResolution.xyz;
    return texture(sdfTexture, tc).r;
}

vec3 sdfGradient(vec3 worldPos) {
    float e = sdfCellSize;
    return vec3(
        querySDF(worldPos + vec3(e, 0.0, 0.0)) - querySDF(worldPos - vec3(e, 0.0, 0.0)),
        querySDF(worldPos + vec3(0.0, e, 0.0)) - querySDF(worldPos - vec3(0.0, e, 0.0)),
        querySDF(worldPos + vec3(0.0, 0.0, e)) - querySDF(worldPos - vec3(0.0, 0.0, e))
    ) / (2.0 * e);
}

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

    vec3 p = solver[id].predPos_lambda.xyz + dp;

    // Project back onto the SDF boundary so this iteration's result already respects
    // the wall. Without this the density solve repeatedly pushes boundary particles
    // through the surface and only Integrate corrects them, producing a frame-rate
    // snap/push cycle that never settles.
    if (enableSDF == 1u) {
        float d   = querySDF(p);
        vec3  g   = sdfGradient(p);
        float len = length(g);
        if (len > 1e-6) {
            if (sdfMode == 1u) {
                // container: keep at least sdfPadding inside the inner wall
                if (d > -sdfPadding) p += (-g / len) * (d + sdfPadding);
            } else {
                // obstacle: keep at least sdfPadding outside the surface
                if (d < sdfPadding) p += (g / len) * (sdfPadding - d);
            }
        }
    }

    solver[id].predPos_lambda.xyz = p;
}