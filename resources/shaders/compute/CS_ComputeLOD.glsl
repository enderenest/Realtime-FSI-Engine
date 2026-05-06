#version 430 core

layout(local_size_x = 256) in;

// =========================================================================
// UBO: Global Fluid Configuration (Binding 0)
// =========================================================================
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

// =========================================================================
// SSBOs
// =========================================================================
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

// Per-frame: world-space camera position
uniform vec3 cameraPos;

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    vec3 pos = solver[id].predPos_lambda.xyz;
    float dist = length(cameraPos - pos);

    // t = 0 → at camera → maxLOD (most iterations)
    // t = 1 → far away  → minLOD (fewest iterations)
    float t = clamp(dist / ubo.lodMaxDist, 0.0, 1.0);

    uint lodVal = uint(round(mix(float(ubo.maxLOD), float(ubo.minLOD), t)));
    lod[id] = clamp(lodVal, ubo.minLOD, ubo.maxLOD);
}
