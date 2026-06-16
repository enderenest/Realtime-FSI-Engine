#version 430 core
layout(local_size_x = 256) in;

// =========================================================================
// UBO: Global Fluid Configuration (Binding 0) — matches FluidConfigUBO exactly
// =========================================================================
layout(std140, binding = 0) uniform FluidConfig {
    vec4  boundsMin;
    vec4  boundsMax;
    vec4  gravity_dt;
    float h; float rho0; float eps; float wq;
    float kCorr; float nCorr; float viscosity; float boundDamping;
    uint  hashSize; uint particleCount; uint enableSCorr; uint enableViscosity;
    float cohesionStrength; float interactionRadius; float interactionStrength; float w0_self;
    uint  hashMask; float poly6Coeff; float spikyCoeff; float invRho0;
    uint  minLOD; uint maxLOD; float lodMaxDist; uint enableAPBF;
} ubo;

// =========================================================================
// Particle SSBO (Binding 0) — read-only, same layout as every other shader
// =========================================================================
struct Particle {
    vec4 pos;
    vec4 predPos;
    vec4 vel;
};

layout(std430, binding = 0) readonly buffer ParticleBuffer {
    Particle particles[];
};

// =========================================================================
// Output: compact near-surface candidate list
//   Binding 6 — single uint used as atomic counter (reset to 0 each frame)
//   Binding 8 — flat array of {pos, vel} pairs for candidates
// =========================================================================
layout(std430, binding = 6) buffer ContactCountBuffer {
    uint contactCount;
};

struct ContactCandidate {
    vec4 pos; // xyz: world position, w: unused
    vec4 vel; // xyz: velocity,       w: unused
};

layout(std430, binding = 8) writeonly buffer ContactCandidateBuffer {
    ContactCandidate candidates[];
};

// =========================================================================
// SDF (same uniforms as CS_Integrate)
// =========================================================================
uniform vec4  sdfOrigin;
uniform float sdfCellSize;
uniform vec4  sdfResolution;
uniform float contactBand;

layout(binding = 1) uniform sampler3D sdfTexture;

float querySDF(vec3 worldPos) {
    vec3 tc = ((worldPos - sdfOrigin.xyz) / sdfCellSize + 0.5) / sdfResolution.xyz;
    return texture(sdfTexture, tc).r;
}

// =========================================================================
// Main
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    vec3  pos  = particles[id].pos.xyz;
    float dist = querySDF(pos);

    if (abs(dist) < contactBand) {
        uint slot = atomicAdd(contactCount, 1u);
        // Guard: slot must stay within the allocated candidate buffer (N elements max)
        if (slot < ubo.particleCount) {
            candidates[slot].pos = vec4(pos, 0.0);
            candidates[slot].vel = particles[id].vel;
        }
    }
}
