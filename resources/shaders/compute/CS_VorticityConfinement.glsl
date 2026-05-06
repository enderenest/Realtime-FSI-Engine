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
} ubo;

// =========================================================================
// SSBOs (Binding 0, 1, 2, 3)
// =========================================================================
struct Particle {
    vec4 pos;
    vec4 vel;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

struct SolverData {
    vec4 predPos_lambda;
    vec4 deltaP_rho;     // .xyz reused for curl storage between passes
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[];
};

layout(std430, binding = 3) buffer CellOffsetBuffer {
    ivec2 offsets[];
};

// =========================================================================
// Uniforms
// =========================================================================
uniform uint  pass;             // 0 = compute curl, 1 = apply vorticity force
uniform float vorticityEpsilon;

// =========================================================================
// Hash + Kernel — pre-computed coefficients
// =========================================================================
uint getGridHash(ivec3 cell) {
    const int p1 = 73856093, p2 = 19349663, p3 = 83492791;
    int n = (cell.x * p1) ^ (cell.y * p2) ^ (cell.z * p3);
    return uint(n) & ubo.hashMask;
}

vec3 calcGradSpikyKernel(vec3 r) {
    float r2 = dot(r, r);
    float h2 = ubo.h * ubo.h;
    if (r2 > h2 || r2 < 1e-10) return vec3(0.0);

    float dist = sqrt(r2);
    float diff = ubo.h - dist;
    float scalar = -ubo.spikyCoeff * diff * diff / dist;
    return r * scalar;
}

// =========================================================================
// MAIN
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    vec3 pos_i = particles[id].pos.xyz;
    vec3 vel_i = particles[id].vel.xyz;

    ivec3 cellCoord = ivec3(floor(pos_i / ubo.h));

    if (pass == 0u) {
        // =====================================================================
        // PASS 0: Compute curl  w_i = (1/rho0) * sum_j (v_j - v_i) x gradW_ij
        // =====================================================================
        vec3 curl = vec3(0.0);

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint nh = getGridHash(cellCoord + ivec3(dx, dy, dz));
            ivec2 se = offsets[nh];
            if (se.x == -1) continue;
            for (int k = se.x; k < se.y; k++) {
                uint j = hashGrid[k].y;
                if (id == j) continue;

                vec3 rij   = pos_i - particles[j].pos.xyz;
                vec3 gradW = calcGradSpikyKernel(rij);
                vec3 vij   = particles[j].vel.xyz - vel_i;

                curl += cross(vij, gradW);
            }
        }

        solver[id].deltaP_rho.xyz = curl * ubo.invRho0;
    }
    else {
        // =====================================================================
        // PASS 1: Compute eta = (1/rho0) * grad|w|, then apply vorticity force
        // =====================================================================
        vec3 eta = vec3(0.0);

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint nh = getGridHash(cellCoord + ivec3(dx, dy, dz));
            ivec2 se = offsets[nh];
            if (se.x == -1) continue;
            for (int k = se.x; k < se.y; k++) {
                uint j = hashGrid[k].y;
                if (id == j) continue;

                vec3 rij      = pos_i - particles[j].pos.xyz;
                vec3 gradW    = calcGradSpikyKernel(rij);
                float omegaJ  = length(solver[j].deltaP_rho.xyz);

                eta += omegaJ * gradW;
            }
        }

        eta *= ubo.invRho0;

        float etaLen = length(eta);
        if (etaLen < 1e-6) return;

        vec3  N     = eta / etaLen;
        vec3  omega = solver[id].deltaP_rho.xyz;
        float dt    = ubo.gravity_dt.w;

        particles[id].vel.xyz += vorticityEpsilon * cross(N, omega) * dt;
    }
}
