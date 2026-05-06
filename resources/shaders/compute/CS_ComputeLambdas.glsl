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

// Per-iteration uniform for APBF
uniform uint currentIter;

// =========================================================================
// SSBOs (Bindings 1, 2, 3, 7)
// =========================================================================
struct SolverData {
    vec4 predPos_lambda; // xyz: predPos, w: lambda
    vec4 deltaP_rho;     // xyz: deltaP, w: rho
};

layout(std430, binding = 1) buffer SolverBuffer {
    SolverData solver[];
};

layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[]; // x: grid hash, y: particle ID
};

layout(std430, binding = 3) buffer CellOffsetBuffer {
    ivec2 offsets[];  // x: startIndex, y: endIndex
};

layout(std430, binding = 7) buffer LODBuffer {
    uint lod[];
};

// =========================================================================
// KERNEL FUNCTIONS — use pre-computed coefficients from UBO
// =========================================================================
float calcPoly6Kernel(vec3 r) {
    float r2 = dot(r, r);
    float h2 = ubo.h * ubo.h;
    if (r2 > h2) return 0.0;

    float diff = h2 - r2;
    return ubo.poly6Coeff * diff * diff * diff;
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
// HASH FUNCTION — bitwise AND instead of integer modulo
// =========================================================================
uint getGridHash(ivec3 cell) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = (cell.x * p1) ^ (cell.y * p2) ^ (cell.z * p3);
    return uint(n) & ubo.hashMask;
}

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= ubo.particleCount) return;

    // APBF: particle is active in iteration l only if lod[id] >= l (eq. 9)
    if (ubo.enableAPBF != 0u && lod[id] < currentIter) return;

    vec3 xi = solver[id].predPos_lambda.xyz;

    float rho_i = ubo.w0_self; // Self-contribution (pre-computed)
    float sum_grad_Ci_sq = 0.0;
    vec3 grad_Ci_i = vec3(0.0);

    // =========================================================================
    // NEIGHBOR SEARCH & ACCUMULATION
    // =========================================================================
    ivec3 cellCoord = ivec3(floor(xi / ubo.h));

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                uint neighborHash = getGridHash(cellCoord + ivec3(x, y, z));
                ivec2 startEnd = offsets[neighborHash];

                if (startEnd.x == -1) continue;

                for (int k = startEnd.x; k < startEnd.y; k++) {
                    uint j = hashGrid[k].y;
                    if (id == j) continue;

                    vec3 rij = xi - solver[j].predPos_lambda.xyz;

                    rho_i += calcPoly6Kernel(rij);

                    vec3 gradW = calcGradSpikyKernel(rij);
                    float gj2 = dot(gradW, gradW);
                    sum_grad_Ci_sq += gj2 * (ubo.invRho0 * ubo.invRho0);

                    grad_Ci_i += gradW;
                }
            }
        }
    }

    // =========================================================================
    // FINALIZE LAMBDA & RHO
    // =========================================================================
    grad_Ci_i *= ubo.invRho0;
    float C_i = max((rho_i / ubo.rho0) - 1.0, -ubo.cohesionStrength);
    sum_grad_Ci_sq += dot(grad_Ci_i, grad_Ci_i);
    float lambda_i = -C_i / (sum_grad_Ci_sq + ubo.eps);

    solver[id].predPos_lambda.w = lambda_i;
    solver[id].deltaP_rho.w = rho_i;
}
