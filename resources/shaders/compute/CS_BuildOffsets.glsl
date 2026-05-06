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

uniform uint clearMode;

// =========================================================================
// SSBOs (Bindings 2 and 3)
// =========================================================================
layout(std430, binding = 2) buffer HashGridBuffer {
    uvec2 hashGrid[]; // x: grid hash, y: particle ID
};

layout(std430, binding = 3) buffer CellOffsetBuffer {
    ivec2 offsets[];  // x: startIndex, y: endIndex
};

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint id = gl_GlobalInvocationID.x;

    if (clearMode == 1u) {
        if (id < ubo.hashSize) {
            offsets[id] = ivec2(-1, -1);
        }
    } 
    else {
        if (id < ubo.particleCount) {
            uint myHash = hashGrid[id].x;
            
            // Short-circuit evaluation safely handles the id == 0 boundary
            if (id == 0 || hashGrid[id - 1].x != myHash) {
                offsets[myHash].x = int(id);
            }
            
            // End index is written exclusively (id + 1) for standard C++ style loops
            if (id == ubo.particleCount - 1 || hashGrid[id + 1].x != myHash) {
                offsets[myHash].y = int(id + 1);
            }
        }
    }
}