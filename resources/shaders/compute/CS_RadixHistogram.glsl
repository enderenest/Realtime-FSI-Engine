#version 430 core

layout(local_size_x = 256) in;

// =========================================================================
// Uniforms
// =========================================================================
uniform uint bitOffset;
uniform uint particleCount;
uniform uint numWorkgroups;

// =========================================================================
// SSBOs
// =========================================================================
layout(std430, binding = 2) buffer SourceBuffer {
    uvec2 source[];
};

layout(std430, binding = 5) buffer HistogramBuffer {
    uint histogram[];
};

// =========================================================================
// Shared Memory
// =========================================================================
shared uint localHist[256];

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint lid  = gl_LocalInvocationID.x;
    uint gid  = gl_GlobalInvocationID.x;
    uint wgid = gl_WorkGroupID.x;

    // Clear local histogram
    if (lid < 256u) {
        localHist[lid] = 0u;
    }
    barrier();

    // Count digits for this workgroup's tile
    if (gid < particleCount) {
        uint key   = source[gid].x;
        uint digit = (key >> bitOffset) & 0xFFu;  // Extract 8 bits for 256 buckets
        atomicAdd(localHist[digit], 1u);
    }
    barrier();

    // Write to global histogram in column-major layout:
    // histogram[bucket * numWorkgroups + workgroupID]
    // This ensures the prefix sum produces correct global scatter offsets.
    if (lid < 256u) {
        histogram[lid * numWorkgroups + wgid] = localHist[lid];
    }
}
