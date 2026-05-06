#version 430 core

layout(local_size_x = 256) in;

// =========================================================================
// Uniforms
// =========================================================================
uniform uint totalCount; // = numWorkgroups * 256 (for 8-bit radix)

// =========================================================================
// SSBOs
// =========================================================================
layout(std430, binding = 5) buffer HistogramBuffer {
    uint histogram[];
};

// =========================================================================
// Shared Memory
// =========================================================================
shared uint temp[256];

// =========================================================================
// MAIN KERNEL
// Single-workgroup hybrid serial + Blelloch exclusive prefix sum.
// Each of 256 threads handles ceil(totalCount/256) elements serially,
// then we Blelloch-scan the 256 partial sums, then distribute back.
// Scales dynamically to handle millions of particles.
// =========================================================================
void main() {
    uint tid = gl_LocalInvocationID.x;

    // Compute how many elements each thread handles
    uint batch = (totalCount + 255u) / 256u;
    uint start = tid * batch;

    // Phase 1: Serial reduction — each thread sums its chunk
    uint localSum = 0u;
    for (uint i = 0u; i < batch; ++i) {
        uint idx = start + i;
        if (idx < totalCount) {
            localSum += histogram[idx];
        }
    }
    temp[tid] = localSum;
    barrier();

    // Phase 2: Blelloch exclusive prefix sum on temp[0..255]

    // Up-sweep (reduce)
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint idx = (tid + 1u) * (stride << 1u) - 1u;
        if (idx < 256u) {
            temp[idx] += temp[idx - stride];
        }
        barrier();
    }

    // Clear the last element (root of the tree)
    if (tid == 0u) {
        temp[255] = 0u;
    }
    barrier();

    // Down-sweep
    for (uint stride = 128u; stride >= 1u; stride >>= 1u) {
        uint idx = (tid + 1u) * (stride << 1u) - 1u;
        if (idx < 256u) {
            uint t = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += t;
        }
        barrier();
    }

    // Phase 3: Distribute prefix offsets back to the histogram
    // temp[tid] now holds the exclusive prefix sum of all elements
    // before this thread's chunk.
    uint threadOffset = temp[tid];

    uint running = threadOffset;
    for (uint i = 0u; i < batch; ++i) {
        uint idx = start + i;
        if (idx < totalCount) {
            uint val = histogram[idx];
            histogram[idx] = running;
            running += val;
        }
    }
}