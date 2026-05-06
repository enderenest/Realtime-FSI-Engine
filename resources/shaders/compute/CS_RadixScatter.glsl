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

layout(std430, binding = 4) buffer DestBuffer {
    uvec2 dest[];
};

layout(std430, binding = 5) buffer HistogramBuffer {
    uint histogram[]; // prefix-summed global base offsets
};

// =========================================================================
// Shared Memory
//
// Bit-flag ballot layout:
//   flags[digit * 8 + word]  =  32-bit mask of which thread-lids have
//   that digit, where word = lid / 32, bit = lid % 32.
//
// 256 digits * 8 words = 2048 uint32s = 8 KB.
// Rank for thread lid with digit d = popcount of all bits set in
// flags[d*8 .. d*8 + lid/32] that precede bit lid%32.
// This is O(8) per thread, fully stable (lower lid → lower rank), and
// uses only hardware bitCount instructions.
// =========================================================================
shared uint flags[2048];

void main() {
    uint lid  = gl_LocalInvocationID.x;
    uint gid  = gl_GlobalInvocationID.x;
    uint wgid = gl_WorkGroupID.x;

    // 1. Extract digit
    uint myDigit = 256u; // sentinel for out-of-bounds threads
    if (gid < particleCount) {
        myDigit = (source[gid].x >> bitOffset) & 0xFFu;
    }

    // 2. Clear flag array: 256 threads * 8 entries each = 2048 total
    for (uint i = 0u; i < 8u; i++) {
        flags[lid * 8u + i] = 0u;
    }
    barrier();

    // 3. Each in-bounds thread atomically sets its bit in its digit's row
    if (gid < particleCount) {
        atomicOr(flags[myDigit * 8u + (lid >> 5u)], 1u << (lid & 31u));
    }
    barrier();

    // 4. Compute stable intra-workgroup rank using popcount
    //    rank = number of threads with lid < mine that share myDigit
    if (gid < particleCount) {
        uint rank    = 0u;
        uint base    = myDigit * 8u;
        uint myWord  = lid >> 5u;   // lid / 32
        uint myBit   = lid & 31u;   // lid % 32

        // Full 32-thread words before mine
        for (uint w = 0u; w < myWord; w++) {
            rank += bitCount(flags[base + w]);
        }
        // Partial word: only bits strictly below myBit
        rank += bitCount(flags[base + myWord] & ((1u << myBit) - 1u));

        // 5. Scatter to destination
        uint globalOffset = histogram[myDigit * numWorkgroups + wgid];
        dest[globalOffset + rank] = source[gid];
    }
}
