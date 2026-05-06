#version 430 core

// Match the workgroup size defined in the C++ step() function
layout(local_size_x = 256) in;

// =========================================================================
// Uniforms
// =========================================================================
uniform uint particleCount; // Total number of particles (formerly u_N)
uniform uint k;             // The current block size being sorted (formerly u_size)
uniform uint j;             // The stride distance for comparisons (formerly u_stride)

// =========================================================================
// SSBOs (Binding 2: Hash Grid Buffer)
// =========================================================================
layout(std430, binding = 2) buffer HashGridBuffer {
    // x: cell hash (the key we are sorting by)
    // y: particle ID (the payload data)
    uvec2 hashGrid[]; 
};

// =========================================================================
// MAIN KERNEL
// =========================================================================
void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    // Find the partner thread to compare against using bitwise XOR
    uint partner = idx ^ j;

    // Safety check: ensure both indices are within bounds, and only process 
    // the pair once (where idx is the lower index) to prevent race conditions.
    if (idx < particleCount && partner < particleCount && idx < partner) {
        
        // Determine if this specific block should be sorted ascending or descending
        bool ascending = ((idx & k) == 0u);
        
        // Read the pairs from memory
        uvec2 a = hashGrid[idx];
        uvec2 b = hashGrid[partner];
        
        // Compare the hash values (the 'x' component)
        if ((a.x > b.x) == ascending) {
            // Swap them in memory
            hashGrid[idx]     = b;
            hashGrid[partner] = a;
        }
    }
}