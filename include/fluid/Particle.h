#ifndef PARTICLE_H
#define PARTICLE_H

#include "core/Types.h"

// GPU-compatible layout: each PVec3 is followed by 4 bytes of padding to match
// GLSL std430 vec3 alignment (16 bytes). Total size: 48 bytes = 3 x vec4.
struct Particle {
	PVec3 pos;     F32 _pad0 = 0.f;
	PVec3 predPos; F32 _pad1 = 0.f;
	PVec3 vel;     F32 _pad2 = 0.f;
};

#endif