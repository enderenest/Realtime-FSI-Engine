#version 430 core

// Shared vertex shader for the fluid depth and thickness passes. Reads particle
// positions straight from the SSBO and sizes each point sprite so it covers the
// projected sphere of radius pointRadius.

struct Particle {
    vec4 pos;
    vec4 predPos;
    vec4 vel;
};

layout(std430, binding = 0) buffer ParticleBuffer { Particle particles[]; };

uniform mat4  view;
uniform mat4  projection;
uniform float pointRadius;    // world-space sphere radius
uniform float screenHeight;   // viewport height in pixels

out vec3  eyePos;             // eye-space sphere center
out float vSpeed;

void main()
{
    vec3 wp = particles[gl_InstanceID].pos.xyz;
    vSpeed  = length(particles[gl_InstanceID].vel.xyz);

    vec4 ep = view * vec4(wp, 1.0);
    eyePos  = ep.xyz;
    gl_Position = projection * ep;

    // Point-sprite diameter in pixels = projected sphere diameter.
    gl_PointSize = pointRadius * projection[1][1] * screenHeight / max(-ep.z, 1e-3);
}
