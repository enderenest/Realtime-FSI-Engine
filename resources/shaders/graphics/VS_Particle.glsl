#version 430 core

struct Particle {
    vec4 pos;
    vec4 vel;
};

// Aligned with our PBFluids architecture
layout(std430, binding = 0) buffer ParticleBuffer { 
    Particle particles[]; 
};

uniform float scale; // Controls point size on screen
uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

out vec3 velocity;

void main()
{
    vec3 instancePos = particles[gl_InstanceID].pos.xyz;
    vec3 instanceVel = particles[gl_InstanceID].vel.xyz;
    
    gl_Position = projection * view * model * vec4(instancePos, 1.0);
    
    // CRITICAL: Tells the Fragment Shader how large the gl_PointCoord canvas is
    gl_PointSize = scale; 
    
    velocity = instanceVel;
}