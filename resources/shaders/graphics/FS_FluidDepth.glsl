#version 430 core

// Depth pass: shade each sprite as a sphere imposter and write the eye-space
// distance (positive, 0 = no fluid) to an R32F target, with a corrected hardware
// depth so the nearest sphere wins.

in vec3  eyePos;
in float vSpeed;

uniform mat4  projection;
uniform float pointRadius;

out float fragDist;   // positive eye-space distance to the sphere surface

void main()
{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;

    vec3 N       = vec3(c, sqrt(1.0 - r2));        // sphere normal facing the camera
    vec3 fragEye = eyePos + N * pointRadius;        // surface point in eye space

    // Correct the hardware depth so overlapping spheres occlude properly.
    vec4 clip   = projection * vec4(fragEye, 1.0);
    gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;

    fragDist = -fragEye.z;                          // camera looks down -z
}
