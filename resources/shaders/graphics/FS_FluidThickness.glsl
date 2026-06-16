#version 430 core

// Thickness pass: additively accumulate how much fluid each pixel looks through.
// Drawn with additive blending and no depth test, so all sprites along the ray sum.
// Used by the composite for Beer-Lambert tinting (clear at thin edges, blue deep).

in vec3  eyePos;
in float vSpeed;

uniform float pointRadius;

out float thickness;

void main()
{
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;

    // Chord length through the sphere at this sprite texel.
    thickness = 2.0 * pointRadius * sqrt(1.0 - r2);
}
