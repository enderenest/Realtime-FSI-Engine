#version 430 core

in vec3 velocity;
out vec4 FragColor;

// 5-stop heatmap: dark blue -> cyan -> yellow -> orange -> red
vec3 heatmap(float t)
{
    const vec3 c0 = vec3(0.05, 0.05, 0.30);  // dark blue
    const vec3 c1 = vec3(0.10, 0.70, 0.90);  // cyan
    const vec3 c2 = vec3(0.95, 0.85, 0.20);  // yellow
    const vec3 c3 = vec3(0.95, 0.55, 0.10);  // orange
    const vec3 c4 = vec3(0.90, 0.15, 0.10);  // red

    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
                   return mix(c3, c4, (t - 0.75) / 0.25);
}

void main()
{
    // Circle mask from point sprite
    vec2 centred = (gl_PointCoord - vec2(0.5)) * 2.0;
    float d2     = dot(centred, centred);
    if (d2 > 1.0) discard;

    // Soft edge
    float alpha = 1.0 - smoothstep(0.85, 1.0, d2);

    // Speed → color (normalize to a wider range so colors stay stable)
    float speed = length(velocity);
    float t     = clamp(speed / 10.0, 0.0, 1.0);
    vec3  col   = heatmap(t);

    FragColor = vec4(col, alpha);
}
