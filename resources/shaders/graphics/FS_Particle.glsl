#version 430 core

in vec3 velocity;
out vec4 FragColor;

// 0 = debug heatmap dots (cheap flat sprites)
// 1 = shaded liquid spheres (render layer)
uniform int renderMode;

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
    // Circle mask from the point sprite (shared by both modes).
    vec2 centred = (gl_PointCoord - vec2(0.5)) * 2.0;
    float d2     = dot(centred, centred);
    if (d2 > 1.0) discard;

    float speed = length(velocity);

    if (renderMode == 1) {
        // ---- Render layer: shaded spherical imposter ----
        // A lit sphere reconstructed entirely from the sprite — no extra geometry,
        // one cheap lighting term. View-space normal: z faces the camera.
        vec3 N = vec3(centred.x, -centred.y, sqrt(max(0.0, 1.0 - d2)));
        vec3 L = normalize(vec3(0.35, 0.55, 0.75));   // view-space key light
        float diff = max(dot(N, L), 0.0);
        vec3  H    = normalize(L + vec3(0.0, 0.0, 1.0));
        float spec = pow(max(dot(N, H), 0.0), 40.0);

        // Water blue, tinted toward bright foam with speed.
        float t    = clamp(speed / 8.0, 0.0, 1.0);
        vec3  base = mix(vec3(0.10, 0.32, 0.72), vec3(0.65, 0.88, 1.0), t);
        float rim  = pow(1.0 - N.z, 3.0);             // fresnel-ish edge lightening

        vec3 col = base * (0.30 + 0.70 * diff) + vec3(spec) * 0.55 + rim * 0.12;
        FragColor = vec4(col, 1.0);                   // opaque -> correct depth occlusion
    } else {
        // ---- Debug: flat heatmap dot with a soft edge ----
        float alpha = 1.0 - smoothstep(0.85, 1.0, d2);
        float t     = clamp(speed / 10.0, 0.0, 1.0);
        FragColor   = vec4(heatmap(t), alpha);
    }
}
