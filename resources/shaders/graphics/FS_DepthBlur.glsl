#version 430 core

// Separable bilateral blur of the fluid eye-distance field. Spatial Gaussian for
// smoothing, range Gaussian on the distance difference so silhouettes are kept
// sharp (no bleeding of the surface over the background). Empty pixels (<=0) stay
// empty. Run once horizontally then once vertically.

in vec2 vUV;

uniform sampler2D depthTex;     // R32F eye-distance (0 = empty)
uniform vec2      texelDir;     // (1/w, 0) horizontal  or  (0, 1/h) vertical
uniform float     blurRadiusPx; // kernel half-width in pixels
uniform float     depthFalloff; // range sigma (world units)

out float outDist;

void main()
{
    float center = texture(depthTex, vUV).r;
    if (center <= 0.0) { outDist = 0.0; return; }

    float sum = 0.0, wsum = 0.0;
    int   R   = int(blurRadiusPx);
    float s2  = 2.0 * blurRadiusPx * blurRadiusPx + 1e-3;
    float r2  = 2.0 * depthFalloff * depthFalloff + 1e-6;

    for (int i = -R; i <= R; ++i) {
        float s = texture(depthTex, vUV + texelDir * float(i)).r;
        if (s <= 0.0) continue;
        float wSpace = exp(-float(i * i) / s2);
        float dd     = s - center;
        float wRange = exp(-(dd * dd) / r2);
        float w      = wSpace * wRange;
        sum  += s * w;
        wsum += w;
    }
    outDist = (wsum > 0.0) ? sum / wsum : center;
}
