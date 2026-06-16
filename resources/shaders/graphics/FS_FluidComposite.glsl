#version 430 core

// Screen-space fluid composite. From the smoothed eye-distance field it
// reconstructs the surface position and normal, then shades clear water:
// refraction of the background, Beer-Lambert tint by thickness, a fresnel mix
// toward a reflection colour, and a specular highlight. Where there is no fluid
// (or opaque geometry is in front) it passes the scene through unchanged.

in vec2 vUV;

uniform sampler2D smoothedDepth;  // R32F eye-distance (0 = empty)
uniform sampler2D thicknessTex;   // R16F accumulated thickness
uniform sampler2D sceneColor;     // opaque scene (mesh + background)
uniform sampler2D sceneDepthTex;  // hardware depth of the opaque scene

uniform vec2  invScreen;          // (1/w, 1/h)
uniform float fx, fy;             // projection[0][0], projection[1][1]
uniform float nearP, farP;
uniform vec3  absorb;             // Beer-Lambert absorption per channel
uniform float refractScale;
uniform vec3  waterTint;

out vec4 FragColor;

// Eye-space position from screen uv and positive eye-distance (dist = -eye.z).
vec3 eyeFromUV(vec2 uv, float dist)
{
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * dist / fx, ndc.y * dist / fy, -dist);
}

// Linear eye-distance (positive) of the opaque scene at uv.
float linSceneDist(vec2 uv)
{
    float d = texture(sceneDepthTex, uv).r;
    float z = d * 2.0 - 1.0;
    return (2.0 * nearP * farP) / (farP + nearP - z * (farP - nearP));
}

void main()
{
    vec3  scene = texture(sceneColor, vUV).rgb;
    float dist  = texture(smoothedDepth, vUV).r;
    if (dist <= 0.0) { FragColor = vec4(scene, 1.0); return; }      // no fluid here

    // Hidden behind opaque geometry?
    if (linSceneDist(vUV) < dist - 0.02) { FragColor = vec4(scene, 1.0); return; }

    // Reconstruct surface position and normal from the distance field.
    vec3 P  = eyeFromUV(vUV, dist);
    vec2 dx = vec2(invScreen.x, 0.0);
    vec2 dy = vec2(0.0, invScreen.y);
    float dxp = texture(smoothedDepth, vUV + dx).r;
    float dxm = texture(smoothedDepth, vUV - dx).r;
    float dyp = texture(smoothedDepth, vUV + dy).r;
    float dym = texture(smoothedDepth, vUV - dy).r;

    // Use the nearer neighbour on each axis to avoid normals bleeding over edges.
    vec3 ddx = (dxp > 0.0 && (abs(dxp - dist) < abs(dist - dxm) || dxm <= 0.0))
               ? eyeFromUV(vUV + dx, dxp) - P
               : P - eyeFromUV(vUV - dx, max(dxm, dist));
    vec3 ddy = (dyp > 0.0 && (abs(dyp - dist) < abs(dist - dym) || dym <= 0.0))
               ? eyeFromUV(vUV + dy, dyp) - P
               : P - eyeFromUV(vUV - dy, max(dym, dist));

    vec3 N = normalize(cross(ddx, ddy));
    if (N.z < 0.0) N = -N;                                   // face the camera

    float thickness = texture(thicknessTex, vUV).r;

    vec3  V    = normalize(-P);
    float fres = pow(clamp(1.0 - dot(N, V), 0.0, 1.0), 5.0);
    fres = mix(0.04, 1.0, fres);

    // Refraction: nudge the background by the surface normal, scaled by thickness.
    vec2 refrUV = clamp(vUV + N.xy * refractScale * clamp(thickness, 0.0, 1.5), 0.0, 1.0);
    vec3 bg     = texture(sceneColor, refrUV).rgb;

    // Beer-Lambert: thin water is clear, deep water takes the tint.
    vec3 atten = exp(-absorb * thickness);
    vec3 refr  = mix(waterTint, bg, atten);

    // Fresnel reflection (simple sky tint) + specular highlight.
    vec3  reflectCol = vec3(0.55, 0.66, 0.78);
    vec3  L = normalize(vec3(0.4, 0.7, 0.6));
    vec3  H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 80.0);

    vec3 col = mix(refr, reflectCol, fres) + spec * 0.8;
    FragColor = vec4(col, 1.0);
}
