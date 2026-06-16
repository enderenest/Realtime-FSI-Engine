#version 330 core

in vec3 vertexColor;
in vec3 vWorldPos;
out vec4 FragColor;

// 0 = flat colour (wireframe debug view)
// 1 = solid shaded surface (render layer)
uniform int renderMode;

void main()
{
    if (renderMode == 1) {
        // Flat face normal from screen-space derivatives — no vertex normals needed,
        // one cheap diffuse term. Two-sided so open/back faces still light up.
        vec3 N = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        vec3 L = normalize(vec3(0.5, 0.85, 0.55));
        float diff = abs(dot(N, L));

        vec3 base = vertexColor;                  // warm surface; anchors stay red
        vec3 col  = base * (0.30 + 0.70 * diff);
        FragColor = vec4(col, 1.0);
    } else {
        FragColor = vec4(vertexColor, 1.0);
    }
}
