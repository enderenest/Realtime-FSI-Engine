#version 430 core

// Full-screen triangle generated from gl_VertexID (draw 3 vertices, no VBO).
out vec2 vUV;

void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;                                  // 0..2, so 0..1 covers the screen
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
