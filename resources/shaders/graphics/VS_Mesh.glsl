#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 VP;

out vec3 vertexColor;
out vec3 vWorldPos;   // world-space position, for the shaded render layer

void main()
{
    gl_Position = VP * vec4(aPos, 1.0);
    vertexColor = aColor;
    vWorldPos   = aPos;
}
