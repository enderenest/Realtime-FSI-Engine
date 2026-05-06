#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;

void main()
{
    vec4 viewPos = view * vec4(aPos, 1.0);
    gl_Position = projection * viewPos;

    // Scale point size by distance so gizmos stay visible
    gl_PointSize = pointSize / -viewPos.z;
}
