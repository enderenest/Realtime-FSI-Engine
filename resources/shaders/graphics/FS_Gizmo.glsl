#version 330 core

out vec4 FragColor;

uniform vec3 gizmoColor;

void main()
{
    // Discard pixels outside a circle to make points look like spheres
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0)
        discard;

    FragColor = vec4(gizmoColor, 1.0);
}
