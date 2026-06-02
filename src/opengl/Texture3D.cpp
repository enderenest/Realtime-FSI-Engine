#include "opengl/Texture3D.h"

#include <cassert>

Texture3D::~Texture3D() {
    if (_id) glDeleteTextures(1, &_id);
}

void Texture3D::upload(int width, int height, int depth, const std::vector<float>& data) {
    assert(static_cast<size_t>(width) * height * depth == data.size());

    const bool needAlloc = (_id == 0) || width != _w || height != _h || depth != _d;

    if (_id == 0) glGenTextures(1, &_id);
    glBindTexture(GL_TEXTURE_3D, _id);

    // Tightly packed float rows.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    if (needAlloc) {
        _w = width; _h = height; _d = depth;

        // Hardware trilinear filtering — this is the whole point of using a
        // texture: querySDF() in the shader gets interpolation for free.
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Clamp to edge: a particle that strays just outside the grid reads the
        // nearest border distance rather than wrapping to the far side.
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, _w, _h, _d, 0,
                     GL_RED, GL_FLOAT, data.data());
    } else {
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, _w, _h, _d,
                        GL_RED, GL_FLOAT, data.data());
    }

    glBindTexture(GL_TEXTURE_3D, 0);
}

void Texture3D::bindToUnit(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_3D, _id);
}
