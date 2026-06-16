#ifndef TEXTURE3D_CLASS_H
#define TEXTURE3D_CLASS_H

#include <glad/glad.h>
#include <vector>

#include "core/Types.h"

// RAII wrapper around a single-channel float GL_TEXTURE_3D.
//
// Holds the boundary SDF on the GPU so the PBF compute shaders can query it
// with hardware trilinear interpolation (a single texture() call). The only
// thing that ever crosses CPU->GPU is this texture, and only when the SDF is
// (re)built — never per particle, never per frame.
class Texture3D {
public:
    Texture3D() = default;
    ~Texture3D();

    Texture3D(const Texture3D&) = delete;
    Texture3D& operator=(const Texture3D&) = delete;

    // Upload (or re-upload) an R32F volume. Allocates lazily on the first call;
    // if the dimensions are unchanged it does a cheap sub-image update instead
    // of reallocating storage (the deform-time local-update path).
    // 'data' must be in texture order: index = x + width * (y + height * z).
    void upload(int width, int height, int depth, const std::vector<float>& data);

    // Update a sub-box of the already-allocated volume via glTexSubImage3D (the
    // dirty-update path). Offset (x0,y0,z0) and extent (w,h,d) are in voxels;
    // 'data' must be w*h*d floats in sub-box texture order: i = x + w*(y + h*z).
    // No-op if storage has not been allocated yet (call upload() first).
    void uploadSubRegion(int x0, int y0, int z0, int w, int h, int d,
                         const std::vector<float>& data);

    // Bind to a texture image unit for sampling (sampler3D in GLSL).
    void bindToUnit(GLuint unit) const;

    GLuint getID() const { return _id; }

private:
    GLuint _id = 0;
    int    _w = 0, _h = 0, _d = 0;
};

#endif // TEXTURE3D_CLASS_H
