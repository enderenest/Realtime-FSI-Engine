#ifndef SDF_BOUNDARY_H
#define SDF_BOUNDARY_H

#include <array>
#include <vector>

#include <Eigen/Core>
#include <glad/glad.h>

#include "core/Types.h"
#include "opengl/Texture3D.h"

// ============================================================
// SDFBoundary
// ------------------------------------------------------------
// Signed-distance-field boundary for fluid-solid coupling.
//
// The build/query split (the reason this is fast):
//   * BUILD runs on the CPU, rarely — once at startup, then only locally when
//     the mesh deforms. This is the O(voxels * triangles) brute-force pass.
//   * QUERY runs on the GPU, millions of times per frame, inside the PBF
//     compute shader as one hardware-trilinear texture() lookup.
//
// This class owns both halves: the CPU grid of signed distances and the GPU
// 3D texture the solver samples. Typical use:
//
//     SDFBoundary sdf;
//     sdf.buildFromMesh(verts, faces, cellSize);   // CPU
//     sdf.uploadToGPU();                            // CPU -> GPU (once)
//     pbf.setSDFBoundary(&sdf);                     // solver queries it
//
// Sign convention: distance is negative inside the mesh, positive outside
// (Baerentzen & Aanaes pseudo-normal sign test), so the gradient points
// outward — exactly the direction to push an intruding particle.
// ============================================================
class SDFBoundary {
public:
    // ---- CPU build ----
    // verts:      vertex positions.
    // faces:      triangle indices into verts (already triangulated).
    // resolution: voxel edge length in world units. Match it to the PBF
    //             particle diameter so the boundary resolves the fluid.
    void buildFromMesh(const std::vector<std::array<double, 3>>& verts,
                       const std::vector<std::array<int, 3>>&    faces,
                       double resolution);

    // ---- GPU upload ----
    // Repacks the CPU grid into texture order and uploads it as an R32F volume.
    // Requires a current GL context. Call once after building, and again after
    // a local rebuild when the mesh deforms (re-uploads in place if the
    // dimensions are unchanged).
    void uploadToGPU();

    // Bind the SDF texture to a sampler image unit for the query shader.
    void bindForQuery(GLuint textureUnit) const { _texture.bindToUnit(textureUnit); }

    // ---- Query metadata (these become shader uniforms) ----
    bool                   valid()      const { return !_distances.empty(); }
    const Eigen::Vector3d& origin()     const { return _origin; }
    double                 cellSize()   const { return _cellSize; }
    const Eigen::Vector3i& dimensions() const { return _dimensions; }

    // ---- CPU-side grid access (test visualizer, local deform updates) ----
    const std::vector<float>& distances() const { return _distances; }
    int index(int x, int y, int z) const {
        return x * _dimensions.y() * _dimensions.z() + y * _dimensions.z() + z;
    }

private:
    Eigen::Vector3d    _origin     = Eigen::Vector3d::Zero();  // world pos of voxel (0,0,0)
    double             _cellSize   = 0.0;
    Eigen::Vector3i    _dimensions = Eigen::Vector3i::Zero();
    std::vector<float> _distances;                             // CPU grid, indexed by index()

    Texture3D          _texture;                               // GPU 3D texture (R32F)
};

#endif // SDF_BOUNDARY_H
