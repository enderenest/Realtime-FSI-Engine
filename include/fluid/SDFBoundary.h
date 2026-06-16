#ifndef SDF_BOUNDARY_H
#define SDF_BOUNDARY_H

#include <array>
#include <cstdint>
#include <memory>
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
    // verbose: print build stats to stdout. Pass false for per-frame rebuilds
    // (two-way coupling) to avoid flooding the console.
    void buildFromMesh(const std::vector<std::array<double, 3>>& verts,
                       const std::vector<std::array<int, 3>>&    faces,
                       double resolution, bool verbose = true);

    // ---- Deformation refit (two-way coupling) ----
    // Cheap update when the mesh deforms but its TOPOLOGY is unchanged (same
    // face list, same vertex count, vertices just moved). Updates triangle
    // corner positions + face normals (O(F)), refits the retained BVH boxes in
    // place instead of rebuilding it (O(F) vs O(F log F)), then recomputes the
    // distance grid over the existing origin/dimensions.
    //
    // Intentionally O(F): the edge/vertex pseudo-normals are kept from the last
    // full buildFromMesh (recomputing them needs the O(F log F) edge-adjacency
    // pass). They drift slowly for small per-frame deformations; call
    // buildFromMesh() periodically (e.g. every N frames) to refresh them and
    // retighten the BVH. Caller must uploadToGPU() afterwards, as with build.
    void refitFromMesh(const std::vector<std::array<double, 3>>& verts);

    // ---- Dirty (local) deformation update (two-way coupling, most frames) ----
    // Like refitFromMesh, but recomputes only the voxels near vertices that
    // actually moved instead of the whole grid. Steps:
    //   1. Detect moved vertices: ||newVerts[i] - lastVerts[i]|| > moveEps.
    //      (Laplacian/ARAP deformation is global but decays fast, so this set is
    //       small in practice — measured, not assumed-local.)
    //   2. Mark dirty voxels within bandRadius of BOTH the old and new position
    //      of each moved vertex (old clears the stale surface, new writes it).
    //   3. Refit the BVH (O(F)) and recompute signed distance for dirty voxels.
    // Records the dirty voxel bounding box for uploadDirtyRegion(). Returns true
    // if anything was dirty (i.e. an upload is needed). Falls back to a full
    // refitFromMesh if no baseline exists yet (first call / size mismatch).
    //
    // bandRadius (world units): pass bandWidth + maxDisplacement + cellSize.
    // Pair with a periodic full buildFromMesh() to reset accumulated drift.
    bool refitDirty(const std::vector<std::array<double, 3>>& newVerts,
                    double bandRadius, double moveEps);

    // Debug aid for the dirty path: after each refitDirty, recompute the full
    // grid and assert the dirty voxels match it (the "match where they overlap"
    // check). Off by default; doubles the work when on.
    void setDebugVerifyDirty(bool on) { _debugVerifyDirty = on; }

    // ---- GPU upload ----
    // Repacks the CPU grid into texture order and uploads it as an R32F volume.
    // Requires a current GL context. Call once after building, and again after
    // a local rebuild when the mesh deforms (re-uploads in place if the
    // dimensions are unchanged).
    void uploadToGPU();

    // Upload only the dirty voxel bounding box recorded by the last refitDirty()
    // via glTexSubImage3D, instead of re-pushing the whole volume. No-op if the
    // last update marked nothing dirty. Requires a current GL context.
    void uploadDirtyRegion();

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

    // ---- Closest-surface query (CPU, two-way coupling contact detection) ----
    // Result of closestTriangle(): the nearest triangle to a query point and the
    // closest point on that triangle's surface.
    struct ClosestSurface {
        int   face = -1;            // index into the 'faces' passed to buildFromMesh (-1 if none)
        PVec3 point{ 0.f, 0.f, 0.f }; // closest point on the surface
    };

    // Nearest triangle to p, using the same BVH built for the distance field
    // (kept alive after buildFromMesh). Returns face = -1 if no mesh is built.
    ClosestSurface closestTriangle(const PVec3& p) const;

private:
    Eigen::Vector3d    _origin     = Eigen::Vector3d::Zero();  // world pos of voxel (0,0,0)
    double             _cellSize   = 0.0;
    Eigen::Vector3i    _dimensions = Eigen::Vector3i::Zero();
    std::vector<float> _distances;                             // CPU grid, indexed by index()

    Texture3D          _texture;                               // GPU 3D texture (R32F)

    // Signed distance at a world point P: nearest surface point via the BVH,
    // sign from the pseudo-normal of the closest feature. The single-voxel
    // kernel shared by the full grid pass and the dirty pass.
    float signedDistanceAt(const Eigen::Vector3d& P) const;

    // Fills _distances by querying the retained BVH at every voxel center.
    // Shared by buildFromMesh (after a fresh build) and refitFromMesh (after a
    // refit). Reads geometry + pseudo-normals from _accel; respects _origin,
    // _cellSize, _dimensions (all set before it is called).
    void recomputeDistances();

    // Pulls moved vertex positions into the retained triangles, refreshes face
    // normals, and refits the BVH in place. The geometry half of refitFromMesh /
    // refitDirty (does NOT touch the distance grid).
    void updateGeometryAndRefit(const std::vector<std::array<double, 3>>& verts);

    // Vertex positions the current grid was last built/refit from, kept so
    // refitDirty() can diff against them and mark the OLD-position dirty band.
    std::vector<std::array<double, 3>> _buildVerts;

    // Dirty-voxel working state (persisted to avoid per-frame reallocation).
    std::vector<std::uint8_t> _dirtyMask;   // grid-sized; 1 if voxel is in _dirtyList
    std::vector<int>          _dirtyList;    // linear indices of dirty voxels
    Eigen::Vector3i _dirtyMin = Eigen::Vector3i::Zero();  // dirty box (voxel coords, inclusive)
    Eigen::Vector3i _dirtyMax = Eigen::Vector3i::Zero();
    bool _dirtyValid      = false;           // did the last refitDirty mark anything?
    bool _debugVerifyDirty = false;

    // CPU triangle list + BVH, kept alive after the build so closestTriangle()
    // can answer contact queries. Defined in the .cpp (holds .cpp-local types).
    struct Accel;
    std::shared_ptr<Accel> _accel;
};

#endif // SDF_BOUNDARY_H
