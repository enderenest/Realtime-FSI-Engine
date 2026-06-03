#include "fluid/SDFBoundary.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <omp.h>

namespace {

// Which surface feature of a triangle is closest to the query point.
// Determines which pseudo-normal to use for the inside/outside sign.
enum class Feature { FACE, EDGE_AB, EDGE_AC, EDGE_BC, VERTEX_A, VERTEX_B, VERTEX_C };

struct ClosestResult { Eigen::Vector3d point; Feature feature; };

// Ericson "Real-Time Collision Detection" 5.1.5, extended to return feature type.
ClosestResult closestPointOnTriangle(
    const Eigen::Vector3d& P,
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C)
{
    const Eigen::Vector3d AB = B - A, AC = C - A, AP = P - A;
    const double d1 = AB.dot(AP), d2 = AC.dot(AP);
    if (d1 <= 0 && d2 <= 0) return {A, Feature::VERTEX_A};

    const Eigen::Vector3d BP = P - B;
    const double d3 = AB.dot(BP), d4 = AC.dot(BP);
    if (d3 >= 0 && d4 <= d3) return {B, Feature::VERTEX_B};

    const Eigen::Vector3d CP = P - C;
    const double d5 = AB.dot(CP), d6 = AC.dot(CP);
    if (d6 >= 0 && d5 <= d6) return {C, Feature::VERTEX_C};

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
        return {A + (d1 / (d1 - d3)) * AB, Feature::EDGE_AB};

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
        return {A + (d2 / (d2 - d6)) * AC, Feature::EDGE_AC};

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return {B + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (C - B), Feature::EDGE_BC};

    const double denom = 1.0 / (va + vb + vc);
    return {A + (vb * denom) * AB + (vc * denom) * AC, Feature::FACE};
}

// Triangle with per-feature pseudo-normals (Baerentzen & Aanaes 2005).
// Using the face normal alone is wrong at edges/vertices shared between
// triangles: whichever adjacent face is nearest first gives an arbitrary sign.
struct Tri {
    Eigen::Vector3d A, B, C;
    int iA, iB, iC;
    Eigen::Vector3d faceNormal;
    Eigen::Vector3d edgeNormalAB, edgeNormalAC, edgeNormalBC; // fallback = faceNormal
};

// ============================================================
// TriangleBVH
// ------------------------------------------------------------
// Median-split bounding-volume hierarchy over the mesh triangles. Built once;
// each voxel then finds its nearest triangle in O(log F) via branch-and-bound
// instead of scanning all F triangles. This is what turns the SDF build from
// O(voxels * triangles) into something that finishes in tens of milliseconds.
//
// closestPoint() returns the closest surface point, the triangle it lies on,
// and which feature (face/edge/vertex) — the caller needs the feature to pick
// the correct pseudo-normal for the inside/outside sign.
// ============================================================
class TriangleBVH {
public:
    struct Hit {
        double          dist2 = 1e30;
        Eigen::Vector3d point  = Eigen::Vector3d::Zero();
        Feature         feature = Feature::FACE;
        int             tri     = -1;
    };

    void build(const std::vector<Tri>& tris) {
        _tris = &tris;
        const int n = static_cast<int>(tris.size());

        _order.resize(n);
        _triAABBMin.resize(n);
        _triAABBMax.resize(n);
        _centroid.resize(n);
        for (int i = 0; i < n; ++i) {
            _order[i] = i;
            const Tri& t = tris[i];
            _triAABBMin[i] = t.A.cwiseMin(t.B).cwiseMin(t.C);
            _triAABBMax[i] = t.A.cwiseMax(t.B).cwiseMax(t.C);
            _centroid[i]   = (t.A + t.B + t.C) / 3.0;
        }

        _nodes.clear();
        _nodes.reserve(2 * std::max(1, n));
        if (n > 0) buildRecursive(0, n);
    }

    // Nearest point on the mesh surface to P (with feature + triangle index).
    Hit closestPoint(const Eigen::Vector3d& P) const {
        Hit best;
        if (!_nodes.empty()) query(0, P, best);
        return best;
    }

private:
    static constexpr int kLeafSize = 4;

    struct Node {
        Eigen::Vector3d bmin, bmax;
        int start = 0, count = 0;   // range into _order (leaf only)
        int left = -1, right = -1;  // child node indices (-1 => leaf)
    };

    // Squared distance from a point to an AABB (0 if inside).
    static double sqDistPointAABB(const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& bmin,
                                  const Eigen::Vector3d& bmax) {
        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double v = p[i];
            if      (v < bmin[i]) { const double t = bmin[i] - v; d2 += t * t; }
            else if (v > bmax[i]) { const double t = v - bmax[i]; d2 += t * t; }
        }
        return d2;
    }

    // Returns the index of the node it creates. Recurses; because _nodes may
    // reallocate, children are linked by index after they are built.
    int buildRecursive(int start, int count) {
        const int nodeIdx = static_cast<int>(_nodes.size());
        _nodes.emplace_back();

        // Bounds over the triangles in this range (full triangle AABBs, so the
        // node tightly encloses geometry for correct distance pruning).
        Eigen::Vector3d bmin( 1e30,  1e30,  1e30);
        Eigen::Vector3d bmax(-1e30, -1e30, -1e30);
        Eigen::Vector3d cmin( 1e30,  1e30,  1e30);
        Eigen::Vector3d cmax(-1e30, -1e30, -1e30);
        for (int i = start; i < start + count; ++i) {
            const int t = _order[i];
            bmin = bmin.cwiseMin(_triAABBMin[t]);
            bmax = bmax.cwiseMax(_triAABBMax[t]);
            cmin = cmin.cwiseMin(_centroid[t]);
            cmax = cmax.cwiseMax(_centroid[t]);
        }
        _nodes[nodeIdx].bmin  = bmin;
        _nodes[nodeIdx].bmax  = bmax;
        _nodes[nodeIdx].start = start;
        _nodes[nodeIdx].count = count;

        if (count <= kLeafSize) return nodeIdx;   // leaf

        // Split at the median centroid along the widest centroid axis.
        const Eigen::Vector3d ext = cmax - cmin;
        int axis = 0;
        if (ext.y() > ext.x()) axis = 1;
        if (ext.z() > ext[axis]) axis = 2;

        const int mid = count / 2;
        std::nth_element(
            _order.begin() + start,
            _order.begin() + start + mid,
            _order.begin() + start + count,
            [&](int a, int b) { return _centroid[a][axis] < _centroid[b][axis]; });

        const int leftIdx  = buildRecursive(start, mid);
        const int rightIdx = buildRecursive(start + mid, count - mid);
        _nodes[nodeIdx].left  = leftIdx;
        _nodes[nodeIdx].right = rightIdx;
        return nodeIdx;
    }

    void query(int nodeIdx, const Eigen::Vector3d& P, Hit& best) const {
        const Node& node = _nodes[nodeIdx];

        // Prune: if the whole box is farther than the best hit, skip it.
        if (sqDistPointAABB(P, node.bmin, node.bmax) >= best.dist2) return;

        if (node.left == -1) {   // leaf — test the triangles
            for (int i = node.start; i < node.start + node.count; ++i) {
                const int t = _order[i];
                const Tri& tri = (*_tris)[t];
                const ClosestResult cr = closestPointOnTriangle(P, tri.A, tri.B, tri.C);
                const double d2 = (P - cr.point).squaredNorm();
                if (d2 < best.dist2) {
                    best.dist2   = d2;
                    best.point   = cr.point;
                    best.feature = cr.feature;
                    best.tri     = t;
                }
            }
            return;
        }

        // Descend into the nearer child first so the farther one prunes harder.
        const double dL = sqDistPointAABB(P, _nodes[node.left].bmin,  _nodes[node.left].bmax);
        const double dR = sqDistPointAABB(P, _nodes[node.right].bmin, _nodes[node.right].bmax);
        if (dL <= dR) {
            if (dL < best.dist2) query(node.left,  P, best);
            if (dR < best.dist2) query(node.right, P, best);
        } else {
            if (dR < best.dist2) query(node.right, P, best);
            if (dL < best.dist2) query(node.left,  P, best);
        }
    }

    const std::vector<Tri>*      _tris = nullptr;
    std::vector<Node>            _nodes;
    std::vector<int>             _order;        // triangle indices, permuted by build
    std::vector<Eigen::Vector3d> _triAABBMin;   // per original-triangle AABB
    std::vector<Eigen::Vector3d> _triAABBMax;
    std::vector<Eigen::Vector3d> _centroid;
};

} // namespace

// Triangle list + BVH retained after the build so closestTriangle() can serve
// contact queries. bvh stores a pointer into tris, so this object must outlive
// any query and must not be relocated (it lives on the heap via shared_ptr).
struct SDFBoundary::Accel {
    std::vector<Tri> tris;
    TriangleBVH      bvh;
};

void SDFBoundary::buildFromMesh(
    const std::vector<std::array<double, 3>>& verts,
    const std::vector<std::array<int, 3>>&    faces,
    double resolution, bool verbose)
{
    if (verbose) {
    #ifdef _OPENMP
        std::cout << "OpenMP enabled, max threads = " << omp_get_max_threads() << "\n";
    #else
        std::cout << "OpenMP NOT enabled\n";
    #endif
    }

    // AABB
    Eigen::Vector3d bboxMin( 1e30,  1e30,  1e30);
    Eigen::Vector3d bboxMax(-1e30, -1e30, -1e30);
    for (const auto& v : verts) {
        bboxMin = bboxMin.cwiseMin(Eigen::Vector3d(v[0], v[1], v[2]));
        bboxMax = bboxMax.cwiseMax(Eigen::Vector3d(v[0], v[1], v[2]));
    }

    const double buffer = resolution * 5.0;
    _origin = bboxMin.array() - buffer;
    const Eigen::Vector3d span = (bboxMax.array() + buffer) - _origin.array();

    _cellSize   = resolution;
    _dimensions = Eigen::Vector3i(
        static_cast<int>(std::ceil(span.x() / _cellSize)),
        static_cast<int>(std::ceil(span.y() / _cellSize)),
        static_cast<int>(std::ceil(span.z() / _cellSize))
    );

    const int total = _dimensions.x() * _dimensions.y() * _dimensions.z();
    _distances.assign(total, 0.0f);

    // Build the triangle list with per-feature pseudo-normals. Stored in _accel
    // (on the heap) so the list + BVH survive the build for later contact queries.
    _accel = std::make_shared<Accel>();
    std::vector<Tri>& tris = _accel->tris;
    tris.reserve(faces.size());
    for (const auto& f : faces) {
        Tri tri;
        tri.iA = f[0]; tri.iB = f[1]; tri.iC = f[2];
        tri.A = Eigen::Vector3d(verts[f[0]][0], verts[f[0]][1], verts[f[0]][2]);
        tri.B = Eigen::Vector3d(verts[f[1]][0], verts[f[1]][1], verts[f[1]][2]);
        tri.C = Eigen::Vector3d(verts[f[2]][0], verts[f[2]][1], verts[f[2]][2]);
        Eigen::Vector3d n = (tri.B - tri.A).cross(tri.C - tri.A);
        const double len = n.norm();
        if (len > 1e-12) n /= len;
        tri.faceNormal = n;
        tri.edgeNormalAB = tri.edgeNormalAC = tri.edgeNormalBC = n;
        tris.push_back(tri);
    }

    // Angle-weighted vertex normals: each incident face weighted by its interior
    // angle at that vertex so skinny triangles don't dominate.
    const int numVerts = static_cast<int>(verts.size());
    std::vector<Eigen::Vector3d> vertNormals(numVerts, Eigen::Vector3d::Zero());
    for (const auto& tri : tris) {
        auto accumulate = [&](int vi, const Eigen::Vector3d& vtx,
                              const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
            const double angle = std::acos(std::clamp(
                (p1 - vtx).normalized().dot((p2 - vtx).normalized()), -1.0, 1.0));
            vertNormals[vi] += angle * tri.faceNormal;
        };
        accumulate(tri.iA, tri.A, tri.B, tri.C);
        accumulate(tri.iB, tri.B, tri.A, tri.C);
        accumulate(tri.iC, tri.C, tri.A, tri.B);
    }
    for (auto& n : vertNormals) {
        const double len = n.norm();
        if (len > 1e-12) n /= len;
    }

    // Edge normals: average of the two incident face normals.
    using EdgeKey = std::pair<int, int>;
    std::map<EdgeKey, std::vector<int>> edgeToTris;
    for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
        const Tri& tri = tris[t];
        auto add = [&](int a, int b) {
            edgeToTris[{std::min(a, b), std::max(a, b)}].push_back(t);
        };
        add(tri.iA, tri.iB); add(tri.iA, tri.iC); add(tri.iB, tri.iC);
    }
    for (auto& [edge, ids] : edgeToTris) {
        if (ids.size() != 2) continue;
        const Eigen::Vector3d sum = tris[ids[0]].faceNormal + tris[ids[1]].faceNormal;
        const double len = sum.norm();
        if (len < 1e-12) continue; // 180 deg fold -- keep face-normal fallback
        const Eigen::Vector3d edgeN = sum / len;
        for (int t : ids) {
            Tri& tri = tris[t];
            const int a = edge.first, b = edge.second;
            if      (std::min(tri.iA, tri.iB) == a && std::max(tri.iA, tri.iB) == b) tri.edgeNormalAB = edgeN;
            else if (std::min(tri.iA, tri.iC) == a && std::max(tri.iA, tri.iC) == b) tri.edgeNormalAC = edgeN;
            else                                                                     tri.edgeNormalBC = edgeN;
        }
    }

    // Spatial acceleration: build a BVH once, then each voxel queries it in
    // O(log F). This replaces the old O(voxels * triangles) brute-force scan.
    // Kept in _accel so closestTriangle() can reuse it after the build.
    TriangleBVH& bvh = _accel->bvh;
    bvh.build(tris);

	// enable CPU parallelism over the voxel grid
    #pragma omp parallel for schedule(dynamic, 1)
    for (int x = 0; x < _dimensions.x(); ++x) {
        for (int y = 0; y < _dimensions.y(); ++y) {
            for (int z = 0; z < _dimensions.z(); ++z) {
                const Eigen::Vector3d P(
                    _origin.x() + x * _cellSize,
                    _origin.y() + y * _cellSize,
                    _origin.z() + z * _cellSize
                );

                const TriangleBVH::Hit hit = bvh.closestPoint(P);

                const Tri& tri = tris[hit.tri];
                Eigen::Vector3d pseudoNormal;
                switch (hit.feature) {
                    case Feature::FACE:     pseudoNormal = tri.faceNormal;      break;
                    case Feature::EDGE_AB:  pseudoNormal = tri.edgeNormalAB;    break;
                    case Feature::EDGE_AC:  pseudoNormal = tri.edgeNormalAC;    break;
                    case Feature::EDGE_BC:  pseudoNormal = tri.edgeNormalBC;    break;
                    case Feature::VERTEX_A: pseudoNormal = vertNormals[tri.iA]; break;
                    case Feature::VERTEX_B: pseudoNormal = vertNormals[tri.iB]; break;
                    case Feature::VERTEX_C: pseudoNormal = vertNormals[tri.iC]; break;
                }
                const double sign = ((P - hit.point).dot(pseudoNormal) >= 0.0) ? 1.0 : -1.0;

                _distances[index(x, y, z)] = static_cast<float>(sign * std::sqrt(hit.dist2));
            }
        }
    }

    if (verbose) {
        std::cout << "SDF computed: "
                  << _dimensions.x() << "x" << _dimensions.y() << "x" << _dimensions.z()
                  << " (" << total << " voxels, " << tris.size() << " triangles, BVH)\n";
    }
}

void SDFBoundary::uploadToGPU() {
    if (_distances.empty()) return;

    const int W = _dimensions.x();
    const int H = _dimensions.y();
    const int D = _dimensions.z();

    // Repack from CPU grid order (index = x*H*D + y*D + z, i.e. z fastest) into
    // GL texture order (index = x + W*(y + H*z), i.e. x fastest). Getting this
    // wrong silently transposes the field and the fluid leaks through the mesh.
    std::vector<float> texData(static_cast<size_t>(W) * H * D);
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                texData[static_cast<size_t>(x) + W * (static_cast<size_t>(y) + static_cast<size_t>(H) * z)]
                    = _distances[index(x, y, z)];

    _texture.upload(W, H, D, texData);
}

SDFBoundary::ClosestSurface SDFBoundary::closestTriangle(const PVec3& p) const {
    ClosestSurface out;
    if (!_accel || _accel->tris.empty()) return out;

    const TriangleBVH::Hit hit = _accel->bvh.closestPoint(Eigen::Vector3d(p.x, p.y, p.z));
    out.face  = hit.tri;   // original triangle index == index into the faces array
    out.point = PVec3{ static_cast<F32>(hit.point.x()),
                       static_cast<F32>(hit.point.y()),
                       static_cast<F32>(hit.point.z()) };
    return out;
}
