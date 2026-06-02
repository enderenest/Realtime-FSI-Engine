#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <array>
#include <cmath>
#include <algorithm>
#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"

namespace fs = std::filesystem;

// ============================================================
// OFF Loader
// ============================================================
struct MeshData {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>>    faces;     // triangulated
};

static bool loadOFF(const std::string& filepath, MeshData& out) {
    std::ifstream f(filepath);
    if (!f) { std::cerr << "Cannot open: " << filepath << "\n"; return false; }

    std::string header;
    f >> header;
    if (header != "OFF") { std::cerr << "Not an OFF file: " << filepath << "\n"; return false; }

    int numV, numF, numE;
    f >> numV >> numF >> numE;

    out.vertices.resize(numV);
    for (int i = 0; i < numV; ++i)
        f >> out.vertices[i][0] >> out.vertices[i][1] >> out.vertices[i][2];

    out.faces.clear();
    out.faces.reserve(numF);
    for (int i = 0; i < numF; ++i) {
        int n; f >> n;
        std::vector<int> poly(n);
        for (int j = 0; j < n; ++j) f >> poly[j];
        // fan triangulation — handles quads and ngons
        for (int j = 1; j + 1 < n; ++j)
            out.faces.push_back({ poly[0], poly[j], poly[j + 1] });
    }
    return true;
}

// ============================================================
// ============================================================
// SDF Grid
// ============================================================

// Which surface feature of a triangle is closest to the query point.
// Determines which pseudo-normal to use for inside/outside sign.
enum class Feature { FACE, EDGE_AB, EDGE_AC, EDGE_BC, VERTEX_A, VERTEX_B, VERTEX_C };

struct ClosestResult { Eigen::Vector3d point; Feature feature; };

// Ericson "Real-Time Collision Detection" §5.1.5, extended to return feature type.
static ClosestResult closestPointOnTriangle(
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

class SDFGrid {
public:
    Eigen::Vector3d origin;
    double          cellSize   = 0.0;
    Eigen::Vector3i dimensions = Eigen::Vector3i::Zero();
    std::vector<float> distances;

    void generateFromMesh(const MeshData& mesh, double resolution) {
        // AABB
        Eigen::Vector3d bboxMin( 1e30,  1e30,  1e30);
        Eigen::Vector3d bboxMax(-1e30, -1e30, -1e30);
        for (const auto& v : mesh.vertices) {
            bboxMin = bboxMin.cwiseMin(Eigen::Vector3d(v[0], v[1], v[2]));
            bboxMax = bboxMax.cwiseMax(Eigen::Vector3d(v[0], v[1], v[2]));
        }

        const double buffer = resolution * 5.0;
        origin = bboxMin.array() - buffer;
        const Eigen::Vector3d span = (bboxMax.array() + buffer) - origin.array();

        cellSize   = resolution;
        dimensions = Eigen::Vector3i(
            static_cast<int>(std::ceil(span.x() / cellSize)),
            static_cast<int>(std::ceil(span.y() / cellSize)),
            static_cast<int>(std::ceil(span.z() / cellSize))
        );

        const int total = dimensions.x() * dimensions.y() * dimensions.z();
        distances.resize(total);

        // Precompute triangles with per-feature pseudo-normals (Bærentzen & Aanæs 2005).
        // Using the face normal alone is wrong at edges/vertices shared between triangles:
        // whichever adjacent face happens to be nearest first gives an arbitrary sign.
        struct Tri {
            Eigen::Vector3d A, B, C;
            int iA, iB, iC;
            Eigen::Vector3d faceNormal;
            Eigen::Vector3d edgeNormalAB, edgeNormalAC, edgeNormalBC; // fallback = faceNormal
        };

        std::vector<Tri> tris;
        tris.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces) {
            Tri tri;
            tri.iA = f[0]; tri.iB = f[1]; tri.iC = f[2];
            tri.A = Eigen::Vector3d(mesh.vertices[f[0]][0], mesh.vertices[f[0]][1], mesh.vertices[f[0]][2]);
            tri.B = Eigen::Vector3d(mesh.vertices[f[1]][0], mesh.vertices[f[1]][1], mesh.vertices[f[1]][2]);
            tri.C = Eigen::Vector3d(mesh.vertices[f[2]][0], mesh.vertices[f[2]][1], mesh.vertices[f[2]][2]);
            Eigen::Vector3d n = (tri.B - tri.A).cross(tri.C - tri.A);
            const double len = n.norm();
            if (len > 1e-12) n /= len;
            tri.faceNormal = n;
            tri.edgeNormalAB = tri.edgeNormalAC = tri.edgeNormalBC = n;
            tris.push_back(tri);
        }

        // Angle-weighted vertex normals: each incident face weighted by its interior angle
        // at that vertex so skinny triangles don't dominate.
        const int numVerts = static_cast<int>(mesh.vertices.size());
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
            if (len < 1e-12) continue; // 180° fold — keep face-normal fallback
            const Eigen::Vector3d edgeN = sum / len;
            for (int t : ids) {
                Tri& tri = tris[t];
                const int a = edge.first, b = edge.second;
                if      (std::min(tri.iA, tri.iB) == a && std::max(tri.iA, tri.iB) == b) tri.edgeNormalAB = edgeN;
                else if (std::min(tri.iA, tri.iC) == a && std::max(tri.iA, tri.iC) == b) tri.edgeNormalAC = edgeN;
                else                                                                        tri.edgeNormalBC = edgeN;
            }
        }

        // Brute-force: find nearest triangle feature per voxel, sign with its pseudo-normal.
        for (int x = 0; x < dimensions.x(); ++x) {
            for (int y = 0; y < dimensions.y(); ++y) {
                for (int z = 0; z < dimensions.z(); ++z) {
                    const Eigen::Vector3d P(
                        origin.x() + x * cellSize,
                        origin.y() + y * cellSize,
                        origin.z() + z * cellSize
                    );

                    double minDist2 = 1e30;
                    double sign     = 1.0;

                    for (const auto& tri : tris) {
                        const auto [closest, feature] = closestPointOnTriangle(P, tri.A, tri.B, tri.C);
                        const double dist2 = (P - closest).squaredNorm();
                        if (dist2 < minDist2) {
                            minDist2 = dist2;
                            Eigen::Vector3d pseudoNormal;
                            switch (feature) {
                                case Feature::FACE:     pseudoNormal = tri.faceNormal;      break;
                                case Feature::EDGE_AB:  pseudoNormal = tri.edgeNormalAB;    break;
                                case Feature::EDGE_AC:  pseudoNormal = tri.edgeNormalAC;    break;
                                case Feature::EDGE_BC:  pseudoNormal = tri.edgeNormalBC;    break;
                                case Feature::VERTEX_A: pseudoNormal = vertNormals[tri.iA]; break;
                                case Feature::VERTEX_B: pseudoNormal = vertNormals[tri.iB]; break;
                                case Feature::VERTEX_C: pseudoNormal = vertNormals[tri.iC]; break;
                            }
                            sign = ((P - closest).dot(pseudoNormal) >= 0.0) ? 1.0 : -1.0;
                        }
                    }

                    distances[index(x, y, z)] = static_cast<float>(sign * std::sqrt(minDist2));
                }
            }
        }

        std::cout << "SDF computed: "
                  << dimensions.x() << "x" << dimensions.y() << "x" << dimensions.z()
                  << " (" << total << " voxels, " << tris.size() << " triangles)\n";
    }

    int index(int x, int y, int z) const {
        return x * dimensions.y() * dimensions.z() + y * dimensions.z() + z;
    }
};

// ============================================================
// Global state
// ============================================================
static std::vector<std::string> availableOffFiles;
static int                      selectedFileIndex = -1;
static SDFGrid                  sdf;

// ============================================================
// File scanner
// ============================================================
static void scanAssetsFolder() {
    availableOffFiles.clear();
    const fs::path path = fs::path(RESOURCES_PATH) / "assets";

    if (!fs::exists(path)) {
        fs::create_directories(path);
        std::cout << "Created assets folder at: " << path << "\n"
                  << "Put .off files inside and hit Rescan.\n";
        return;
    }

    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".off")
            availableOffFiles.push_back(entry.path().string());
    }
    std::sort(availableOffFiles.begin(), availableOffFiles.end());
}

// ============================================================
// Load + compute SDF + visualize
// ============================================================
static void loadMeshAndComputeSDF(const std::string& filepath) {
    polyscope::removeAllStructures();

    MeshData meshData;
    if (!loadOFF(filepath, meshData)) return;

    const std::string meshName = fs::path(filepath).stem().string();
    polyscope::registerSurfaceMesh(meshName, meshData.vertices, meshData.faces);

    const double pbfCellSize = 0.1;  // should match PBF particle diameter
    sdf.generateFromMesh(meshData, pbfCellSize);

    // Visualize every voxel center
    std::vector<std::array<double, 3>> pts;
    std::vector<float>                 dists;
    for (int x = 0; x < sdf.dimensions.x(); ++x) {
        for (int y = 0; y < sdf.dimensions.y(); ++y) {
            for (int z = 0; z < sdf.dimensions.z(); ++z) {
                pts.push_back({
                    sdf.origin.x() + x * sdf.cellSize,
                    sdf.origin.y() + y * sdf.cellSize,
                    sdf.origin.z() + z * sdf.cellSize
                });
                dists.push_back(sdf.distances[sdf.index(x, y, z)]);
            }
        }
    }

    if (!pts.empty()) {
        // Symmetric range around 0: blue = inside (negative), red = outside (positive)
        float maxAbs = 0.f;
        for (float d : dists) maxAbs = std::max(maxAbs, std::abs(d));

        auto* pc = polyscope::registerPointCloud("SDF Grid", pts);
        auto* q  = pc->addScalarQuantity("distance", dists);
        q->setColorMap("hsv");
        q->setMapRange({-maxAbs, maxAbs});
        q->setEnabled(true);
        pc->setPointRadius(0.003);
    }
}

// ============================================================
// ImGui callback
// ============================================================
static void myPolyscopeCallback() {
    ImGui::Text("SDF FSI Test Environment");
    ImGui::Separator();

    if (availableOffFiles.empty()) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "No .off files found in 'assets/' folder.");
        if (ImGui::Button("Rescan")) scanAssetsFolder();
        return;
    }

    const char* preview = (selectedFileIndex >= 0)
        ? availableOffFiles[selectedFileIndex].c_str()
        : "Select a mesh...";

    if (ImGui::BeginCombo("Assets", preview)) {
        for (int i = 0; i < (int)availableOffFiles.size(); ++i) {
            const bool selected = (selectedFileIndex == i);
            if (ImGui::Selectable(availableOffFiles[i].c_str(), selected)) {
                selectedFileIndex = i;
                loadMeshAndComputeSDF(availableOffFiles[i]);
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    if (ImGui::Button("Rescan Folder")) {
        selectedFileIndex = -1;
        scanAssetsFolder();
    }
}

// ============================================================
// Entry point
// ============================================================
int main() {
    polyscope::init();
    polyscope::options::programName = "Realtime FSI Engine - SDF Test";

    scanAssetsFolder();
    polyscope::state::userCallback = myPolyscopeCallback;
    polyscope::show();

    return 0;
}
