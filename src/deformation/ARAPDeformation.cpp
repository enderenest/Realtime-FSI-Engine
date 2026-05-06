#include "deformation/ARAPDeformation.h"
#include <iostream>

ARAPDeformation::ARAPDeformation(MyMesh& mesh) : _mesh(mesh) {}

ARAPDeformation::~ARAPDeformation() {}

void ARAPDeformation::initialize() {
    convertMeshToEigen();
    _V_deformed = _V_initial; // warm start begins from rest pose
}

void ARAPDeformation::convertMeshToEigen() {
    int numVertices = _mesh.n_vertices();
    int numFaces = _mesh.n_faces();

    // --- Build V: N x 3 vertex matrix ---
    _V_initial.resize(numVertices, 3);
    for (int i = 0; i < numVertices; ++i) {
        auto p = _mesh.point(OpenMesh::VertexHandle(i));
        _V_initial(i, 0) = p[0];
        _V_initial(i, 1) = p[1];
        _V_initial(i, 2) = p[2];
    }

    // --- Build F: triangulated face matrix ---
    // PolyMesh faces may be quads or n-gons; fan-triangulate each face.
    std::vector<Eigen::Vector3i> triangles;
    triangles.reserve(numFaces * 2); // rough estimate

    for (auto fh : _mesh.faces()) {
        // Collect vertex indices for this face
        std::vector<int> faceVerts;
        for (auto fvIt = _mesh.cfv_iter(fh); fvIt.is_valid(); ++fvIt) {
            faceVerts.push_back(fvIt->idx());
        }

        // Fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
        for (size_t j = 1; j + 1 < faceVerts.size(); ++j) {
            triangles.push_back(Eigen::Vector3i(faceVerts[0], faceVerts[j], faceVerts[j + 1]));
        }
    }

    _F.resize(triangles.size(), 3);
    for (int i = 0; i < (int)triangles.size(); ++i) {
        _F.row(i) = triangles[i];
    }
}

void ARAPDeformation::updateOpenMeshFromEigen() {
    int numVertices = _V_deformed.rows();
    for (int i = 0; i < numVertices; ++i) {
        _mesh.set_point(OpenMesh::VertexHandle(i),
            MyMesh::Point(_V_deformed(i, 0), _V_deformed(i, 1), _V_deformed(i, 2)));
    }
}

void ARAPDeformation::clearConstraints() {
    _anchorIndices.clear();
    _anchorPositions.clear();
    _controlIndex = -1;
    _controlTarget = EVec3::Zero();
}

void ARAPDeformation::clearControlPoint() {
    _controlIndex = -1;
    _controlTarget = EVec3::Zero();
}

void ARAPDeformation::addAnchor(int vertexIndex) {
    // Reject duplicates
    for (int idx : _anchorIndices)
        if (idx == vertexIndex) return;

    auto point = _mesh.point(OpenMesh::VertexHandle(vertexIndex));
    _anchorIndices.push_back(vertexIndex);
    _anchorPositions.push_back(EVec3(point[0], point[1], point[2]));
}

void ARAPDeformation::setControlPoint(int vertexIndex, const EVec3& targetPosition) {
    _controlIndex = vertexIndex;
    _controlTarget = targetPosition;
}

void ARAPDeformation::precomputeSystem() {
    // Build b: the list of ALL constrained vertex IDs (anchors + control point)
    int constraintCount = _anchorIndices.size() + (_controlIndex >= 0 ? 1 : 0);

    // ARAP mathematically requires at least 1 constraint!
    if (constraintCount == 0) {
        std::cerr << "[ARAP] Cannot precompute: No constraints provided!\n";
        return;
    }

    Eigen::VectorXi b(constraintCount);

    for (int i = 0; i < (int)_anchorIndices.size(); ++i) {
        b(i) = _anchorIndices[i];
    }
    if (_controlIndex >= 0) {
        b(constraintCount - 1) = _controlIndex;
    }

    // Run libigl's ARAP precomputation (builds K, CSM, factorizes the system)
    _arapData.max_iter = 10;
    bool ok = igl::arap_precomputation(_V_initial, _F, 3, b, _arapData);
    if (!ok) {
        std::cerr << "ARAP precomputation failed!\n";
    }
}

void ARAPDeformation::solve() {
    // Build bc: the target positions for each constrained vertex (same order as b)
    int constraintCount = _anchorIndices.size() + (_controlIndex >= 0 ? 1 : 0);
    Eigen::MatrixXd bc(constraintCount, 3);

    // Anchors stay at their original positions
    for (int i = 0; i < (int)_anchorIndices.size(); ++i) {
        bc(i, 0) = _anchorPositions[i].x();
        bc(i, 1) = _anchorPositions[i].y();
        bc(i, 2) = _anchorPositions[i].z();
    }

    // Control point goes to the user-specified target
    if (_controlIndex >= 0) {
        bc(constraintCount - 1, 0) = _controlTarget.x();
        bc(constraintCount - 1, 1) = _controlTarget.y();
        bc(constraintCount - 1, 2) = _controlTarget.z();
    }

    // Solve ARAP: _V_deformed is both the warm-start input and the output
    igl::arap_solve(bc, _arapData, _V_deformed);

    // Write deformed positions back to OpenMesh
    updateOpenMeshFromEigen();
}
