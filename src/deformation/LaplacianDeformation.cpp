#include "deformation/LaplacianDeformation.h"
#include <iostream>

LaplacianDeformation::LaplacianDeformation(MyMesh& mesh) : _mesh(mesh) {
    _mesh.add_property(_cotangentWeights, "cotangent_weights");
}

LaplacianDeformation::~LaplacianDeformation() {
    _mesh.remove_property(_cotangentWeights);
}

void LaplacianDeformation::initialize() {
    //std::cout << "Initializing LaplacianDeformation System...\n";
    computeCotangentWeights();
    buildLaplacianMatrix();
    computeDifferentialCoordinates();
}

void LaplacianDeformation::computeCotangentWeights() {
    //std::cout << "Computing Cotangent Weights...\n";

    int numEdges = _mesh.n_edges();

    for (int i = 0; i < numEdges; ++i) {
        OpenMesh::EdgeHandle eh(i);
        OpenMesh::HalfedgeHandle halfedge0 = _mesh.halfedge_handle(eh, 0);
        OpenMesh::HalfedgeHandle halfedge1 = _mesh.halfedge_handle(eh, 1);

        double cotangentSum = 0.0;

        // Cotangent contribution from the first adjacent face
        if (!_mesh.is_boundary(halfedge0)) {
            auto edgeStart    = _mesh.point(_mesh.from_vertex_handle(halfedge0));
            auto edgeEnd      = _mesh.point(_mesh.to_vertex_handle(halfedge0));
            auto oppositeVert = _mesh.point(_mesh.to_vertex_handle(_mesh.next_halfedge_handle(halfedge0)));

            EVec3 toStart(edgeStart[0] - oppositeVert[0], edgeStart[1] - oppositeVert[1], edgeStart[2] - oppositeVert[2]);
            EVec3 toEnd  (edgeEnd[0]   - oppositeVert[0], edgeEnd[1]   - oppositeVert[1], edgeEnd[2]   - oppositeVert[2]);

            double cosAngle   = toStart.dot(toEnd);
            double sinAngle   = toStart.cross(toEnd).norm();
            if (sinAngle > 1e-6)
                cotangentSum += cosAngle / sinAngle;
        }

        // Cotangent contribution from the second adjacent face
        if (!_mesh.is_boundary(halfedge1)) {
            auto edgeStart    = _mesh.point(_mesh.from_vertex_handle(halfedge1));
            auto edgeEnd      = _mesh.point(_mesh.to_vertex_handle(halfedge1));
            auto oppositeVert = _mesh.point(_mesh.to_vertex_handle(_mesh.next_halfedge_handle(halfedge1)));

            EVec3 toStart(edgeStart[0] - oppositeVert[0], edgeStart[1] - oppositeVert[1], edgeStart[2] - oppositeVert[2]);
            EVec3 toEnd  (edgeEnd[0]   - oppositeVert[0], edgeEnd[1]   - oppositeVert[1], edgeEnd[2]   - oppositeVert[2]);

            double cosAngle   = toStart.dot(toEnd);
            double sinAngle   = toStart.cross(toEnd).norm();
            if (sinAngle > 1e-6)
                cotangentSum += cosAngle / sinAngle;
        }

        double finalWeight = std::max(0.5 * cotangentSum, 1e-4);
        _mesh.property(_cotangentWeights, eh) = finalWeight;
    }

    //std::cout << "Cotangent weights computed successfully.\n";
}

void LaplacianDeformation::buildLaplacianMatrix() {
    //std::cout << "Building Laplacian Matrix...\n";

    int vertexCount = _mesh.n_vertices();
    std::vector<ETriplet> triplets;
    triplets.reserve(vertexCount * 7);

    for (int i = 0; i < vertexCount; ++i) {
        OpenMesh::VertexHandle vh(i);
        int row = i;

        // Sum cotangent weights around this vertex for normalization
        double weightSum = 0.0;
        for (auto neighborIt = _mesh.cvv_iter(vh); neighborIt.is_valid(); ++neighborIt) {
            OpenMesh::EdgeHandle edge = _mesh.edge_handle(_mesh.find_halfedge(vh, *neighborIt));
            weightSum += _mesh.property(_cotangentWeights, edge);
        }

        // Collect triplets for this vertex locally
        std::vector<ETriplet> localTriplets;

        // Diagonal: L_ii = 1.0
        localTriplets.push_back(ETriplet(row, row, 1.0));

        // Off-diagonal: L_ij = -w_ij / sum(w) so that each row sums to zero
        for (auto neighborIt = _mesh.cvv_iter(vh); neighborIt.is_valid(); ++neighborIt) {
            int col = neighborIt->idx();
            OpenMesh::EdgeHandle edge = _mesh.edge_handle(_mesh.find_halfedge(vh, *neighborIt));
            double weight = _mesh.property(_cotangentWeights, edge);
            localTriplets.push_back(ETriplet(row, col, -weight / weightSum));
        }

        {
            triplets.insert(triplets.end(), localTriplets.begin(), localTriplets.end());
        }
    }

    _laplacian.resize(vertexCount, vertexCount);
    _laplacian.setFromTriplets(triplets.begin(), triplets.end());

    //std::cout << "Laplacian Matrix built: " << vertexCount << "x" << vertexCount
    //          << " with " << _laplacian.nonZeros() << " non-zeros.\n";
}

void LaplacianDeformation::computeDifferentialCoordinates() {
    //std::cout << "Computing Differential Coordinates...\n";

    int vertexCount = _mesh.n_vertices();

    // Extract mesh vertex positions into per-axis vectors
    EVecX positionsX(vertexCount), positionsY(vertexCount), positionsZ(vertexCount);

    for (int i = 0; i < vertexCount; ++i) {
        auto point = _mesh.point(OpenMesh::VertexHandle(i));
        positionsX(i) = point[0];
        positionsY(i) = point[1];
        positionsZ(i) = point[2];
    }

    // delta = L * V
    _deltaX = _laplacian * positionsX;
    _deltaY = _laplacian * positionsY;
    _deltaZ = _laplacian * positionsZ;

    //std::cout << "Differential coordinates computed for " << vertexCount << " vertices.\n";
}

void LaplacianDeformation::clearConstraints() {
    _anchorIndices.clear();
    _anchorPositions.clear();
    _controlIndex = -1;
    _controlTarget = EVec3::Zero();
    _controlIndices.clear();
    _controlTargets.clear();
}

void LaplacianDeformation::clearControlPoint() {
    _controlIndex = -1;
    _controlTarget = EVec3::Zero();
}

void LaplacianDeformation::addAnchor(int vertexIndex) {
    // Reject duplicates
    for (int idx : _anchorIndices)
        if (idx == vertexIndex) return;

    auto point = _mesh.point(OpenMesh::VertexHandle(vertexIndex));
    //std::cout << "Anchor added: vertex " << vertexIndex
    //          << " at (" << point[0] << ", " << point[1] << ", " << point[2] << ")\n";
    _anchorIndices.push_back(vertexIndex);
    _anchorPositions.push_back(EVec3(point[0], point[1], point[2]));
}

void LaplacianDeformation::setControlPoint(int vertexIndex, const EVec3& targetPosition) {
    _controlIndex = vertexIndex;
    _controlTarget = targetPosition;
}

void LaplacianDeformation::setControlPoints(const std::vector<int>& indices,
                                            const std::vector<EVec3>& targets) {
    _controlIndices = indices;
    _controlTargets = targets;
}

void LaplacianDeformation::clearControlPoints() {
    _controlIndices.clear();
    _controlTargets.clear();
}
void LaplacianDeformation::precomputeSystem() {
    //std::cout << "Precomputing solver system...\n";

    int vertexCount = _mesh.n_vertices();
    int constraintCount = _anchorIndices.size() + (_controlIndex >= 0 ? 1 : 0)
                        + (int)_controlIndices.size();
    int totalRows = vertexCount + constraintCount;

    // Laplacian mathematically requires at least 1 constraint!
    if (constraintCount == 0) {
        std::cerr << "[Laplacian] Cannot precompute: No constraints provided!\n";
        return;
    }

    // Build A: stack L on top, then one row per constraint
    std::vector<ETriplet> triplets;

    // Copy L into the top part of A
    for (int k = 0; k < _laplacian.outerSize(); ++k) {
        for (ESparseMatrix::InnerIterator it(_laplacian, k); it; ++it) {
            triplets.push_back(ETriplet(it.row(), it.col(), it.value()));
        }
    }

    // Anchor constraints
    int constraintRow = vertexCount;
    for (int anchorIdx : _anchorIndices) {
        triplets.push_back(ETriplet(constraintRow, anchorIdx, 1.0));
        constraintRow++;
    }

    // Control point constraint (legacy single)
    if (_controlIndex >= 0) {
        triplets.push_back(ETriplet(constraintRow, _controlIndex, 1.0));
        constraintRow++;
    }

    // Dynamic handle constraints (one row each)
    for (int handleIdx : _controlIndices) {
        triplets.push_back(ETriplet(constraintRow, handleIdx, 1.0));
        constraintRow++;
    }

    ESparseMatrix systemMatrix(totalRows, vertexCount);
    systemMatrix.setFromTriplets(triplets.begin(), triplets.end());

    // Precompute A^T and A^T * A, then factorize
    _At = systemMatrix.transpose();
    _AtA = _At * systemMatrix;
    _solver.compute(_AtA);

    if (_solver.info() != Eigen::Success) {
        std::cerr << "Solver factorization failed!\n";
        return;
    }

    //std::cout << "Solver precomputed: " << totalRows << " rows x " << vertexCount << " cols, "
    //          << constraintCount << " constraints.\n";
}
void LaplacianDeformation::solve() {
    int vertexCount = _mesh.n_vertices();
    int totalRows = vertexCount + _anchorIndices.size() + (_controlIndex >= 0 ? 1 : 0)
                  + (int)_controlIndices.size();

    // Build right-hand side b for each axis
    // Top part: differential coordinates (delta)
    // Bottom part: constraint target positions
    EVecX bX(totalRows), bY(totalRows), bZ(totalRows);

    bX.head(vertexCount) = _deltaX;
    bY.head(vertexCount) = _deltaY;
    bZ.head(vertexCount) = _deltaZ;

    // Anchor targets: stored original positions
    int constraintRow = vertexCount;
    for (int i = 0; i < (int)_anchorIndices.size(); ++i) {
        bX(constraintRow) = _anchorPositions[i].x();
        bY(constraintRow) = _anchorPositions[i].y();
        bZ(constraintRow) = _anchorPositions[i].z();
        constraintRow++;
    }

    // Control point target: user-defined position (legacy single)
    if (_controlIndex >= 0) {
        bX(constraintRow) = _controlTarget.x();
        bY(constraintRow) = _controlTarget.y();
        bZ(constraintRow) = _controlTarget.z();
        constraintRow++;
    }

    // Dynamic handle targets (same order as precomputeSystem)
    for (int i = 0; i < (int)_controlIndices.size(); ++i) {
        bX(constraintRow) = _controlTargets[i].x();
        bY(constraintRow) = _controlTargets[i].y();
        bZ(constraintRow) = _controlTargets[i].z();
        constraintRow++;
    }

    // Solve A^T * A * x = A^T * b for each axis
    EVecX solvedX = _solver.solve(_At * bX);
    EVecX solvedY = _solver.solve(_At * bY);
    EVecX solvedZ = _solver.solve(_At * bZ);

    // Write deformed positions back to the mesh
    for (int i = 0; i < vertexCount; ++i) {
        _mesh.set_point(OpenMesh::VertexHandle(i), MyMesh::Point(solvedX(i), solvedY(i), solvedZ(i)));
    }
}