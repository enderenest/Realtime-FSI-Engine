#include "deformation/LaplacianDeformation.h"
#include <algorithm>
#include <iterator>
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
    // Note: the n x n normal system (_A, _bX/Y/Z) is built lazily by the
    // incremental path on first use (buildNormalSystem), so the default soft
    // solver pays nothing for it.
}

// Build the n x n normal matrix A = L^T L (symmetric, SPD once >= 1 vertex is
// constrained) and the per-axis gradient RHS b = L^T delta. A tiny diagonal
// regularization keeps the detached system safely positive-definite. This is
// the persistent system the incremental factor lives on (Step 1 / Step 2).
void LaplacianDeformation::buildNormalSystem() {
    const int n = (int)_mesh.n_vertices();
    if (n == 0) return;

    ESparseMatrix Lt = _laplacian.transpose();
    _A = (Lt * _laplacian).pruned();

    constexpr double kReg = 1e-8;     // regularize the constant nullspace of L^T L
    for (int k = 0; k < n; ++k) _A.coeffRef(k, k) += kReg;
    _A.makeCompressed();

    _bX = Lt * _deltaX;
    _bY = Lt * _deltaY;
    _bZ = Lt * _deltaZ;

    // ---- AMD fill-reducing reordering (computed ONCE; _A never changes) ------
    // Eigen's AMDOrdering writes P.indices()[newIdx] = originalVertex, i.e. the
    // reordered matrix is A(perm, perm). We build _Aperm = P A P^T explicitly so
    // the core IncrementalCholesky factorizes the bushy-etree ordering instead of
    // the natural chain. Everything that indexes the factor (constraint mask,
    // changed columns, solve RHS/solution) is mapped through _perm/_invPerm.
    {
        Eigen::AMDOrdering<int> ordering;
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
        ordering(_A.selfadjointView<Eigen::Upper>(), P);   // P.indices()[new] = old

        _perm.assign(n, 0);
        _invPerm.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            _perm[i] = P.indices()(i);
            _invPerm[_perm[i]] = i;
        }

        // _Aperm(a,b) = _A(perm[a], perm[b]); place original (i,j) at (invPerm[i], invPerm[j]).
        std::vector<ETriplet> trips;
        trips.reserve(_A.nonZeros());
        for (int k = 0; k < _A.outerSize(); ++k)
            for (ESparseMatrix::InnerIterator it(_A, k); it; ++it)
                trips.emplace_back(_invPerm[(int)it.row()], _invPerm[(int)it.col()], it.value());
        _Aperm.resize(n, n);
        _Aperm.setFromTriplets(trips.begin(), trips.end());
        _Aperm.makeCompressed();
    }

    _incReady = false;
    _prevConstrained.clear();
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

    // ---- Incremental Cholesky path (hard constraints, Eq.14 + Alg.3) --------
    if (_useIncremental) {
        const int n = (int)_mesh.n_vertices();
        std::vector<int> cur = currentConstrainedSorted();
        if (cur.empty()) {
            std::cerr << "[Laplacian/inc] Cannot precompute: no constraints.\n";
            return;
        }

        if (!_incReady) {
            // First time: build the persistent normal system (incl. AMD order),
            // then the elimination tree (once) on the PERMUTED matrix.
            buildNormalSystem();
            _inc.analyze(_Aperm);
            _incReady = true;
        }

        // Constraint mask in ORIGINAL order (used by the solve, which assembles
        // its RHS from _A) and in PERMUTED order (used by the factor).
        _constrainedMask.assign(n, 0);
        std::vector<char> maskPerm(n, 0);
        for (int v : cur) { _constrainedMask[v] = 1; maskPerm[_invPerm[v]] = 1; }
        _inc.setConstrained(maskPerm);

        if (_prevConstrained.empty()) {
            // No prior state to diff against: full factorization of A^B.
            _inc.factorizeFull(_Aperm);
        } else {
            // Set changed: I0 = entering ∪ leaving (symmetric difference), plus
            // their neighbours (the columns of A^B whose entries actually moved),
            // mapped to permuted indices. Algorithm 3 recomputes only the etree
            // paths from I0.
            std::vector<int> changed;
            std::set_symmetric_difference(cur.begin(), cur.end(),
                                          _prevConstrained.begin(), _prevConstrained.end(),
                                          std::back_inserter(changed));

            std::vector<int> I0;
            I0.reserve(changed.size() * 8);
            std::vector<char> seen(n, 0);   // indexed by PERMUTED column
            auto push = [&](int origV) {
                const int p = _invPerm[origV];
                if (!seen[p]) { seen[p] = 1; I0.push_back(p); }
            };
            for (int v : changed) {
                push(v);
                for (ESparseMatrix::InnerIterator it(_A, v); it; ++it) push((int)it.row());
            }

            _inc.updateColumns(_Aperm, I0);
        }

        _prevConstrained = std::move(cur);
        return;
    }

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
    // ---- Incremental Cholesky path: three RHS share the persistent factor ----
    if (_useIncremental) {
        if (!_incReady) { std::cerr << "[Laplacian/inc] solve before precompute.\n"; return; }
        const int n = (int)_mesh.n_vertices();

        EVecX tx(n), ty(n), tz(n);
        gatherConstraintTargets(tx, ty, tz);

        EVecX x(n), y(n), z(n);
        solveAxisIncremental(_bX, tx, x);
        solveAxisIncremental(_bY, ty, y);
        solveAxisIncremental(_bZ, tz, z);

        for (int i = 0; i < n; ++i)
            _mesh.set_point(OpenMesh::VertexHandle(i), MyMesh::Point(x(i), y(i), z(i)));
        return;
    }

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

// ============================================================
// Incremental Cholesky helpers
// ============================================================

std::vector<int> LaplacianDeformation::currentConstrainedSorted() const {
    std::vector<int> c = _anchorIndices;
    if (_controlIndex >= 0) c.push_back(_controlIndex);
    c.insert(c.end(), _controlIndices.begin(), _controlIndices.end());
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

void LaplacianDeformation::gatherConstraintTargets(EVecX& tx, EVecX& ty, EVecX& tz) const {
    const int n = (int)_mesh.n_vertices();
    tx.setZero(n); ty.setZero(n); tz.setZero(n);
    auto set = [&](int v, const EVec3& p) { tx(v) = p.x(); ty(v) = p.y(); tz(v) = p.z(); };

    for (size_t i = 0; i < _anchorIndices.size(); ++i) set(_anchorIndices[i], _anchorPositions[i]);
    if (_controlIndex >= 0) set(_controlIndex, _controlTarget);
    for (size_t i = 0; i < _controlIndices.size(); ++i) set(_controlIndices[i], _controlTargets[i]);
}

void LaplacianDeformation::solveAxisIncremental(const EVecX& b, const EVecX& targets,
                                                EVecX& outX) const {
    const int n = (int)_mesh.n_vertices();

    // rhs starts at the gradient b = L^T delta. Move every constrained column's
    // contribution to the RHS (free rows i: rhs_i -= A_ik * c_k), exactly the
    // A_IC c_C correction that the detached matrix A^B no longer carries.
    EVecX rhs = b;
    for (int k = 0; k < n; ++k) {
        if (!_constrainedMask[k]) continue;
        const double ck = targets(k);
        for (ESparseMatrix::InnerIterator it(_A, k); it; ++it)
            rhs(it.row()) -= it.value() * ck;
    }
    // Pin constrained rows to their targets (A^B has identity rows there, so the
    // solve returns x_k = rhs_k). Overwrites the self-subtraction above.
    for (int k = 0; k < n; ++k)
        if (_constrainedMask[k]) rhs(k) = targets(k);

    // The factor lives in AMD-permuted space: gather rhs -> solve -> scatter back.
    EVecX rhsP(n);
    for (int i = 0; i < n; ++i) rhsP(i) = rhs(_perm[i]);    // rhsP[new] = rhs[old]
    _inc.solveInPlace(rhsP);
    outX.resize(n);
    for (int i = 0; i < n; ++i) outX(_perm[i]) = rhsP(i);   // outX[old] = sol[new]
}

double LaplacianDeformation::verifyIncremental() {
    if (!_useIncremental || !_incReady) return -1.0;
    const int n = (int)_mesh.n_vertices();

    // Build A^B explicitly (Eq.14) and factor it with a fresh Eigen LDLT.
    ESparseMatrix AB(n, n);
    {
        std::vector<ETriplet> trips;
        trips.reserve(_A.nonZeros());
        for (int k = 0; k < _A.outerSize(); ++k)
            for (ESparseMatrix::InnerIterator it(_A, k); it; ++it) {
                const int i = (int)it.row(), j = (int)it.col();
                if (_constrainedMask[i] || _constrainedMask[j]) {
                    if (i == j) trips.emplace_back(i, j, 1.0);   // identity diagonal
                } else {
                    trips.emplace_back(i, j, it.value());
                }
            }
        AB.setFromTriplets(trips.begin(), trips.end());
    }

    EVecX tx(n), ty(n), tz(n);
    gatherConstraintTargets(tx, ty, tz);

    EVecX rhs = _bX;
    for (int k = 0; k < n; ++k) {
        if (!_constrainedMask[k]) continue;
        for (ESparseMatrix::InnerIterator it(_A, k); it; ++it) rhs(it.row()) -= it.value() * tx(k);
    }
    for (int k = 0; k < n; ++k) if (_constrainedMask[k]) rhs(k) = tx(k);

    ELDLTSolver eig;
    eig.compute(AB);
    if (eig.info() != Eigen::Success) { std::cerr << "[inc verify] Eigen factor failed\n"; return -2.0; }
    const EVecX xEig = eig.solve(rhs);

    // _inc factorized the AMD-permuted A^B, so round-trip the RHS/solution.
    EVecX rhsP(n);
    for (int i = 0; i < n; ++i) rhsP(i) = rhs(_perm[i]);
    _inc.solveInPlace(rhsP);
    EVecX xInc(n);
    for (int i = 0; i < n; ++i) xInc(_perm[i]) = rhsP(i);

    const double maxDiff = (xInc - xEig).cwiseAbs().maxCoeff();
    std::cout << "[inc verify] max |x_inc - x_eigen| = " << maxDiff << "\n";
    return maxDiff;
}