#pragma once

#include "core/Types.h"
#include "deformation/IncrementalCholesky.h"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <vector>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

class LaplacianDeformation {
public:
    LaplacianDeformation(MyMesh& mesh);
    ~LaplacianDeformation();

    void initialize();

    // Constraints
    void addAnchor(int vertexIndex);
    void setControlPoint(int vertexIndex, const EVec3& targetPosition);

    // Dynamic handle set (multiple control points) for fluid-driven deformation.
    // The constrained SET (anchors + handle indices) determines the factorization;
    // call precomputeSystem() only when the set changes. Targets-only changes need
    // just another solve(). Replaces the previous handle set wholesale.
    void setControlPoints(const std::vector<int>& indices, const std::vector<EVec3>& targets);
    void clearControlPoints();
    const std::vector<int>& getControlIndices() const { return _controlIndices; }

    void clearConstraints();
    void clearControlPoint();

    // Execution
    void precomputeSystem();
    void solve();

    // ---- Incremental Cholesky path (hard constraints, Eq.14 + Alg.3) ----
    // When enabled, precomputeSystem()/solve() use the persistent LDL^T factor
    // (IncrementalCholesky) instead of refactoring the soft normal equations:
    //   * first precompute  -> analyze (etree, once) + full factorize
    //   * set changed       -> recompute only columns on etree paths (Alg.3)
    //   * set unchanged      -> caller skips precompute; solve() reuses the factor
    // The constraints become HARD (vertices hit their targets exactly), so the
    // deformation looks slightly different from the soft path. Off by default.
    void setUseIncremental(bool on) { _useIncremental = on; }
    bool isUsingIncremental() const { return _useIncremental; }

    // Verification: max |x_incremental - x_eigen| for the current x-axis system,
    // where x_eigen comes from a fresh Eigen SimplicialLDLT of the same A^B.
    // Should be ~0. Returns -1 if the incremental path is not ready.
    double verifyIncremental();

    // Accessors for rendering gizmos
    const std::vector<int>& getAnchorIndices() const { return _anchorIndices; }
    const std::vector<EVec3>& getAnchorPositions() const { return _anchorPositions; }
    int getControlIndex() const { return _controlIndex; }
    EVec3 getControlTarget() const { return _controlTarget; }
    const MyMesh& getMesh() const { return _mesh; }

private:
    MyMesh& _mesh;

    ESparseMatrix _laplacian;
    ESparseMatrix _AtA;  // Precomputed A^T * A
    ESparseMatrix _At;   // Precomputed A^T (for right-hand side)
    ECholmodSolver _solver;

    // Per-axis differential coordinates: delta = L * V
    EVecX _deltaX, _deltaY, _deltaZ;

    OpenMesh::EPropHandleT<double> _cotangentWeights;

    // Constraint data
    std::vector<int> _anchorIndices;
    std::vector<EVec3> _anchorPositions;
    int _controlIndex = -1;
    EVec3 _controlTarget = EVec3::Zero();

    // Dynamic handle set (fluid-driven). Rows are appended after anchors and the
    // legacy single control point, in this exact order, in both precompute and solve.
    std::vector<int>   _controlIndices;
    std::vector<EVec3> _controlTargets;

    void computeCotangentWeights();
    void buildLaplacianMatrix();
    void computeDifferentialCoordinates();

    // ---- Incremental Cholesky state ----
    bool _useIncremental = false;
    bool _incReady       = false;

    IncrementalCholesky _inc;
    ESparseMatrix _A;                 // normal matrix L^T L (+ tiny diagonal reg), symmetric
    EVecX _bX, _bY, _bZ;              // gradient RHS L^T delta per axis (= A * rest)
    std::vector<char> _constrainedMask;   // size n: 1 if vertex currently constrained (ORIGINAL order)
    std::vector<int>  _prevConstrained;   // sorted constrained set from the last precompute

    // AMD fill-reducing reordering for the incremental factor. _A is FIXED for the
    // whole sim, so it is permuted ONCE into _Aperm = P A P^T and the core
    // IncrementalCholesky only ever sees AMD order; masks/changed-cols/RHS are
    // mapped at the boundary. Natural mesh order gives a chain-like elimination
    // tree (long Alg.3 paths -> I1 ~ all columns); AMD makes it bushy so I1 stays
    // small. _perm is new->old (_perm[newIdx]=origVertex), _invPerm its inverse.
    ESparseMatrix     _Aperm;
    std::vector<int>  _perm;
    std::vector<int>  _invPerm;

    // Build _A and the per-axis RHS once (called from initialize()).
    void buildNormalSystem();
    // Sorted union of all currently constrained vertices (anchors + control pts).
    std::vector<int> currentConstrainedSorted() const;
    // Per-axis target position for every constrained vertex (others left zero).
    void gatherConstraintTargets(EVecX& tx, EVecX& ty, EVecX& tz) const;
    // Solve one axis through the incremental factor: rhs = b - A(:,C) c, then
    // pin constrained rows to their target, then back-substitute.
    void solveAxisIncremental(const EVecX& b, const EVecX& targets, EVecX& outX) const;
};