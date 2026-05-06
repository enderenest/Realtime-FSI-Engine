#pragma once

#include "core/Types.h"
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

    void clearConstraints();
    void clearControlPoint();

    // Execution
    void precomputeSystem();
    void solve();

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

    void computeCotangentWeights();
    void buildLaplacianMatrix();
    void computeDifferentialCoordinates();
};