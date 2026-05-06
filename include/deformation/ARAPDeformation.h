#pragma once

#include "core/Types.h"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <vector>

// libigl headers
#include <Eigen/Core>
#include <igl/arap.h>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

class ARAPDeformation {
public:
    ARAPDeformation(MyMesh& mesh);
    ~ARAPDeformation();

    void initialize();

    // Constraints (Identical API to Laplacian)
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

    // --- libigl Core Data ---
    Eigen::MatrixXd _V_initial; // Original resting vertices (N x 3)
    Eigen::MatrixXd _V_deformed; // Current moving vertices (N x 3)
    Eigen::MatrixXi _F;         // Mesh faces (M x 3)

    // libigl's internal ARAP solver state (replaces your _laplacian and _solver)
    igl::ARAPData _arapData;

    // Constraint data (Used to build libigl's 'b' and 'bc' vectors during precompute)
    std::vector<int> _anchorIndices;
    std::vector<EVec3> _anchorPositions;
    int _controlIndex = -1;
    EVec3 _controlTarget = EVec3::Zero();

    // Internal Helpers
    void convertMeshToEigen();
    void updateOpenMeshFromEigen();
};