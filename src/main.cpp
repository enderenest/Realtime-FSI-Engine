#include <iostream>
#include <vector>
#include <chrono>

// Core Eigen Headers
#include <Eigen/Core>
#include <Eigen/SparseCore>

// SuiteSparse/CHOLMOD Wrapper Header
// If this line throws an error, SuiteSparse is not linked properly in CMake
#include <Eigen/CholmodSupport>

// Type definitions to keep the code clean
using Scalar = double;
using ESparseMatrix = Eigen::SparseMatrix<Scalar>;
using EVec = Eigen::VectorXd;
using ETriplet = Eigen::Triplet<Scalar>;
using ECholmodSolver = Eigen::CholmodSupernodalLLT<ESparseMatrix>;

int main() {
    std::cout << "--- CHOLMOD Supernodal Solver Test ---" << std::endl;

    // 1. Simulate a mesh dimension (e.g., 1000 vertices)
    const int numVertices = 1000;

    // 2. Prepare the Sparse Matrix (A) and Right-Hand Side (b)
    // We are simulating the Normal Equations: (A^T A) x = A^T b
    // The matrix must be Symmetric and Positive-Definite (SPD).
    ESparseMatrix A(numVertices, numVertices);
    EVec b = EVec::Random(numVertices);
    EVec x(numVertices);

    // 3. Populate the Matrix with "Triplets" (Row, Col, Value)
    // This simulates a 1D Laplacian operator (connections to neighbors)
    std::vector<ETriplet> triplets;
    triplets.reserve(numVertices * 3);

    for (int i = 0; i < numVertices; ++i) {
        // Diagonal entry (Weight of the vertex itself)
        triplets.push_back(ETriplet(i, i, 2.0));

        // Connect to left neighbor
        if (i > 0) {
            triplets.push_back(ETriplet(i, i - 1, -1.0));
        }
        // Connect to right neighbor
        if (i < numVertices - 1) {
            triplets.push_back(ETriplet(i, i + 1, -1.0));
        }
    }

    // Compress the triplets into the Compressed Sparse Column (CSC) format
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    std::cout << "Matrix A generated: " << A.rows() << "x" << A.cols()
        << " with " << A.nonZeros() << " non-zeros." << std::endl;

    // 4. Initialize the CHOLMOD Solver
    ECholmodSolver solver;

    // --- TIMING START ---
    auto start = std::chrono::high_resolution_clock::now();

    // 5. Precompute Phase (Analyze pattern and compute L L^T)
    solver.analyzePattern(A);
    solver.factorize(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "FAILED: CHOLMOD factorization failed. Is the matrix SPD?" << std::endl;
        return -1;
    }

    // 6. Solve Phase (Back-substitution for target positions)
    x = solver.solve(b);

    if (solver.info() != Eigen::Success) {
        std::cerr << "FAILED: CHOLMOD solve phase failed." << std::endl;
        return -1;
    }

    // --- TIMING END ---
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "SUCCESS: CHOLMOD factorized and solved the system in "
        << duration.count() << " ms." << std::endl;
    std::cout << "Sanity check - first vertex position output: " << x[0] << std::endl;

    return 0;
}