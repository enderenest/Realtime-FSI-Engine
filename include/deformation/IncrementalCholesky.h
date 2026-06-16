#pragma once

#include "core/Types.h"
#include <vector>

// ============================================================
// IncrementalCholesky
// ------------------------------------------------------------
// Simplicial LDL^T factorization of a FIXED-pattern symmetric matrix A, with
// in-place column updates on elimination-tree paths. This is the numerical
// engine for the dynamic-constraint deformation of Herholz et al.
// (Equation 14 + Algorithms 1 & 3), Section 4.1 ("removing degrees of freedom").
//
// The whole point: the factor L lives forever and never changes dimension.
// Constraining vertex k does not delete it — it "detaches" k by setting row and
// column k of the working matrix A^B to the kth identity row/column (Eq.14):
//
//     A^B_ij = 0            if (i in B or j in B) and i != j
//              1            if i in B and i == j
//              A_ij         otherwise
//
// Because A^B only ZEROES entries of A (never adds), the symbolic structure
// (elimination tree, nonzero pattern of L) is the SAME for every constraint
// state. We compute it ONCE from the unconstrained A (analyze), then every
// constraint-set change recomputes only the L columns on the tree paths from the
// changed columns up to the root (updateColumns, Algorithm 3). A target-only
// change touches the factor not at all — the caller just re-solves.
//
// Conventions:
//   * A is symmetric; we read only its upper triangle (entries (i,k), i <= k).
//   * "column k" in Algorithm 3 == one up-looking factorization step k here.
//   * L is unit lower-triangular; only its strictly-lower entries are stored
//     (CSC: _Lp/_Li/_Lx, row indices ascending per column), D holds the diagonal.
//   * A constrained vertex k ends up with row k and column k of L equal to e_k
//     and D[k] = 1, so a solve passes its RHS straight through as x_k = b_k.
//
// Correctness is checkable: factorizeFull() == updateColumns() over all columns,
// and a solve must match a fresh Eigen factorization of the same A^B (the
// LaplacianDeformation wrapper exposes that comparison).
// ============================================================
class IncrementalCholesky {
public:
    // Step 2: symbolic analysis from the UNCONSTRAINED A (pattern must cover
    // every constraint state we will reach). Computes the elimination tree and
    // the fixed nonzero pattern of L. O(nnz). Call once.
    void analyze(const ESparseMatrix& A);

    // Set the current constraint mask (Eq.14). constrained[k] != 0 detaches k.
    // Applied lazily by the numeric routines; does not itself touch the factor.
    void setConstrained(std::vector<char> constrainedMask);

    // Full numeric factorization of the current A^B (recomputes every column).
    // Equivalent to updateColumns() with all columns marked changed.
    void factorizeFull(const ESparseMatrix& A);

    // Algorithm 3: recompute only the columns on elimination-tree paths from
    // changedCols up to the root. changedCols (= I0) must list the columns of
    // A^B that actually differ: the entering/leaving vertices and their
    // neighbours. Must be preceded by setConstrained() reflecting the new state.
    void updateColumns(const ESparseMatrix& A, const std::vector<int>& changedCols);

    // Solve A^B x = b in place (forward / diagonal / backward). The three axis
    // RHS share this one factor (Step 6).
    void solveInPlace(EVecX& b) const;

    int  size() const { return _n; }
    bool ok()   const { return _ok; }

    // Elimination-tree parent of k (-1 at a root). Exposed for diagnostics.
    const std::vector<int>& parent() const { return _parent; }

private:
    int  _n  = 0;
    bool _ok = false;

    std::vector<int>    _parent;   // elimination tree (parent[k] = -1 at root)
    std::vector<int>    _Lp, _Li;  // CSC pattern of strictly-lower L (rows ascending)
    std::vector<double> _Lx;       // values aligned with _Li
    std::vector<double> _D;        // LDL^T diagonal
    std::vector<char>   _constrained;

    // Per-step workspace (reused across columns; never reallocated mid-run).
    mutable std::vector<double> _Y;        // dense accumulator
    mutable std::vector<int>    _pattern;  // row-k pattern scratch
    mutable std::vector<int>    _flag;     // visited stamps (monotone, see _mark)
    mutable int                 _mark = 0;

    // Recompute up-looking step k (writes D[k] and L(k, j) for j in row-k
    // pattern). Reads only columns < k, so processing in ascending order makes
    // both factorizeFull and updateColumns correct.
    void recomputeColumn(const ESparseMatrix& A, int k);

    // Position in _Li/_Lx of entry L(row=k, col=i). k must be in column i's
    // pattern. O(log nnz_col) binary search; the pattern is fixed.
    int colSlot(int i, int k) const;
};
