#include "deformation/IncrementalCholesky.h"

#include <algorithm>
#include <iostream>

// Iterate the UPPER-triangle off-diagonal entries (i, k) with i < k of column k
// of the symmetric matrix A. A must be compressed with ascending inner (row)
// indices (Eigen's default after setFromTriplets / products); we stop at the
// first row >= k. A may hold both triangles — rows i > k are simply never seen.
namespace {
template <typename F>
inline void forEachUpperOffDiag(const ESparseMatrix& A, int k, F&& f) {
    for (ESparseMatrix::InnerIterator it(A, k); it; ++it) {
        const int i = (int)it.row();
        if (i >= k) break;           // diagonal or lower: done with this column
        f(i, it.value());
    }
}
inline double diagonalOf(const ESparseMatrix& A, int k) {
    for (ESparseMatrix::InnerIterator it(A, k); it; ++it) {
        if ((int)it.row() == k) return it.value();
        if ((int)it.row() >  k) break;
    }
    return 0.0;
}
} // namespace

void IncrementalCholesky::analyze(const ESparseMatrix& A) {
    _n  = (int)A.rows();
    _ok = false;
    _parent.assign(_n, -1);
    _flag.assign(_n, 0);
    _mark = 0;

    // --- elimination tree + per-column counts (Davis' ldl_symbolic) ----------
    std::vector<int> lnz(_n, 0);
    std::vector<int> stamp(_n, -1);
    for (int k = 0; k < _n; ++k) {
        stamp[k] = k;
        forEachUpperOffDiag(A, k, [&](int i, double /*v*/) {
            // walk i up the partially-built etree until we reach a node already
            // visited for this k; first visit links it as a child of k. The walk
            // terminates naturally at k (stamp[k] == k).
            for (; stamp[i] != k; i = _parent[i]) {
                if (_parent[i] == -1) _parent[i] = k;
                ++lnz[i];
                stamp[i] = k;
            }
        });
    }

    _Lp.assign(_n + 1, 0);
    for (int k = 0; k < _n; ++k) _Lp[k + 1] = _Lp[k] + lnz[k];
    _Li.assign(_Lp[_n], 0);
    _Lx.assign(_Lp[_n], 0.0);
    _D.assign(_n, 0.0);

    // --- fill row indices (ascending per column) -----------------------------
    // Re-walk the same etree paths, appending row k into each ancestor column.
    std::vector<int> cnt(_n, 0);
    std::fill(stamp.begin(), stamp.end(), -1);
    for (int k = 0; k < _n; ++k) {
        stamp[k] = k;
        forEachUpperOffDiag(A, k, [&](int i, double /*v*/) {
            for (; stamp[i] != k; i = _parent[i]) {
                _Li[_Lp[i] + cnt[i]] = k;   // k > i, and k increases -> ascending
                ++cnt[i];
                stamp[i] = k;
            }
        });
    }
    // Within each column the indices are already ascending (k increased), but be
    // defensive in case A's iteration order ever differs.
    for (int i = 0; i < _n; ++i)
        std::sort(_Li.begin() + _Lp[i], _Li.begin() + _Lp[i + 1]);

    _ok = (_n > 0);
}

void IncrementalCholesky::setConstrained(std::vector<char> constrainedMask) {
    _constrained = std::move(constrainedMask);
    if ((int)_constrained.size() != _n) _constrained.assign(_n, 0);
}

int IncrementalCholesky::colSlot(int i, int k) const {
    const auto begin = _Li.begin() + _Lp[i];
    const auto end   = _Li.begin() + _Lp[i + 1];
    const auto it    = std::lower_bound(begin, end, k);
    return (int)(it - _Li.begin());   // caller guarantees k is present
}

void IncrementalCholesky::recomputeColumn(const ESparseMatrix& A, int k) {
    if (_Y.empty())       _Y.assign(_n, 0.0);
    if (_pattern.empty()) _pattern.assign(_n, 0);

    const int stamp = ++_mark;       // unique, monotone -> no per-pass flag reset
    _flag[k] = stamp;                // traversal stops when it reaches column k

    // Build the row-k pattern from the UNCONSTRAINED structure so slot positions
    // are identical in every constraint state. Gather A^B(:,k) into _Y.
    int top = _n;
    _Y[k] = 0.0;
    forEachUpperOffDiag(A, k, [&](int i0, double v) {
        // A^B(i,k): zeroed if i is constrained (k is handled below). k constrained
        // -> the whole column is e_k, so contributions are irrelevant anyway.
        const double aik = _constrained[i0] ? 0.0 : v;
        _Y[i0] += aik;
        int len = 0;
        for (int i = i0; _flag[i] != stamp; i = _parent[i]) {
            _pattern[len++] = i;
            _flag[i] = stamp;
        }
        while (len > 0) _pattern[--top] = _pattern[--len];   // reverse into [top, n)
    });

    // --- constrained k: row/col k of L become e_k, D[k] = 1 (Eq.14) ----------
    if (_constrained[k]) {
        _D[k] = 1.0;
        for (int p = top; p < _n; ++p) {        // zero L(k, i) for i in pattern
            const int i = _pattern[p];
            _Lx[colSlot(i, k)] = 0.0;
            _Y[i] = 0.0;                         // clear gather residue (no free-branch cleanup)
        }
        return;
    }

    // --- free k: standard up-looking LDL^T column ----------------------------
    _D[k] = diagonalOf(A, k);
    for (int p = top; p < _n; ++p) {
        const int i  = _pattern[p];
        const double yi = _Y[i];
        _Y[i] = 0.0;

        // Read column i's entries for rows < k (the prefix before slot of k).
        const int s = colSlot(i, k);
        for (int q = _Lp[i]; q < s; ++q) _Y[_Li[q]] -= _Lx[q] * yi;

        const double lki = (_D[i] != 0.0) ? yi / _D[i] : 0.0;
        _D[k] -= lki * yi;
        _Lx[s] = lki;                            // overwrite L(k, i) in place
    }
}

void IncrementalCholesky::factorizeFull(const ESparseMatrix& A) {
    if (!_ok) return;
    _Y.assign(_n, 0.0);
    _pattern.assign(_n, 0);
    std::fill(_flag.begin(), _flag.end(), 0);
    _mark = 0;
    std::fill(_Lx.begin(), _Lx.end(), 0.0);
    for (int k = 0; k < _n; ++k) recomputeColumn(A, k);
}

void IncrementalCholesky::updateColumns(const ESparseMatrix& A,
                                        const std::vector<int>& changedCols) {
    if (!_ok || changedCols.empty()) return;
    if (_Y.empty())       _Y.assign(_n, 0.0);
    if (_pattern.empty()) _pattern.assign(_n, 0);

    // I1 = union of elimination-tree paths from each changed column to the root.
    // Mark with a fresh stamp, collect, then process in ASCENDING order so each
    // step reads already-updated lower columns.
    const int stamp = ++_mark;
    std::vector<int> I1;
    I1.reserve(changedCols.size() * 8);
    for (int seed : changedCols) {
        for (int j = seed; j != -1 && _flag[j] != stamp; j = _parent[j]) {
            _flag[j] = stamp;
            I1.push_back(j);
        }
    }
    std::sort(I1.begin(), I1.end());
    for (int k : I1) recomputeColumn(A, k);
}

void IncrementalCholesky::solveInPlace(EVecX& b) const {
    // L z = b  (unit lower, strictly-lower entries in _Lx)
    for (int k = 0; k < _n; ++k) {
        const double bk = b[k];
        for (int p = _Lp[k]; p < _Lp[k + 1]; ++p) b[_Li[p]] -= _Lx[p] * bk;
    }
    // D y = z
    for (int k = 0; k < _n; ++k) b[k] /= _D[k];
    // L^T x = y  (backward)
    for (int k = _n - 1; k >= 0; --k) {
        double xk = b[k];
        for (int p = _Lp[k]; p < _Lp[k + 1]; ++p) xk -= _Lx[p] * b[_Li[p]];
        b[k] = xk;
    }
}
