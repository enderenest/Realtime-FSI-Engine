#include "fluid/NeighborSearch.h"

#include <algorithm>
#include <cmath>
#include <limits>

static constexpr U32 MAX_U32 = std::numeric_limits<U32>::max();

NeighborSearch::NeighborSearch(F32 cellSize, U32 hashSize)
    : _cellSize(cellSize)
    , _hashSize(hashSize)
{
    // no extra validation here
}

NeighborSearch::CellCoord NeighborSearch::positionToCell(const PVec3& p) const
{
    // Works for both positive and negative positions
    return {
      (I32)std::floor(p.x / _cellSize),
      (I32)std::floor(p.y / _cellSize),
      (I32)std::floor(p.z / _cellSize)
    };
}

U32 NeighborSearch::hashCell(I32 x, I32 y, I32 z) const
{
    const U32 p1 = 73856093u;
    const U32 p2 = 19349663u;
    const U32 p3 = 83492791u;

    // IMPORTANT:
    // Signed -> unsigned conversion is well-defined in C++ (mod 2^32),
    // so this works for negative coordinates too.
    const U32 ux = (U32)x;
    const U32 uy = (U32)y;
    const U32 uz = (U32)z;

    return (ux * p1) ^ (uy * p2) ^ (uz * p3);
}

U32 NeighborSearch::getKey(U32 hash) const
{
    return hash % _hashSize;
}

void NeighborSearch::build(const std::vector<Particle>& particles)
{
    const I32 N = (I32)particles.size();
    if (N == 0) {
        _spatialLookup.clear();
        _startIndices.assign(_hashSize, MAX_U32);
        _neighbors.clear();
        return;
    }

    const F32 h = _cellSize;
    const F32 h2 = h * h;

    _spatialLookup.resize(N);
    _startIndices.assign(_hashSize, MAX_U32);

    // Reuse neighbor list storage instead of rebuilding it every frame
    if ((I32)_neighbors.size() < N) {
        const I32 oldSize = (I32)_neighbors.size();
        _neighbors.resize(N);

        for (I32 i = oldSize; i < N; ++i) {
            _neighbors[i].reserve(64);
        }
    }

    for (I32 i = 0; i < N; ++i) {
        _neighbors[i].clear();
    }

    // 1) Fill spatialLookup (index + key)
    for (I32 i = 0; i < N; ++i) {
        const CellCoord cell = positionToCell(particles[i].predPos);
        const U32 hash = hashCell(cell.x, cell.y, cell.z);
        const U32 key = getKey(hash);
        _spatialLookup[i] = { i, key };
    }

    // 2) Sort by key
    std::sort(_spatialLookup.begin(), _spatialLookup.end(),
        [](const Entry& a, const Entry& b) { return a.key < b.key; });

    // 3) Build startIndices: first index where each key appears
    for (I32 i = 0; i < N; ++i) {
        const U32 key = _spatialLookup[i].key;
        if (_startIndices[key] == MAX_U32) {
            _startIndices[key] = (U32)i;
        }
    }

    // 4) Build neighbor lists by checking 27 surrounding cells
    for (I32 i = 0; i < N; ++i) {
        const PVec3& pi = particles[i].predPos;
        const CellCoord base = positionToCell(pi);

        for (I32 dz = -1; dz <= 1; ++dz)
            for (I32 dy = -1; dy <= 1; ++dy)
                for (I32 dx = -1; dx <= 1; ++dx)
                {
                    const CellCoord c{ base.x + dx, base.y + dy, base.z + dz };

                    const U32 hash = hashCell(c.x, c.y, c.z);
                    const U32 key = getKey(hash);

                    const U32 start = _startIndices[key];
                    if (start == MAX_U32) continue;

                    for (U32 k = start; k < (U32)_spatialLookup.size(); ++k) {
                        if (_spatialLookup[k].key != key) break;

                        const I32 j = _spatialLookup[k].index;
                        if (j == i) continue;

                        const PVec3 dpos = pi - particles[j].predPos;
                        if (norm2(dpos) <= h2) {
                            _neighbors[i].push_back(j);
                        }
                    }
                }
    }
}

const std::vector<I32>& NeighborSearch::getNeighbors(I32 i) const
{
    return _neighbors[i];
}