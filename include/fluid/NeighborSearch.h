#ifndef NEIGHBOR_SEARCH_H
#define NEIGHBOR_SEARCH_H

#include <vector>
#include "core/Types.h"
#include "fluid/Particle.h"

class NeighborSearch {
public:
    struct Entry {
        I32 index;
        U32 key;
    };

    explicit NeighborSearch(F32 cellSize, U32 hashSize);

    void build(const std::vector<Particle>& particles);

    const std::vector<I32>& getNeighbors(I32 i) const;

private:
    struct CellCoord { I32 x, y, z; };

    CellCoord positionToCell(const PVec3& p) const;
    U32 hashCell(I32 x, I32 y, I32 z) const;
    U32 getKey(U32 hash) const;

private:
    F32 _cellSize;
    U32 _hashSize;

    std::vector<Entry> _spatialLookup;
    std::vector<U32>   _startIndices;
    std::vector<std::vector<I32>> _neighbors;
};

#endif