#ifndef SCENE_H
#define SCENE_H

#include "core/config.h"
#include <vector>

// ============================================================
// Scene
// ------------------------------------------------------------
// A single bundle of everything the demo boots with: the fluid solver
// parameters, the camera framing, and — the focus of this project — the
// solid MESH the fluid interacts with. The application always starts in the
// anchor-picking workflow with the selected scene's mesh already loaded; there
// is no separate "fluid only" mode.
//
// Scenes come in two flavours, set by sdfContainer:
//   container (true)  — fluid is spawned INSIDE a closed mesh and held there.
//   surface   (false) — fluid is spawned ABOVE the mesh and pours onto it.
// ============================================================
struct Scene {
    const char* name;
    FluidConfig fluid;
    float camYaw;
    float camPitch;
    float camDist;
    float pointSize;

    // ---- Interaction mesh (loaded when the scene is selected) ----
    const char* meshFile;     // file under resources/assets, e.g. "198.off"
    float       meshFitSize;  // fit the mesh's largest extent to this many world units
    float       sdfCell;      // SDF voxel size (~ particle diameter)
    float       sdfPadding;   // keep particle centers this far off the surface
    bool        sdfContainer; // false: surface (fluid pours onto/around the mesh)
                              // true:  container (fluid held inside the mesh)
};

inline const std::vector<Scene>& getScenes()
{
    // All scenes share the same domain (6^3) and fitting (extent -> 3 units,
    // centred at ~(3, 2.4, 3)). Container scenes spawn fluid inside that volume;
    // surface scenes spawn a block above it (y ~ 4.3..5.6) so it pours down.
    static const std::vector<Scene> scenes = {

        // 0 — Container Cube: fluid spawned INSIDE a closed cube, held by the SDF.
        {
            "Container Cube",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -9.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   4,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 15,
                /* spawnMin */          { 1.9f, 2.0f, 1.9f },   // inside the cube [1.5,4.5]^3
                /* spawnMax */          { 4.1f, 3.7f, 4.1f },
                /* spawnRandom */       false,
                /* spacing */           0.11f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.3f,
                /* hashSize */          1 << 18,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.35f,
            /* camDist */  13.0f,
            /* pointSize */ 6.0f,
            /* meshFile */     "cube1538.off",
            /* meshFitSize */  3.0f,
            /* sdfCell */      0.10f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ true,
        },

        // 1 — Surface Cube: fluid pours onto the outside of a cube.
        {
            "Surface Cube",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -9.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   4,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 1.9f, 4.3f, 1.9f },   // block above the cube
                /* spawnMax */          { 4.1f, 5.6f, 4.1f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 17,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.35f,
            /* camDist */  12.0f,
            /* pointSize */ 7.0f,
            /* meshFile */     "cube6146.off",
            /* meshFitSize */  3.0f,
            /* sdfCell */      0.10f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
        },

        // 2 — Surface Homer: fluid pours onto a Homer figure.
        {
            "Surface Homer",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -9.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   4,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 2.2f, 4.3f, 2.2f },
                /* spawnMax */          { 3.8f, 5.6f, 3.8f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 17,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.30f,
            /* camDist */  12.0f,
            /* pointSize */ 7.0f,
            /* meshFile */     "homer_0.15_40_simplified.off",
            /* meshFitSize */  3.0f,
            /* sdfCell */      0.08f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
        },

        // 3 — Surface Hand: fluid pours onto a hand mesh.
        {
            "Surface Hand",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -9.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   4,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 2.2f, 4.3f, 2.2f },
                /* spawnMax */          { 3.8f, 5.6f, 3.8f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 17,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   true,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.30f,
            /* camDist */  12.0f,
            /* pointSize */ 7.0f,
            /* meshFile */     "200.off",
            /* meshFitSize */  3.0f,
            /* sdfCell */      0.08f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
        },
    };
    return scenes;
}

#endif
