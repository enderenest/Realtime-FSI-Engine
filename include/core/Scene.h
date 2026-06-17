#ifndef SCENE_H
#define SCENE_H

#include "core/config.h"
#include "core/Types.h"
#include <vector>

// ============================================================
// SceneDeform
// ------------------------------------------------------------
// Per-scene two-way deformation tuning. These mirror the live g_* tuning
// globals in main.cpp and are applied by loadScene(); they decide how soft or
// stiff a scene's mesh reacts to the fluid push and how its handle set is
// formed. A stiffer mesh (low stiffness, low maxTotalDisp) barely dents; a
// soft one (high stiffness, larger clamps) shows the "money shot" deformation.
// ============================================================
struct SceneDeform {
    float stiffness;            // force -> displacement gain
    float maxDisplacement;      // per-frame displacement clamp (keeps the solve stable)
    float maxTotalDisp;         // hard clamp on |deformed - rest| (stops blow-up)
    bool  restoreEnabled;       // elastic spring-back toward the rest shape
    float restoreStrength;      // fraction of the remaining offset removed per frame
    float controlThreshold;     // |force| to ENTER the handle set
    float controlThresholdLow;  // |force| to LEAVE the handle set (hysteresis, < enter)
    int   localKRing;           // geodesic ring radius of the local solve patch
    float contactRadius;        // surface band (world units) counted as fluid contact
};

// ============================================================
// Scene
// ------------------------------------------------------------
// A single bundle of everything the demo boots with: the fluid solver
// parameters, the camera framing, and — the focus of this project — the
// solid MESH the fluid interacts with, including how it is placed and how it
// deforms. The application always starts in the anchor-picking workflow with
// the selected scene's mesh already loaded; there is no separate "fluid only"
// mode.
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

    // ---- Mesh placement (applied during Setup, before anchors/spawn) ----
    // The mesh is first centred/scaled to meshFitSize, then offset and spun in
    // place by these. Lets a scene frame an asset the way the fluid should hit it
    // (e.g. tilt the hand so water runs off the palm). Editable live in the panel.
    PVec3 meshTranslate;      // world-space offset from the fitted/centred pose
    PVec3 meshRotDeg;         // euler rotation in degrees about the mesh centre (X,Y,Z)

    // ---- Deformation characteristics ----
    SceneDeform deform;
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
                /* solverIterations */   3,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 1.0f, 1.0f, 1.0f },   // inside the cube [1.5,4.5]^3
                /* spawnMax */          { 5.0f, 2.5f, 5.0f },
                /* spawnRandom */       false,
                /* spacing */           0.11f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.3f,
                /* hashSize */          1 << 15,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 2.0f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   false,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.35f,
            /* camDist */  13.0f,
            /* pointSize */ 8.0f,
            /* meshFile */     "cube1538.off",
            /* meshFitSize */  4.5f,
            /* sdfCell */      0.10f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ true,
            /* meshTranslate */ { 0.0f, 0.5f, 0.0f },
            /* meshRotDeg */    { 0.0f, 0.0f, 0.0f },
            /* deform */ {
                /* stiffness */           1.0f,   // stiff box: barely yields to the held fluid
                /* maxDisplacement */     0.02f,
                /* maxTotalDisp */        0.25f,
                /* restoreEnabled */      true,
                /* restoreStrength */     0.02f,
                /* controlThreshold */    1.8f,
                /* controlThresholdLow */ 1.5f,
                /* localKRing */          2,
                /* contactRadius */       0.12f,
            },
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
                /* spawnMin */          { 0.2f, 0.2f, 0.2f },   // block above the cube
                /* spawnMax */          { 5.8f, 1.0f, 5.8f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 15,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 2.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   false,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.35f,
            /* camDist */  12.0f,
            /* pointSize */ 8.0f,
            /* meshFile */     "cube1538.off",
            /* meshFitSize */  3.0f,
            /* sdfCell */      0.10f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
            /* meshTranslate */ { 0.0f, 0.5f, 0.0f },
            /* meshRotDeg */    { 0.0f, 0.0f, 0.0f },
            /* deform */ {
                /* stiffness */           1.0f,
                /* maxDisplacement */     0.02f,
                /* maxTotalDisp */        0.30f,
                /* restoreEnabled */      true,
                /* restoreStrength */     0.02f,
                /* controlThreshold */    1.20f,
                /* controlThresholdLow */ 1.0f,
                /* localKRing */          3,
                /* contactRadius */       0.12f,
            },
        },

        // 2 — Surface Bull: fluid pours onto a Bull figure.
        {
            "Surface Bull",
            {
                /* dt */                1.0f / 60.0f,
                /* gravity */           { 0.0f, -9.81f, 0.0f },
                /* h */                 0.15f,
                /* rho0 */              1000.0f,
                /* solverIterations */   5,
                /* substepIterations */  2,
                /* eps */               0.0001f,
                /* particleCount */     1 << 14,
                /* spawnMin */          { 0.2f, 0.2f, 0.2f },
                /* spawnMax */          { 5.8f, 1.0f, 5.8f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 15,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 2.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   false,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.30f,
            /* camDist */  12.0f,
            /* pointSize */ 8.0f,
            /* meshFile */     "386.off",
            /* meshFitSize */  4.0f,
            /* sdfCell */      0.08f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
            /* meshTranslate */ { 0.0f, 0.0f, 0.0f },
            /* meshRotDeg */    { 270.0f, 0.0f, 0.0f },
            /* deform */ {
                /* stiffness */           2.0f,   // soft organic shell: the deformation "money shot"
                /* maxDisplacement */     0.020f,
                /* maxTotalDisp */        0.40f,
                /* restoreEnabled */      true,
                /* restoreStrength */     0.04f,
                /* controlThreshold */    0.5f,
                /* controlThresholdLow */ 0.3f,
                /* localKRing */          2,
                /* contactRadius */       0.12f,
            },
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
                /* spawnMin */          { 0.2f, 0.2f, 0.2f },
                /* spawnMax */          { 5.8f, 1.0f, 5.8f },
                /* spawnRandom */       false,
                /* spacing */           0.1f,
                /* initialVelocity */   { 0.0f, 0.0f, 0.0f },
                /* boundsMin */         { 0.0f, 0.0f, 0.0f },
                /* boundsMax */         { 6.0f, 6.0f, 6.0f },
                /* boundDamping */      0.4f,
                /* hashSize */          1 << 15,
                /* enableSCorr */       true,
                /* kCorr */             0.001f,
                /* nCorr */             4.0f,
                /* deltaQ */            0.3f,
                /* cohesionStrength */  0.001f,
                /* enableViscosity */   true,
                /* viscosity */         0.1f,
                /* interactionRadius */ 1.5f,
                /* interactionStrength */ 25.0f,
                /* enableVorticity */   false,
                /* vorticityEpsilon */  0.05f,
            },
            /* camYaw */   0.6f,
            /* camPitch */ 0.30f,
            /* camDist */  12.0f,
            /* pointSize */ 8.0f,
            /* meshFile */     "200.off",
            /* meshFitSize */  4.5f,
            /* sdfCell */      0.08f,
            /* sdfPadding */   0.05f,
            /* sdfContainer */ false,
            /* meshTranslate */ { 0.0f, 0.0f, 0.0f },
            /* meshRotDeg */    { 90.0f, 0.0f, 0.0f },
            /* deform */ {
                /* stiffness */           1.5f,   // soft, but holds shape better than Homer
                /* maxDisplacement */     0.025f,
                /* maxTotalDisp */        0.45f,
                /* restoreEnabled */      true,
                /* restoreStrength */     0.05f,
                /* controlThreshold */    0.45f,
                /* controlThresholdLow */ 0.28f,
                /* localKRing */          3,
                /* contactRadius */       0.12f,
            },
        },
    };
    return scenes;
}

#endif
