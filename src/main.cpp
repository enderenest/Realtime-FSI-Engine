// main.cpp — Standalone GLFW/OpenGL viewer with GPU-direct instanced rendering
//
// Performance architecture:
//   Compute shaders write particle data to SSBO binding 0
//   Vertex shader reads directly from that SSBO via gl_InstanceID
//   ZERO CPU readback — data never leaves the GPU

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "core/Config.h"
#include "core/Scene.h"
#include "fluid/PBFluids.h"
#include "fluid/Particle.h"
#include "fluid/SDFBoundary.h"

#include "deformation/LaplacianDeformation.h"   // Step 3: Laplacian solid deformation
#include "interaction/Picking.h"                // Step 3: ray-cast anchor picking

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <array>
#include <filesystem>
#include <memory>
#include <chrono>

// ===================================================================
// Window / Camera state
// ===================================================================
static int   g_winW = 1600, g_winH = 900;
static float g_camYaw = 0.45f;
static float g_camPitch = 0.35f;
static float g_camDist = 12.5f;
static glm::vec3 g_camTarget(1.0f, 1.0f, 1.0f);

static void framebufferCB(GLFWwindow*, int w, int h) {
    g_winW = w; g_winH = h;
    glViewport(0, 0, w, h);
}

static void scrollCB(GLFWwindow*, double, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_camDist -= (float)yoff * 0.3f;
    g_camDist = glm::clamp(g_camDist, 0.3f, 30.0f);
}

static glm::mat4 viewMatrix() {
    float cx = g_camDist * cosf(g_camPitch) * sinf(g_camYaw);
    float cy = g_camDist * sinf(g_camPitch);
    float cz = g_camDist * cosf(g_camPitch) * cosf(g_camYaw);
    return glm::lookAt(g_camTarget + glm::vec3(cx, cy, cz), g_camTarget, { 0, 1, 0 });
}

// ===================================================================
// Shader helpers
// ===================================================================
static std::string readFile(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; return ""; }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static GLuint compileShader(GLenum type, const char* src, const char* label) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log); std::cerr << label << ":\n" << log << "\n"; }
    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log); std::cerr << "Link:\n" << log << "\n"; }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

static GLuint loadProgram(const char* vsPath, const char* fsPath) {
    std::string vsSrc = readFile(vsPath), fsSrc = readFile(fsPath);
    return linkProgram(
        compileShader(GL_VERTEX_SHADER, vsSrc.c_str(), vsPath),
        compileShader(GL_FRAGMENT_SHADER, fsSrc.c_str(), fsPath));
}

static GLuint createInlineProgram(const char* vsSrc, const char* fsSrc) {
    return linkProgram(
        compileShader(GL_VERTEX_SHADER, vsSrc, "inline-vs"),
        compileShader(GL_FRAGMENT_SHADER, fsSrc, "inline-fs"));
}

// ===================================================================
// Embedded line shaders (boundary box)
// ===================================================================
static const char* lineVS = R"(
#version 430 core
layout(location=0) in vec3 aPos;
uniform mat4 VP;
void main() { gl_Position = VP * vec4(aPos, 1.0); }
)";

static const char* lineFS = R"(
#version 430 core
uniform vec3 color;
out vec4 FragColor;
void main() { FragColor = vec4(color, 1.0); }
)";

// ===================================================================
// Boundary box geometry (12 edges = 24 vertices)
// ===================================================================
static void fillBoxVerts(float* out, const PVec3& mn, const PVec3& mx) {
    // 12 edges, each 2 vertices, each 3 floats
    float v[] = {
        mn.x,mn.y,mn.z, mx.x,mn.y,mn.z,  mx.x,mn.y,mn.z, mx.x,mx.y,mn.z,
        mx.x,mx.y,mn.z, mn.x,mx.y,mn.z,  mn.x,mx.y,mn.z, mn.x,mn.y,mn.z,
        mn.x,mn.y,mx.z, mx.x,mn.y,mx.z,  mx.x,mn.y,mx.z, mx.x,mx.y,mx.z,
        mx.x,mx.y,mx.z, mn.x,mx.y,mx.z,  mn.x,mx.y,mx.z, mn.x,mn.y,mx.z,
        mn.x,mn.y,mn.z, mn.x,mn.y,mx.z,  mx.x,mn.y,mn.z, mx.x,mn.y,mx.z,
        mx.x,mx.y,mn.z, mx.x,mx.y,mx.z,  mn.x,mx.y,mn.z, mn.x,mx.y,mx.z,
    };
    std::copy(v, v + 72, out);
}

// ===================================================================
// Particle spawning
// ===================================================================
static float frand01(std::mt19937& rng) {
    static std::uniform_real_distribution<float> d(0.f, 1.f);
    return d(rng);
}

static std::vector<Particle> spawnParticles(const FluidConfig& c) {
    std::vector<Particle> ps;
    ps.reserve(c.particleCount);
    std::mt19937 rng(1337);

    if (c.spawnRandom) {
        PVec3 pad{ c.spacing, c.spacing, c.spacing };
        PVec3 lo = c.spawnMin + pad, hi = c.spawnMax - pad;
        for (U32 i = 0; i < c.particleCount; ++i) {
            Particle pt{};
            pt.pos = { lo.x + (hi.x - lo.x) * frand01(rng),
                       lo.y + (hi.y - lo.y) * frand01(rng),
                       lo.z + (hi.z - lo.z) * frand01(rng) };
            pt.predPos = pt.pos;
            pt.vel = c.initialVelocity;
            ps.push_back(pt);
        }
    }
    else {
        float dx = std::max(1e-6f, c.spacing);
        for (float z = c.spawnMin.z; z <= c.spawnMax.z && (U32)ps.size() < c.particleCount; z += dx)
            for (float y = c.spawnMin.y; y <= c.spawnMax.y && (U32)ps.size() < c.particleCount; y += dx)
                for (float x = c.spawnMin.x; x <= c.spawnMax.x && (U32)ps.size() < c.particleCount; x += dx) {
                    Particle pt{}; pt.pos = { x,y,z }; pt.predPos = pt.pos; pt.vel = c.initialVelocity;
                    ps.push_back(pt);
                }
    }
    return ps;
}

// ===================================================================
// SDF solid boundary (one-way coupling)
// -------------------------------------------------------------------
// Load a mesh, build its signed-distance field, hand it to the fluid, and keep
// the triangle geometry around so we can draw it as a wireframe. The SDF can
// act as an obstacle (fluid flows around) or a container (fluid held inside —
// the water-leak test). This is the only part added on top of the base viewer.
// ===================================================================
static SDFBoundary g_sdf;
static bool   g_sdfEnabled   = false;
static bool   g_sdfContainer = false;  // false: obstacle; true: keep fluid inside the mesh
static float  g_sdfPadding   = 0.05f;  // keep particle centers this far off the surface
static float  g_sdfMeshSize  = 3.0f;   // fit a loaded asset's largest extent to this (world units)
static float  g_sdfCell      = 0.10f;  // SDF voxel size (~ particle diameter)

// Procedural container box for the leak-test scene (world-space, matches the
// "SDF Container (Leak Test)" scene's spawn region / bounds).
static const PVec3 kContainerCenter{ 4.0f, 4.0f, 4.0f };
static const float kContainerHalf = 2.0f;

static GLuint  g_meshVAO = 0, g_meshVBO = 0, g_meshEBO = 0;
static GLsizei g_meshIdxCount = 0;

static std::vector<std::string> g_offFiles;
static int g_selFile = -1;

struct MeshData {
    std::vector<std::array<double, 3>> verts;
    std::vector<std::array<int, 3>>    faces;   // triangulated
};

// ===================================================================
// Two-way coupling — Step 1: contact detection + per-vertex force accumulation
// -------------------------------------------------------------------
// Fluid particles within a thin band of the mesh surface push on it. For each
// contacting particle we find the closest triangle (SDF's BVH), split its push
// force across that triangle's three vertices by barycentric weight, and
// accumulate into a per-vertex force array. Step 2 draws those forces as arrows
// so we can verify contact location/direction/magnitude before any deformation.
// (No deformation here — forces are computed and visualized only.)
// ===================================================================
static MeshData            g_meshData;          // current fitted mesh (same coords as SDF/BVH)
static std::vector<PVec3>  g_vertForces;         // accumulated push force, per mesh vertex
static std::vector<Particle> g_readback;         // scratch: GPU->CPU particle copy
static std::vector<float>  g_arrowVerts;         // scratch: arrow line-segment vertices (xyz)

static bool   g_couplingEnabled  = false;        // master toggle for contact detection
static float  g_contactRadius    = 0.12f;        // surface band counted as contact (world units)
static bool   g_useVelMagnitude  = true;         // force magnitude = velocity-into-surface vs constant
static float  g_forceScale       = 0.50f;        // arrow length per unit force (visualization only)
static float  g_controlThreshold = 0.50f;        // |force| above which a vertex becomes a control point
static int    g_contactCount     = 0;            // particles that contributed force this frame
static int    g_controlCount     = 0;            // vertices above the control-point threshold

static GLuint  g_arrowVAO = 0, g_arrowVBO = 0;
static GLsizei g_arrowVertCount = 0;             // number of arrow line vertices to draw

// ===================================================================
// Two-way coupling — Step 3: anchors-first state machine + Laplacian deform
// -------------------------------------------------------------------
// Anchors are vertices pinned in place (boundary conditions of the solve); the
// user picks them before particles exist. Handles are vertices the fluid pushes,
// determined every frame from contact forces. The Cholesky factorization depends
// on the constrained SET (anchors + handles): anchors are permanent, handles
// change => refactor only when the handle set changes; otherwise just re-solve.
//
//   Inactive : legacy fluid app (unchanged behaviour)
//   Setup    : mesh loaded, picking anchors, no particles
//   Ready    : anchors confirmed, factorization built, particles spawned & paused
//   Running  : fluid flows, contacts -> handles, mesh deforms
//   Paused   : frozen
// ===================================================================
enum class DeformState { Inactive, Setup, Ready, Running, Paused };
static DeformState g_dstate = DeformState::Inactive;

static MyMesh                                 g_deformMesh;   // persistent rest+deformed mesh (OpenMesh)
static std::unique_ptr<LaplacianDeformation>  g_deformer;     // owns the cached factorization
static std::vector<int>                       g_anchors;      // picked anchor vertex indices
static std::vector<int>                       g_lastHandleSet;// previous frame's handle set (sorted; set-change test)
static std::vector<EVec3>                      g_restPositions;// rest vertex positions (captured at Confirm)

static float g_stiffness       = 1.0f;    // force -> displacement gain
static float g_maxDisplacement = 0.02f;   // per-frame displacement clamp (keeps the solve stable)
static float g_maxTotalDisp    = 0.40f;   // hard clamp on |deformed - rest| (stops the mesh blowing up)
static bool  g_restoreEnabled  = true;    // elastic spring-back toward the rest shape
static float g_restoreStrength = 0.05f;   // fraction of the remaining offset removed per frame
static int   g_refactorCount   = 0;       // diagnostics: factorizations this run
static int   g_handleCount     = 0;       // active handles this frame

// Incremental Cholesky (hard constraints, Eq.14 + Alg.3). When on, a handle-set
// change updates the persistent factor along elimination-tree paths instead of
// refactoring; a targets-only frame just re-solves. g_incVerify compares each
// rebuild against a fresh Eigen factorization (slow; debugging only).
//
// OFF by default: the current engine factors with NATURAL ordering, which fills
// in badly on larger meshes and stalls analyze()/factorizeFull(). It needs a
// fill-reducing reordering (AMD) before it is safe to enable on big meshes.
// Until then the proven CHOLMOD soft solver (supernodal + AMD) is the default.
static bool  g_useIncremental  = false;
static bool  g_incVerify       = false;

// SDF/BVH update cadence (two-way coupling): dirty (local) update most frames,
// full rebuild every Nth to reset accumulated drift. Shared counter so the BVH
// rebuild and the SDF full rebuild fire on the same frame.
static int   g_sdfSyncCounter      = 0;
static int   g_sdfFullRebuildEvery = 100;  // N: tune later (CLAUDE.md Day 3)
static float g_sdfDirtyBand        = 0.0f; // 0 => derive from padding; see syncDeformedMesh

// Refresh the collision SDF only every Nth deform frame. The rendered mesh still
// moves every frame; only the boundary the fluid samples lags by a couple frames,
// which the SDF padding/band absorbs (per-frame moves are clamped to
// g_maxDisplacement). This is the single biggest coupling-side cost cut on a
// detailed mesh — the dirty recompute + BVH refit + texture upload no longer run
// every frame. 1 == update every frame (old behaviour).
static int   g_sdfUpdateEvery   = 3;
static int   g_sdfUpdateCounter = 0;

// Handle-set hysteresis. A vertex ENTERS the control set at the high threshold
// and only LEAVES below the low one, so borderline vertices stop flickering in
// and out. A stable set means precomputeSystem() (the per-frame refactor — full
// CHOLMOD or incremental Alg.3) runs only on a real change; most frames are just
// a re-solve. g_controlThreshold is the enter level; this is the (lower) exit.
static float g_controlThresholdLow = 0.30f;

// Per-phase timing (ms), shown in the overlay so we optimize the real bottleneck.
static double g_msContacts  = 0.0;   // computeContactForces (readback + parallel scan)
static double g_msRefactor  = 0.0;   // deformStep: setControlPoints + precomputeSystem
static double g_msSolve     = 0.0;   // deformStep: solve()
static double g_msSdfUpdate = 0.0;   // syncDeformedMesh: BVH refit + distance recompute
static double g_msSdfUpload = 0.0;   // syncDeformedMesh: GPU texture upload
using PerfClock = std::chrono::high_resolution_clock;
static inline double msSince(PerfClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(PerfClock::now() - t0).count();
}

static GLuint  g_anchorVAO = 0, g_anchorVBO = 0;
static GLsizei g_anchorPtCount = 0;      // anchor points to draw
static bool    g_prevLeftDown = false;   // edge-detect for anchor picking clicks

static bool isAnchor(int v) {
    for (int a : g_anchors) if (a == v) return true;
    return false;
}

static bool loadOFF(const std::string& path, MeshData& out) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }

    // Skip comment lines (start with #) and blank lines before the header.
    std::string header;
    while (std::getline(f, header)) {
        if (header.empty() || header[0] == '#') continue;
        // trim trailing whitespace
        while (!header.empty() && (header.back() == '\r' || header.back() == ' ')) header.pop_back();
        break;
    }
    if (header.find("OFF") == std::string::npos) {
        std::cerr << "Not an OFF file: " << path << "\n"; return false;
    }

    // Skip comment/blank lines before the counts line.
    int numV = 0, numF = 0, numE = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        if (ss >> numV >> numF >> numE) break;
    }

    // Read each vertex line in full; take only the first 3 values as XYZ.
    // Some exporters embed normals/colors without updating the header tag,
    // so flag-based skipping is unreliable — reading the whole line is safer.
    out.verts.resize(numV);
    for (int i = 0; i < numV; ++i) {
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] != '#') break;
        }
        std::istringstream ss(line);
        ss >> out.verts[i][0] >> out.verts[i][1] >> out.verts[i][2];
        // remaining values on the line (normals, colors, texcoords) are ignored
    }

    out.faces.clear(); out.faces.reserve(numF);
    for (int i = 0; i < numF; ++i) {
        while (std::getline(f, line)) {
            if (!line.empty() && line[0] != '#') break;
        }
        std::istringstream ss(line);
        int n; ss >> n;
        std::vector<int> poly(n);
        for (int j = 0; j < n; ++j) ss >> poly[j];
        for (int j = 1; j + 1 < n; ++j)        // fan triangulation
            out.faces.push_back({ poly[0], poly[j], poly[j + 1] });
    }
    return true;
}

// Center the mesh in the domain and scale its largest extent to targetSize, so
// an arbitrary asset lands where the fluid can reach it.
static void fitMeshToDomain(MeshData& m, const PVec3& center, double targetSize) {
    if (m.verts.empty()) return;
    double lo[3] = { 1e30, 1e30, 1e30 }, hi[3] = { -1e30, -1e30, -1e30 };
    for (auto& v : m.verts)
        for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], v[k]); hi[k] = std::max(hi[k], v[k]); }
    const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]), cz = 0.5 * (lo[2] + hi[2]);
    const double extent = std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9 });
    const double s = targetSize / extent;
    for (auto& v : m.verts) {
        v[0] = (v[0] - cx) * s + center.x;
        v[1] = (v[1] - cy) * s + center.y;
        v[2] = (v[2] - cz) * s + center.z;
    }
}

static void scanOffAssets() {
    namespace fs = std::filesystem;
    g_offFiles.clear();
    const fs::path dir = fs::path(RESOURCES_PATH) / "assets";
    if (!fs::exists(dir)) return;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".off") g_offFiles.push_back(e.path().string());
    std::sort(g_offFiles.begin(), g_offFiles.end());
}

// Build the SDF from a world-space mesh, attach it to the fluid (in the current
// obstacle/container mode), and upload the triangle geometry for wireframe draw.
static void uploadMeshAndSDF(const MeshData& m, PBFluids& fluid) {
    g_sdf.buildFromMesh(m.verts, m.faces, g_sdfCell);
    g_sdf.uploadToGPU();
    fluid.setSDFBoundary(&g_sdf);
    fluid.setSDFPadding(g_sdfPadding);
    fluid.setSDFContainment(g_sdfContainer);
    g_sdfEnabled = true;

    std::vector<float> pos; pos.reserve(m.verts.size() * 3);
    for (auto& v : m.verts) { pos.push_back((float)v[0]); pos.push_back((float)v[1]); pos.push_back((float)v[2]); }
    std::vector<unsigned> idx; idx.reserve(m.faces.size() * 3);
    for (auto& fc : m.faces) { idx.push_back(fc[0]); idx.push_back(fc[1]); idx.push_back(fc[2]); }
    g_meshIdxCount = (GLsizei)idx.size();

    glBindVertexArray(g_meshVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_meshVBO);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(float), pos.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_meshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned), idx.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    // Keep the mesh on the CPU (same world coords as the SDF/BVH) so two-way
    // coupling can look up triangle vertices for barycentric force splitting.
    g_meshData = m;
    g_vertForces.assign(m.verts.size(), PVec3{ 0.f, 0.f, 0.f });
}

// Load mesh -> fit into domain -> build SDF -> attach -> draw. (obstacle by default)
static void loadMeshBoundary(const std::string& path, PBFluids& fluid) {
    MeshData m;
    if (!loadOFF(path, m)) return;

    const FluidConfig& c = fluid.params();
    const PVec3 center{
        0.5f * (c.boundsMin.x + c.boundsMax.x),
        c.boundsMin.y + 0.40f * (c.boundsMax.y - c.boundsMin.y),
        0.5f * (c.boundsMin.z + c.boundsMax.z)
    };
    fitMeshToDomain(m, center, g_sdfMeshSize);
    uploadMeshAndSDF(m, fluid);
    std::cout << "SDF boundary loaded: " << path << "\n";
}

// A closed axis-aligned box with outward-facing triangles (so the SDF is
// negative inside). Used as the "big mesh that holds particles" leak test.
static MeshData makeBoxMesh(const PVec3& center, float half) {
    MeshData m;
    const float h = half;
    m.verts = {
        {center.x - h, center.y - h, center.z - h}, // 0
        {center.x + h, center.y - h, center.z - h}, // 1
        {center.x + h, center.y + h, center.z - h}, // 2
        {center.x - h, center.y + h, center.z - h}, // 3
        {center.x - h, center.y - h, center.z + h}, // 4
        {center.x + h, center.y - h, center.z + h}, // 5
        {center.x + h, center.y + h, center.z + h}, // 6
        {center.x - h, center.y + h, center.z + h}, // 7
    };
    m.faces = {
        {0,2,1}, {0,3,2},   // -Z
        {4,5,6}, {4,6,7},   // +Z
        {0,7,3}, {0,4,7},   // -X
        {1,2,6}, {1,6,5},   // +X
        {0,1,5}, {0,5,4},   // -Y (floor)
        {3,7,6}, {3,6,2},   // +Y (ceiling)
    };
    return m;
}

// Build the container box, switch the SDF to container mode, and attach it.
static void loadContainerBox(PBFluids& fluid) {
    g_sdfContainer = true;
    MeshData box = makeBoxMesh(kContainerCenter, kContainerHalf);
    uploadMeshAndSDF(box, fluid);
    std::cout << "SDF container box loaded (leak test)\n";
}

// ===================================================================
// Two-way coupling — Step 1 implementation
// ===================================================================

// Cheap nearest-voxel SDF lookup used only to reject particles that are clearly
// far from the surface before paying for a BVH query. Returns a large value when
// the point is outside the SDF grid.
static float sampleSDFNearest(const PVec3& P) {
    const Eigen::Vector3d& o   = g_sdf.origin();
    const double           cs  = g_sdf.cellSize();
    const Eigen::Vector3i& dim = g_sdf.dimensions();
    const int ix = (int)std::lround((P.x - o.x()) / cs);
    const int iy = (int)std::lround((P.y - o.y()) / cs);
    const int iz = (int)std::lround((P.z - o.z()) / cs);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= dim.x() || iy >= dim.y() || iz >= dim.z())
        return 1e30f;
    return g_sdf.distances()[g_sdf.index(ix, iy, iz)];
}

// Barycentric coordinates (u,v,w) of point Pt with respect to triangle (A,B,C),
// so Pt = u*A + v*B + w*C. Pt comes from closestPointOnTriangle, so it lies on
// the triangle and the weights are already in [0,1]; we clamp/normalize for safety.
static void barycentric(const EVec3& Pt, const EVec3& A, const EVec3& B, const EVec3& C,
                        float& u, float& v, float& w) {
    const EVec3 v0 = B - A, v1 = C - A, v2 = Pt - A;
    const double d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
    const double d20 = v2.dot(v0), d21 = v2.dot(v1);
    const double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-20) { u = 1.f; v = 0.f; w = 0.f; return; }
    const double vv = (d11 * d20 - d01 * d21) / denom;
    const double ww = (d00 * d21 - d01 * d20) / denom;
    double uu = 1.0 - vv - ww;
    uu = std::max(0.0, uu); double vc = std::max(0.0, vv), wc = std::max(0.0, ww);
    const double s = uu + vc + wc;
    if (s > 1e-20) { uu /= s; vc /= s; wc /= s; }
    u = (float)uu; v = (float)vc; w = (float)wc;
}

// Append one force arrow (a shaft + two head barbs) to 'out' as GL_LINES vertices.
static void pushArrow(std::vector<float>& out, const PVec3& base, const PVec3& vec) {
    auto push = [&](const PVec3& p) { out.push_back(p.x); out.push_back(p.y); out.push_back(p.z); };
    const PVec3 tip = base + vec;
    push(base); push(tip);                       // shaft

    const float len = norm(vec);
    if (len < 1e-6f) return;
    const PVec3 d = vec * (1.0f / len);
    // A vector not parallel to d, to build a perpendicular for the arrowhead.
    const PVec3 ref = (std::fabs(d.y) < 0.9f) ? PVec3{ 0,1,0 } : PVec3{ 1,0,0 };
    PVec3 side{ d.y * ref.z - d.z * ref.y,        // d x ref
                d.z * ref.x - d.x * ref.z,
                d.x * ref.y - d.y * ref.x };
    const float sl = norm(side);
    if (sl < 1e-6f) return;
    side *= (1.0f / sl);
    const float hs = len * 0.25f;                 // head size
    const PVec3 backTip = tip - d * hs;
    push(tip); push(backTip + side * hs);
    push(tip); push(backTip - side * hs);
}

// Detect contacts, distribute push forces onto mesh vertices by barycentric
// weight, and rebuild the arrow buffer. No deformation — Step 1 + Step 2 only.
static void computeContactForces(PBFluids& fluid) {
    const auto _tContacts = PerfClock::now();
    g_arrowVertCount = 0;
    g_contactCount   = 0;
    g_controlCount   = 0;
    g_msContacts     = 0.0;
    if (!g_sdf.valid() || g_meshData.verts.empty()) return;

    std::fill(g_vertForces.begin(), g_vertForces.end(), PVec3{ 0.f, 0.f, 0.f });
    fluid.readbackParticles(g_readback);

    const float band = g_contactRadius;
    const float cell = (float)g_sdf.cellSize();

    // Parallel over particles. The shared g_sdf (BVH + voxel grid) and g_meshData
    // are read-only here, so the only write hazard is the per-vertex force
    // accumulation: each thread sums into a private buffer, then we reduce once.
    const int nParticles = (int)g_readback.size();
    const int nVerts     = (int)g_vertForces.size();
    int contactCount = 0;

    #pragma omp parallel
    {
        std::vector<PVec3> localForces(nVerts, PVec3{ 0.f, 0.f, 0.f });
        int localContacts = 0;

        #pragma omp for schedule(static) nowait
        for (int pi = 0; pi < nParticles; ++pi) {
            const Particle& pt = g_readback[pi];
            const PVec3 P = pt.pos;

            // Reject particles clearly away from the surface (cheap), then test exactly.
            if (std::fabs(sampleSDFNearest(P)) > band + cell) continue;

            const SDFBoundary::ClosestSurface cs = g_sdf.closestTriangle(P);
            if (cs.face < 0) continue;

            const PVec3 toSurf = cs.point - P;    // points from the fluid into the surface
            const float dist   = norm(toSurf);
            if (dist > band) continue;

            const std::array<int, 3>& f = g_meshData.faces[cs.face];
            const EVec3 A(g_meshData.verts[f[0]][0], g_meshData.verts[f[0]][1], g_meshData.verts[f[0]][2]);
            const EVec3 B(g_meshData.verts[f[1]][0], g_meshData.verts[f[1]][1], g_meshData.verts[f[1]][2]);
            const EVec3 C(g_meshData.verts[f[2]][0], g_meshData.verts[f[2]][1], g_meshData.verts[f[2]][2]);

            // Push direction: into the surface. Use particle->surface when well-defined,
            // else the inward face normal (faces are oriented outward, so negate).
            PVec3 dir;
            if (dist > 1e-6f) {
                dir = toSurf * (1.0f / dist);
            } else {
                const EVec3 n = (B - A).cross(C - A).normalized();
                dir = toPlain(-n);
            }

            // Magnitude: how hard the fluid presses. Velocity component into the
            // surface (grows with the push), or a flat 1.0 per contact.
            float mag = 1.0f;
            if (g_useVelMagnitude) {
                mag = std::max(0.0f, dot(pt.vel, dir));
                if (mag <= 0.0f) continue;        // not actually pressing inward
            }

            float u, v, w;
            const EVec3 Pt(cs.point.x, cs.point.y, cs.point.z);
            barycentric(Pt, A, B, C, u, v, w);

            const PVec3 force = dir * mag;
            localForces[f[0]] += force * u;
            localForces[f[1]] += force * v;
            localForces[f[2]] += force * w;
            ++localContacts;
        }

        #pragma omp critical
        {
            for (int k = 0; k < nVerts; ++k) g_vertForces[k] += localForces[k];
            contactCount += localContacts;
        }
    }
    g_contactCount = contactCount;

    // Build the arrow line buffer and count control points (threshold crossings).
    g_arrowVerts.clear();
    for (size_t i = 0; i < g_vertForces.size(); ++i) {
        const PVec3 F = g_vertForces[i];
        const float m = norm(F);
        if (m < 1e-5f) continue;
        if (m >= g_controlThreshold) ++g_controlCount;
        const PVec3 base{ (float)g_meshData.verts[i][0],
                          (float)g_meshData.verts[i][1],
                          (float)g_meshData.verts[i][2] };
        pushArrow(g_arrowVerts, base, F * g_forceScale);
    }
    g_arrowVertCount = (GLsizei)(g_arrowVerts.size() / 3);

    if (g_arrowVertCount > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, g_arrowVBO);
        glBufferData(GL_ARRAY_BUFFER, g_arrowVerts.size() * sizeof(float),
                     g_arrowVerts.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    g_msContacts = msSince(_tContacts);
}

// ===================================================================
// Two-way coupling — Step 3 implementation (state machine + deformer)
// ===================================================================

// Rebuild the GL wireframe VBO positions from g_meshData (after a deform). The
// vertex count never changes, so a sub-data update is enough.
static void updateMeshVBO() {
    std::vector<float> pos; pos.reserve(g_meshData.verts.size() * 3);
    for (auto& v : g_meshData.verts) {
        pos.push_back((float)v[0]); pos.push_back((float)v[1]); pos.push_back((float)v[2]);
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_meshVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, pos.size() * sizeof(float), pos.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Build the persistent OpenMesh from the current (rest) g_meshData. Vertex order
// matches g_meshData / g_vertForces indices, so contact indices map straight in.
static void buildDeformMeshFromData() {
    g_deformMesh.clear();
    std::vector<MyMesh::VertexHandle> vh(g_meshData.verts.size());
    for (size_t i = 0; i < g_meshData.verts.size(); ++i)
        vh[i] = g_deformMesh.add_vertex(MyMesh::Point(
            g_meshData.verts[i][0], g_meshData.verts[i][1], g_meshData.verts[i][2]));
    for (auto& f : g_meshData.faces)
        g_deformMesh.add_face(vh[f[0]], vh[f[1]], vh[f[2]]);
}

// Refresh the anchor-point buffer (green dots) from current deform-mesh positions.
static void rebuildAnchorBuffer() {
    g_anchorPtCount = (GLsizei)g_anchors.size();
    if (g_anchorPtCount == 0 || g_deformMesh.n_vertices() == 0) { g_anchorPtCount = 0; return; }
    std::vector<float> pts; pts.reserve(g_anchors.size() * 3);
    for (int a : g_anchors) {
        auto p = g_deformMesh.point(MyMesh::VertexHandle(a));
        pts.push_back((float)p[0]); pts.push_back((float)p[1]); pts.push_back((float)p[2]);
    }
    glBindBuffer(GL_ARRAY_BUFFER, g_anchorVBO);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float), pts.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// After a solve: deform mesh -> g_meshData -> GL wireframe -> SDF/BVH update +
// GPU upload, so the fluid collides against the new shape next frame.
//
// Most frames: dirty update (refit BVH + recompute only the voxels near moved
// vertices + partial texture upload). Every Nth frame: full rebuild to reset any
// accumulated drift at dirty-region boundaries (CLAUDE.md Day 2 stability note).
static void syncDeformedMesh(PBFluids& fluid) {
    for (int i = 0; i < (int)g_meshData.verts.size(); ++i) {
        auto p = g_deformMesh.point(MyMesh::VertexHandle(i));
        g_meshData.verts[i] = { p[0], p[1], p[2] };
    }
    updateMeshVBO();        // rendered mesh moves every frame (cheap sub-data upload)
    rebuildAnchorBuffer();

    g_msSdfUpdate = 0.0;
    g_msSdfUpload = 0.0;

    // Refresh the COLLISION SDF only every Nth deform frame. The fluid then
    // samples a boundary that lags a couple frames, absorbed by the SDF padding;
    // per-frame moves are clamped to g_maxDisplacement so it cannot leak. This is
    // the main coupling-side cut on a detailed mesh — refit + dirty recompute +
    // upload no longer run every frame.
    const int every = std::max(1, g_sdfUpdateEvery);
    if (g_sdfUpdateCounter++ % every != 0) {
        fluid.setSDFBoundary(&g_sdf);
        fluid.setSDFPadding(g_sdfPadding);
        fluid.setSDFContainment(g_sdfContainer);
        return;
    }

    const auto _tUpd = PerfClock::now();
    const bool fullRebuild = (g_sdfSyncCounter++ % g_sdfFullRebuildEvery == 0);
    if (fullRebuild || !g_sdf.valid()) {
        g_sdf.buildFromMesh(g_meshData.verts, g_meshData.faces, g_sdfCell, /*verbose=*/false);
        g_msSdfUpdate = msSince(_tUpd);
        const auto _tUp = PerfClock::now();
        g_sdf.uploadToGPU();
        g_msSdfUpload = msSince(_tUp);
    } else {
        // Dirty band must cover the shell the shader samples (padding) plus the
        // largest move SINCE THE LAST SDF REFRESH (every * per-frame clamp), plus
        // a cell of slack. refitDirty marks old AND new positions, so the moved
        // corridor stays fully contained even across skipped frames.
        const double band       = (g_sdfDirtyBand > 0.0f ? (double)g_sdfDirtyBand
                                                          : std::max((double)g_sdfPadding, 2.0 * g_sdf.cellSize()));
        const double bandRadius = band + (double)every * (double)g_maxDisplacement + g_sdf.cellSize();
        const bool   dirty      = g_sdf.refitDirty(g_meshData.verts, bandRadius, /*moveEps=*/1e-4);
        g_msSdfUpdate = msSince(_tUpd);
        if (dirty) {
            const auto _tUp = PerfClock::now();
            g_sdf.uploadDirtyRegion();
            g_msSdfUpload = msSince(_tUp);
        }
    }

    fluid.setSDFBoundary(&g_sdf);
    fluid.setSDFPadding(g_sdfPadding);
    fluid.setSDFContainment(g_sdfContainer);
}

// One deformation step (STATE_RUNNING).
//
// A vertex is a handle this frame if the fluid is pushing it (force over the
// threshold) OR it is still relaxing back toward rest (restoring force on, and it
// was a handle last frame). For each handle the target is:
//
//     target = current + fluidPush          // dent inward
//     target += restoreStrength*(rest-target)// elastic pull back toward rest
//     target = clamp(target, rest, maxTotalDisp)
//
// Free (unconstrained) vertices follow the smooth, rest-based reconstruction, so
// when the handles ease back to rest the whole mesh does too. The factorization is
// rebuilt only when the handle SET changes; targets-only changes just re-solve.
static void deformStep(PBFluids& fluid, float dt) {
    g_handleCount = 0;
    if (!g_deformer || (int)g_restPositions.size() != (int)g_vertForces.size()) return;

    const std::vector<int>& prev = g_lastHandleSet;   // sorted (built in ascending order)
    auto wasHandle = [&](int i) { return std::binary_search(prev.begin(), prev.end(), i); };

    std::vector<int>   handles;
    std::vector<EVec3> targets;
    for (int i = 0; i < (int)g_vertForces.size(); ++i) {
        if (isAnchor(i)) continue;                    // anchors have their own pinned rows

        auto p = g_deformMesh.point(MyMesh::VertexHandle(i));
        const EVec3 cur  = EVec3(p[0], p[1], p[2]);
        const EVec3 rest = g_restPositions[i];

        const float fmag    = norm(g_vertForces[i]);
        const bool  wasH    = wasHandle(i);
        // Hysteresis: a vertex ENTERS the control set at the high threshold and
        // only LEAVES below the low one. This stops borderline vertices flickering
        // in/out, which is what forced a refactor (precomputeSystem) every frame.
        const bool  pushed   = (fmag >= g_controlThreshold)
                            || (wasH && fmag >= g_controlThresholdLow);
        const double offset  = (cur - rest).norm();
        // Keep relaxing only vertices we were already controlling, so the solve's
        // free neighbours don't all get promoted to handles.
        const bool  relaxing = g_restoreEnabled && wasH && offset > 1e-3;
        if (!pushed && !relaxing) continue;

        EVec3 target = cur;

        // Fluid push into the surface, clamped per frame so spikes can't explode.
        if (pushed) {
            PVec3 disp = g_vertForces[i] * (g_stiffness * dt);
            const float dm = norm(disp);
            if (dm > g_maxDisplacement && dm > 1e-12f) disp *= (g_maxDisplacement / dm);
            target += EVec3(disp.x, disp.y, disp.z);
        }

        // Elastic restoring force: ease the target toward the rest position.
        if (g_restoreEnabled)
            target += (double)g_restoreStrength * (rest - target);

        // Safety clamp: never let a vertex stray further than g_maxTotalDisp from rest.
        EVec3 fromRest = target - rest;
        const double off = fromRest.norm();
        if (off > g_maxTotalDisp) target = rest + fromRest * (g_maxTotalDisp / off);

        handles.push_back(i);
        targets.push_back(target);
    }
    g_handleCount = (int)handles.size();

    const bool setChanged = (handles != g_lastHandleSet);
    g_deformer->setControlPoints(handles, targets);
    g_msRefactor = 0.0;
    if (setChanged) {
        const auto _t = PerfClock::now();
        g_deformer->precomputeSystem();               // SET changed -> Alg.3 update (or refactor)
        g_msRefactor = msSince(_t);
        ++g_refactorCount;
        if (g_incVerify) g_deformer->verifyIncremental();
    }
    g_lastHandleSet = handles;

    // Solve when there are active constraints, or exactly once more right after the
    // set empties (so the mesh settles cleanly back to its rest shape).
    g_msSolve = 0.0;
    if (!handles.empty() || setChanged) {
        const auto _t = PerfClock::now();
        g_deformer->solve();
        g_msSolve = msSince(_t);
        syncDeformedMesh(fluid);
    }
}

// SETUP transition: drop particles, build the rest mesh, clear anchors.
static void enterSetup(PBFluids& fluid) {
    g_dstate = DeformState::Setup;
    g_deformer.reset();                            // release the old factorization first
    fluid.setParticles(std::vector<Particle>{});  // no particles while picking anchors
    buildDeformMeshFromData();
    g_anchors.clear();
    g_lastHandleSet.clear();
    g_restPositions.clear();
    rebuildAnchorBuffer();
}

// SETUP -> READY: build the anchors-only factorization, then spawn (paused).
static void confirmAnchorsAndSpawn(PBFluids& fluid, const FluidConfig& fc) {
    if (g_anchors.empty()) { std::cerr << "[Deform] Pick at least one anchor first.\n"; return; }
    g_deformer = std::make_unique<LaplacianDeformation>(g_deformMesh);
    g_deformer->initialize();                     // rest differential coords + normal system (once)
    g_deformer->setUseIncremental(g_useIncremental);
    for (int a : g_anchors) g_deformer->addAnchor(a);
    g_deformer->precomputeSystem();               // analyze (etree) + factorize with anchors only
    ++g_refactorCount;
    g_lastHandleSet.clear();

    // Snapshot the rest shape so the restoring force has a target to return to.
    g_restPositions.resize(g_deformMesh.n_vertices());
    for (int i = 0; i < (int)g_restPositions.size(); ++i) {
        auto p = g_deformMesh.point(MyMesh::VertexHandle(i));
        g_restPositions[i] = EVec3(p[0], p[1], p[2]);
    }

    fluid.setParticles(spawnParticles(fc));       // spawn; stays paused until Start
    g_dstate = DeformState::Ready;
}

// HARD RESET: return both the solid and the fluid to their initial state.
// Drops all deformation state, reloads the boundary mesh to its rest shape
// (which rebuilds the SDF + BVH + wireframe), and respawns the fluid.
static void hardReset(PBFluids& fluid, const FluidConfig& fc) {
    g_dstate = DeformState::Inactive;
    g_deformer.reset();
    g_anchors.clear();
    g_lastHandleSet.clear();
    g_restPositions.clear();
    g_anchorPtCount = 0;
    g_arrowVertCount = 0;
    g_handleCount = 0;

    // Reload the boundary mesh from its source so deformation is undone.
    if (g_selFile >= 0 && g_selFile < (int)g_offFiles.size())
        loadMeshBoundary(g_offFiles[g_selFile], fluid);
    else if (g_sdfContainer)
        loadContainerBox(fluid);

    // Respawn the fluid at its initial configuration.
    fluid.setParticles(spawnParticles(fc));
}

// ===================================================================
// MAIN
// ===================================================================
int main() {
    // ------ GLFW / OpenGL init -----------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g_winW, g_winH, "HybridSim", nullptr, nullptr);
    if (!window) { std::cerr << "GLFW window failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "GLAD failed\n"; return -1; }

    glfwSwapInterval(0);                       // uncapped FPS
    glfwSetFramebufferSizeCallback(window, framebufferCB);
    glfwSetScrollCallback(window, scrollCB);
    glViewport(0, 0, g_winW, g_winH);

    // ------ ImGui init -------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.0f;

    // ------ Simulation -------------------------------------------
    Config cfg{};
    const auto& scenes = getScenes();
    int currentScene = 0;
    PBFluids fluid(cfg.fluid);
    fluid.setParticles(spawnParticles(cfg.fluid));
    fluid.setCollisionPadding(0.0f);

    // ------ Particle shader (reads SSBO binding 0) ---------------
    GLuint particleProg = loadProgram(
        RESOURCES_PATH "shaders/graphics/VS_Particle.glsl",
        RESOURCES_PATH "shaders/graphics/FS_Particle.glsl");
    GLuint emptyVAO;
    glGenVertexArrays(1, &emptyVAO);

    // ------ Line shader (boundary box) ---------------------------
    GLuint lineProg = createInlineProgram(lineVS, lineFS);

    GLuint boxVAO, boxVBO;
    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);
    glBindVertexArray(boxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // Upload initial box
    float boxData[72];
    fillBoxVerts(boxData, cfg.fluid.boundsMin, cfg.fluid.boundsMax);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(boxData), boxData);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ------ SDF mesh buffers (drawn with lineProg as wireframe) --
    glGenVertexArrays(1, &g_meshVAO);
    glGenBuffers(1, &g_meshVBO);
    glGenBuffers(1, &g_meshEBO);
    scanOffAssets();

    // ------ Force-arrow buffers (Step 2 visualization, drawn with lineProg) --
    glGenVertexArrays(1, &g_arrowVAO);
    glGenBuffers(1, &g_arrowVBO);
    glBindVertexArray(g_arrowVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_arrowVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // ------ Anchor-point buffers (Step 3, drawn with lineProg as GL_POINTS) --
    glGenVertexArrays(1, &g_anchorVAO);
    glGenBuffers(1, &g_anchorVBO);
    glBindVertexArray(g_anchorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_anchorVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // Center camera on bounds
    g_camTarget = glm::vec3(
        (cfg.fluid.boundsMin.x + cfg.fluid.boundsMax.x) * 0.5f,
        (cfg.fluid.boundsMin.y + cfg.fluid.boundsMax.y) * 0.5f,
        (cfg.fluid.boundsMin.z + cfg.fluid.boundsMax.z) * 0.5f);

    // ------ State ------------------------------------------------
    bool  paused = false;
    bool  stepOnce = false;
    bool  respawn = false;
    float pointSize = 8.0f;
    bool  rightDown = false;
    bool  midDown = false;
    double lastRX = 0, lastRY = 0, lastMX = 0, lastMY = 0;

    // FPS counter
    double fpsTime = glfwGetTime();
    int    frames = 0, fps = 0;

    // =================================================================
    // RENDER LOOP
    // =================================================================
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // FPS
        double now = glfwGetTime();
        frames++;
        if (now - fpsTime >= 1.0) { fps = frames; frames = 0; fpsTime = now; }

        // ESC to close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // ---- Orbit camera (right drag) --------------------------
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !ImGui::GetIO().WantCaptureMouse) {
            double mx, my; glfwGetCursorPos(window, &mx, &my);
            if (!rightDown) { rightDown = true; lastRX = mx; lastRY = my; }
            g_camYaw += (float)(mx - lastRX) * 0.005f;
            g_camPitch += (float)(my - lastRY) * 0.005f;
            g_camPitch = glm::clamp(g_camPitch, -1.5f, 1.5f);
            lastRX = mx; lastRY = my;
        }
        else { rightDown = false; }

        // ---- Pan camera (middle drag) ---------------------------
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS && !ImGui::GetIO().WantCaptureMouse) {
            double mx, my; glfwGetCursorPos(window, &mx, &my);
            if (!midDown) { midDown = true; lastMX = mx; lastMY = my; }
            glm::mat4 V = viewMatrix();
            glm::vec3 right(V[0][0], V[1][0], V[2][0]);
            glm::vec3 up(V[0][1], V[1][1], V[2][1]);
            float speed = g_camDist * 0.002f;
            g_camTarget += (-(float)(mx - lastMX) * right + (float)(my - lastMY) * up) * speed;
            lastMX = mx; lastMY = my;
        }
        else { midDown = false; }

        // ---- Left click: anchor picking (Setup) or fluid push (else) ----
        {
            const bool leftDown = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
                                  && !ImGui::GetIO().WantCaptureMouse;
            const bool leftPressed = leftDown && !g_prevLeftDown;   // rising edge

            glm::mat4 V = viewMatrix();
            float aspect = (float)g_winW / std::max((float)g_winH, 1.f);
            glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 100.f);

            bool  interacting = false;
            PVec3 hitPos{ 0,0,0 };

            if (g_dstate == DeformState::Setup) {
                // Toggle anchor vertices on the rest mesh by clicking.
                if (leftPressed && g_deformMesh.n_vertices() > 0) {
                    double mx, my; glfwGetCursorPos(window, &mx, &my);
                    Ray ray = screenToRay((float)mx, (float)my, g_winW, g_winH, V, P);
                    PickResult pr = pickVertex(ray, g_deformMesh);
                    if (pr.hit && pr.vertexIndex >= 0) {
                        auto it = std::find(g_anchors.begin(), g_anchors.end(), pr.vertexIndex);
                        if (it == g_anchors.end()) g_anchors.push_back(pr.vertexIndex);
                        else                       g_anchors.erase(it);
                        rebuildAnchorBuffer();
                    }
                }
                // (no fluid interaction while picking anchors)
            }
            else if (leftDown) {
                double mx, my; glfwGetCursorPos(window, &mx, &my);

                float ndcX = 2.0f * (float)mx / (float)g_winW - 1.0f;
                float ndcY = 1.0f - 2.0f * (float)my / (float)g_winH;

                glm::vec4 rayClip(ndcX, ndcY, -1.0f, 1.0f);
                glm::vec4 rayEye = glm::inverse(P) * rayClip;
                rayEye.z = -1.0f; rayEye.w = 0.0f;
                glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(V) * rayEye));

                float cx = g_camDist * cosf(g_camPitch) * sinf(g_camYaw);
                float cy = g_camDist * sinf(g_camPitch);
                float cz = g_camDist * cosf(g_camPitch) * cosf(g_camYaw);
                glm::vec3 rayOrig = g_camTarget + glm::vec3(cx, cy, cz);

                // Intersect with a plane through camTarget, facing the camera
                glm::vec3 camFwd = glm::normalize(g_camTarget - rayOrig);
                float denom = glm::dot(camFwd, rayDir);
                if (fabsf(denom) > 1e-6f) {
                    float t = glm::dot(camFwd, g_camTarget - rayOrig) / denom;
                    if (t > 0.0f) {
                        glm::vec3 wp = rayOrig + t * rayDir;
                        hitPos = { wp.x, wp.y, wp.z };
                        interacting = true;
                    }
                }
            }

            fluid.setInteraction(interacting, hitPos);
            g_prevLeftDown = leftDown;
        }

        // ---- ImGui frame ----------------------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("FPS: %d  |  Particles: %u", fps, fluid.params().particleCount);
        ImGui::Separator();

        // Scene selector
        if (ImGui::Combo("Scene", &currentScene,
            [](void* data, int idx, const char** out) -> bool {
                auto* s = (const std::vector<Scene>*)data;
                if (idx < 0 || idx >= (int)s->size()) return false;
                *out = (*s)[idx].name;
                return true;
            },
            (void*)&scenes, (int)scenes.size()))
        {
            const Scene& sc = scenes[currentScene];
            cfg.fluid = sc.fluid;
            pointSize = sc.pointSize;
            g_camYaw = sc.camYaw;
            g_camPitch = sc.camPitch;
            g_camDist = sc.camDist;
            g_camTarget = glm::vec3(
                (sc.fluid.boundsMin.x + sc.fluid.boundsMax.x) * 0.5f,
                (sc.fluid.boundsMin.y + sc.fluid.boundsMax.y) * 0.5f,
                (sc.fluid.boundsMin.z + sc.fluid.boundsMax.z) * 0.5f);

            fluid.setParams(cfg.fluid);
            fluid.setBounds(cfg.fluid.boundsMin, cfg.fluid.boundsMax, cfg.fluid.boundDamping);

            fillBoxVerts(boxData, cfg.fluid.boundsMin, cfg.fluid.boundsMax);
            glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(boxData), boxData);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            fluid.setParticles(spawnParticles(cfg.fluid));

            // Switching scenes invalidates any in-progress deformation session.
            g_dstate = DeformState::Inactive;
            g_deformer.reset();
            g_anchors.clear();
            g_lastHandleSet.clear();
            g_restPositions.clear();
            g_anchorPtCount = 0;
            g_arrowVertCount = 0;

            // Scene-specific SDF setup: the container scene drops in a box that
            // holds the fluid; every other scene starts with the SDF cleared.
            if (std::string(sc.name) == "SDF Container (Leak Test)") {
                loadContainerBox(fluid);
            } else {
                g_sdfEnabled = false;
                g_sdfContainer = false;
                fluid.setSDFBoundary(nullptr);
            }
        }
        ImGui::Separator();

        if (ImGui::Checkbox("Paused", &paused)) {}
        ImGui::SameLine();
        if (ImGui::Button("Step"))    stepOnce = true;
        ImGui::SameLine();
        if (ImGui::Button("Respawn")) respawn = true;

        ImGui::Separator(); ImGui::Text("Rendering");
        ImGui::SliderFloat("Point Size (px)", &pointSize, 1.0f, 40.0f);

        ImGui::Separator(); ImGui::Text("Time & Forces");
        bool solverDirty = false;
        solverDirty |= ImGui::SliderFloat("dt", &cfg.fluid.dt, 1.f / 240.f, 1.f / 30.f);
        solverDirty |= ImGui::SliderFloat3("gravity", &cfg.fluid.gravity.x, -10.f, 10.f);

        ImGui::Separator(); ImGui::Text("PBF Params");
        solverDirty |= ImGui::SliderFloat("h", &cfg.fluid.h, 0.01f, 0.30f);
        solverDirty |= ImGui::SliderFloat("rho0", &cfg.fluid.rho0, 100.f, 3000.f);
        {
            int v = (int)cfg.fluid.solverIterations;
            if (ImGui::SliderInt("solver iters", &v, 1, 12)) { cfg.fluid.solverIterations = (U32)v; solverDirty = true; }
        }
        {
            int v = (int)cfg.fluid.substepIterations;
            if (ImGui::SliderInt("substeps", &v, 1, 5)) { cfg.fluid.substepIterations = (U32)v; solverDirty = true; }
        }
        solverDirty |= ImGui::SliderFloat("eps", &cfg.fluid.eps, 1e-8f, 1e-3f, "%.8f", ImGuiSliderFlags_Logarithmic);

        ImGui::Separator(); ImGui::Text("Spawn");
        bool spawnDirty = false;
        spawnDirty |= ImGui::Checkbox("spawnRandom", &cfg.fluid.spawnRandom);
        spawnDirty |= ImGui::SliderFloat("spacing", &cfg.fluid.spacing, 0.005f, 0.15f);
        spawnDirty |= ImGui::SliderFloat3("spawnMin", &cfg.fluid.spawnMin.x, -10.f, 10.f);
        spawnDirty |= ImGui::SliderFloat3("spawnMax", &cfg.fluid.spawnMax.x, -10.f, 10.f);
        spawnDirty |= ImGui::SliderFloat3("initialVel", &cfg.fluid.initialVelocity.x, -5.f, 5.f);

        ImGui::Separator(); ImGui::Text("Bounds (AABB)");
        bool boundsDirty = false;
        boundsDirty |= ImGui::SliderFloat3("boundsMin", &cfg.fluid.boundsMin.x, -20.0f, 20.f);
        boundsDirty |= ImGui::SliderFloat3("boundsMax", &cfg.fluid.boundsMax.x, -20.0f, 20.f);
        boundsDirty |= ImGui::SliderFloat("boundDamping", &cfg.fluid.boundDamping, 0.f, 1.f);

        ImGui::Separator(); ImGui::Text("SDF Boundary");
        if (ImGui::Checkbox("enableSDF", &g_sdfEnabled))
            fluid.setSDFBoundary(g_sdfEnabled && g_sdf.valid() ? &g_sdf : nullptr);
        if (ImGui::Checkbox("container (keep fluid inside)", &g_sdfContainer))
            fluid.setSDFContainment(g_sdfContainer);
        if (ImGui::Button("Load Container Box")) loadContainerBox(fluid);
        if (ImGui::SliderFloat("sdfPadding", &g_sdfPadding, 0.0f, 0.2f, "%.3f"))
            fluid.setSDFPadding(g_sdfPadding);
        ImGui::SliderFloat("sdfMeshSize", &g_sdfMeshSize, 0.5f, 20.0f);
        ImGui::SliderFloat("sdfCell", &g_sdfCell, 0.04f, 0.25f, "%.3f");
        ImGui::SliderInt("sdfUpdateEvery (collision refresh)", &g_sdfUpdateEvery, 1, 8);
        {
            const char* preview = (g_selFile >= 0 && g_selFile < (int)g_offFiles.size())
                ? g_offFiles[g_selFile].c_str() : "Load .off mesh...";
            if (ImGui::BeginCombo("mesh", preview)) {
                for (int i = 0; i < (int)g_offFiles.size(); ++i) {
                    bool sel = (g_selFile == i);
                    if (ImGui::Selectable(g_offFiles[i].c_str(), sel)) {
                        g_selFile = i;
                        loadMeshBoundary(g_offFiles[i], fluid);
                    }
                }
                ImGui::EndCombo();
            }
            if (g_selFile >= 0 && ImGui::Button("Rebuild SDF"))
                loadMeshBoundary(g_offFiles[g_selFile], fluid);
            ImGui::SameLine();
            if (ImGui::Button("Rescan")) scanOffAssets();
        }

        ImGui::Separator(); ImGui::Text("Two-Way Coupling (Step 1: Contacts)");
        ImGui::Checkbox("enableCoupling (contact forces + arrows)", &g_couplingEnabled);
        ImGui::SliderFloat("contactRadius", &g_contactRadius, 0.01f, 0.5f, "%.3f");
        ImGui::Checkbox("magnitude = velocity into surface", &g_useVelMagnitude);
        ImGui::SliderFloat("arrowScale", &g_forceScale, 0.01f, 3.0f, "%.3f");
        ImGui::SliderFloat("controlThreshold (handle enter)", &g_controlThreshold, 0.0f, 5.0f, "%.3f");
        ImGui::SliderFloat("controlThresholdLow (handle exit)", &g_controlThresholdLow, 0.0f, 5.0f, "%.3f");
        if (g_controlThresholdLow > g_controlThreshold) g_controlThresholdLow = g_controlThreshold;
        ImGui::Text("contacts: %d   control pts: %d", g_contactCount, g_controlCount);

        ImGui::Separator(); ImGui::Text("Deformation (Step 3: anchors -> handles)");
        {
            const char* stateName =
                g_dstate == DeformState::Inactive ? "Inactive" :
                g_dstate == DeformState::Setup    ? "Setup (pick anchors)" :
                g_dstate == DeformState::Ready    ? "Ready (paused)" :
                g_dstate == DeformState::Running  ? "Running" : "Paused";
            ImGui::Text("state: %s   anchors: %d   handles: %d   refactors: %d",
                        stateName, (int)g_anchors.size(), g_handleCount, g_refactorCount);

            // Per-phase timing of the coupling hot path. The biggest number is the
            // bottleneck to attack next. refactor==0 most frames means the handle
            // set is stable (hysteresis working); a big refactor means the set
            // churned -> enable incremental (AMD) or widen the hysteresis gap.
            ImGui::Text("ms  contacts %.2f | refactor %.2f | solve %.2f | sdf %.2f | upload %.2f",
                        g_msContacts, g_msRefactor, g_msSolve, g_msSdfUpdate, g_msSdfUpload);

            const bool haveMesh = g_sdfEnabled && !g_meshData.verts.empty();

            if (g_dstate == DeformState::Inactive) {
                if (!haveMesh) ImGui::TextDisabled("Load an SDF mesh first.");
                if (haveMesh && ImGui::Button("Setup Anchors (clears particles)"))
                    enterSetup(fluid);
            }
            else if (g_dstate == DeformState::Setup) {
                ImGui::TextWrapped("Left-click mesh vertices to toggle anchors (green).");
                if (ImGui::Button("Confirm Anchors & Spawn"))
                    confirmAnchorsAndSpawn(fluid, cfg.fluid);
                ImGui::SameLine();
                if (ImGui::Button("Clear Anchors")) { g_anchors.clear(); rebuildAnchorBuffer(); }
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) { g_dstate = DeformState::Inactive; g_anchors.clear(); rebuildAnchorBuffer(); }
            }
            else if (g_dstate == DeformState::Ready) {
                if (ImGui::Button("Start")) g_dstate = DeformState::Running;
                ImGui::SameLine();
                if (ImGui::Button("Back to Setup")) enterSetup(fluid);
            }
            else if (g_dstate == DeformState::Running) {
                if (ImGui::Button("Pause")) g_dstate = DeformState::Paused;
            }
            else if (g_dstate == DeformState::Paused) {
                if (ImGui::Button("Resume")) g_dstate = DeformState::Running;
                ImGui::SameLine();
                if (ImGui::Button("Back to Setup")) enterSetup(fluid);
            }

            ImGui::SliderFloat("stiffness", &g_stiffness, 0.0f, 20.0f, "%.2f");
            ImGui::SliderFloat("maxDisplacement/frame", &g_maxDisplacement, 0.001f, 0.5f, "%.3f");
            ImGui::SliderFloat("maxTotalDisplacement", &g_maxTotalDisp, 0.01f, 2.0f, "%.3f");

            ImGui::Checkbox("restoring force (spring back to rest)", &g_restoreEnabled);
            ImGui::SliderFloat("restoreStrength", &g_restoreStrength, 0.0f, 0.5f, "%.3f");

            // Hard reset: both solid and fluid back to their initial state.
            if (ImGui::Button("HARD RESET (mesh + fluid)")) {
                hardReset(fluid, cfg.fluid);
                paused = true;   // leave it frozen at the initial state
            }
        }

        ImGui::Separator(); ImGui::Text("Neighbor Search");
        {
            int hs = (int)cfg.fluid.hashSize;
            if (ImGui::InputInt("hashSize", &hs)) { if (hs < 1024) hs = 1024; cfg.fluid.hashSize = (U32)hs; solverDirty = true; }
        }

        ImGui::Separator(); ImGui::Text("sCorr");
        bool scorrDirty = false;
        scorrDirty |= ImGui::Checkbox("enableSCorr", &cfg.fluid.enableSCorr);
        scorrDirty |= ImGui::SliderFloat("kCorr", &cfg.fluid.kCorr, 0.f, 0.02f);
        scorrDirty |= ImGui::SliderFloat("nCorr", &cfg.fluid.nCorr, 1.f, 8.f);
        scorrDirty |= ImGui::SliderFloat("deltaQ", &cfg.fluid.deltaQ, 0.05f, 0.6f);

        ImGui::Separator(); ImGui::Text("Cohesion");
        solverDirty |= ImGui::SliderFloat("cohesionStrength", &cfg.fluid.cohesionStrength, 0.0f, 0.05f, "%.4f");

        ImGui::Separator(); ImGui::Text("Mouse Interaction");
        solverDirty |= ImGui::SliderFloat("interactionRadius", &cfg.fluid.interactionRadius, 0.1f, 5.0f);
        solverDirty |= ImGui::SliderFloat("interactionStrength", &cfg.fluid.interactionStrength, -20.0f, 20.0f);

        ImGui::Separator(); ImGui::Text("Viscosity");
        bool viscDirty = false;
        viscDirty |= ImGui::Checkbox("enableViscosity", &cfg.fluid.enableViscosity);
        viscDirty |= ImGui::SliderFloat("viscosity", &cfg.fluid.viscosity, 0.f, 0.2f);
        if (viscDirty) solverDirty = true;

        ImGui::Separator(); ImGui::Text("Vorticity Confinement");
        bool vortDirty = false;
        vortDirty |= ImGui::Checkbox("enableVorticity", &cfg.fluid.enableVorticity);
        vortDirty |= ImGui::SliderFloat("epsilon", &cfg.fluid.vorticityEpsilon, 0.f, 1.f);
        if (vortDirty) solverDirty = true;

        ImGui::Separator(); ImGui::Text("APBF (Adaptive Iterations)");
        bool apbfDirty = false;
        apbfDirty |= ImGui::Checkbox("enableAPBF", &cfg.fluid.enableAPBF);
        if (cfg.fluid.enableAPBF) {
            {
                int v = (int)cfg.fluid.minLOD;
                if (ImGui::SliderInt("minLOD", &v, 1, 8)) { cfg.fluid.minLOD = (U32)v; apbfDirty = true; }
            }
            {
                int v = (int)cfg.fluid.maxLOD;
                if (ImGui::SliderInt("maxLOD", &v, 1, 12)) { cfg.fluid.maxLOD = (U32)v; apbfDirty = true; }
            }
            apbfDirty |= ImGui::SliderFloat("lodMaxDist", &cfg.fluid.lodMaxDist, 1.0f, 30.0f);
        }
        if (apbfDirty) solverDirty = true;

        ImGui::Separator();
        ImGui::TextDisabled("LMB: interact  |  RMB: orbit  |  MMB: pan  |  Scroll: zoom");
        ImGui::End();

        // ---- Apply parameter changes ----------------------------
        if (solverDirty || scorrDirty || boundsDirty) {
            fluid.setParams(cfg.fluid);
            fluid.setBounds(cfg.fluid.boundsMin, cfg.fluid.boundsMax, cfg.fluid.boundDamping);
        }
        if (boundsDirty) {
            fillBoxVerts(boxData, cfg.fluid.boundsMin, cfg.fluid.boundsMax);
            glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(boxData), boxData);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        if (spawnDirty) respawn = true;
        if (respawn) {
            fluid.setParticles(spawnParticles(cfg.fluid));
            respawn = false;
        }

        // ---- Two-way coupling: contacts (Step 1) + deform (Step 3) ----
        // Runs BEFORE fluid.step() on purpose. computeContactForces reads the
        // particle buffer the GPU finished during the PREVIOUS frame, so the
        // readback no longer blocks the CPU waiting on the step we are about to
        // issue — that hard sync was the 100->30 FPS stall. As a bonus, deform
        // updates the SDF before step() samples it, so the fluid collides
        // against the latest shape. One-frame contact latency, invisible here.
        // Contacts are needed when visualizing arrows OR when Running (handles
        // come from these forces). Deformation runs only in the Running state.
        const bool wantContacts = g_sdfEnabled &&
                                  (g_couplingEnabled || g_dstate == DeformState::Running);
        if (wantContacts) computeContactForces(fluid);
        else              g_arrowVertCount = 0;

        if (g_dstate == DeformState::Running) deformStep(fluid, cfg.fluid.dt);

        // ---- Simulation step (GPU only) -------------------------
        // Inactive (legacy): driven by the paused/Step controls.
        // Deformation states: only Running advances the fluid.
        bool doStep = (g_dstate == DeformState::Inactive) ? (!paused || stepOnce)
                                                          : (g_dstate == DeformState::Running);
        if (doStep) {
            // Pass camera position for APBF LOD computation
            float cx = g_camDist * cosf(g_camPitch) * sinf(g_camYaw);
            float cy = g_camDist * sinf(g_camPitch);
            float cz = g_camDist * cosf(g_camPitch) * cosf(g_camYaw);
            glm::vec3 camPos = g_camTarget + glm::vec3(cx, cy, cz);
            fluid.setCameraPos({ camPos.x, camPos.y, camPos.z });

            fluid.step();
            stepOnce = false;
        }

        // ---- Clear -----------------------------------------------
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glm::mat4 V = viewMatrix();
        float aspect = (float)g_winW / std::max((float)g_winH, 1.f);
        glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 100.f);
        glm::mat4 VP = P * V;

        // ---- Draw particles (instanced from SSBO) ---------------
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(particleProg);
        glUniformMatrix4fv(glGetUniformLocation(particleProg, "view"), 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(glGetUniformLocation(particleProg, "projection"), 1, GL_FALSE, glm::value_ptr(P));
        glUniformMatrix4fv(glGetUniformLocation(particleProg, "model"), 1, GL_FALSE, glm::value_ptr(glm::mat4(1.f)));
        glUniform1f(glGetUniformLocation(particleProg, "scale"), pointSize);

        fluid.bindParticlesForRendering();   // SSBO 0 stays on GPU

        glBindVertexArray(emptyVAO);
        // Use the live particle count (0 during Setup, may differ from the config
        // when a grid spawn under-fills) so we never read past the SSBO.
        glDrawArraysInstanced(GL_POINTS, 0, 1, fluid.params().particleCount);
        glBindVertexArray(0);

        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_BLEND);

        // ---- Draw boundary box ----------------------------------
        glUseProgram(lineProg);
        glUniformMatrix4fv(glGetUniformLocation(lineProg, "VP"), 1, GL_FALSE, glm::value_ptr(VP));
        glUniform3f(glGetUniformLocation(lineProg, "color"), 0.35f, 0.55f, 0.85f);

        glBindVertexArray(boxVAO);
        glDrawArrays(GL_LINES, 0, 24);
        glBindVertexArray(0);

        // ---- Draw SDF boundary mesh (wireframe) -----------------
        if (g_sdfEnabled && g_meshIdxCount > 0) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glUseProgram(lineProg);
            glUniformMatrix4fv(glGetUniformLocation(lineProg, "VP"), 1, GL_FALSE, glm::value_ptr(VP));
            glUniform3f(glGetUniformLocation(lineProg, "color"), 0.9f, 0.8f, 0.4f);
            glBindVertexArray(g_meshVAO);
            glDrawElements(GL_TRIANGLES, g_meshIdxCount, GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        // ---- Draw contact-force arrows (Step 2 checkpoint) ------
        if (g_arrowVertCount > 0) {
            glUseProgram(lineProg);
            glUniformMatrix4fv(glGetUniformLocation(lineProg, "VP"), 1, GL_FALSE, glm::value_ptr(VP));
            glUniform3f(glGetUniformLocation(lineProg, "color"), 1.0f, 0.25f, 0.1f);  // red-orange
            glBindVertexArray(g_arrowVAO);
            glDrawArrays(GL_LINES, 0, g_arrowVertCount);
            glBindVertexArray(0);
        }

        // ---- Draw anchor points (Step 3) ------------------------
        // lineProg has no gl_PointSize, so use fixed-function point size
        // (GL_PROGRAM_POINT_SIZE must stay disabled for glPointSize to apply).
        if (g_anchorPtCount > 0) {
            glUseProgram(lineProg);
            glUniformMatrix4fv(glGetUniformLocation(lineProg, "VP"), 1, GL_FALSE, glm::value_ptr(VP));
            glUniform3f(glGetUniformLocation(lineProg, "color"), 0.2f, 1.0f, 0.3f);   // green anchors
            glPointSize(12.0f);
            glBindVertexArray(g_anchorVAO);
            glDrawArrays(GL_POINTS, 0, g_anchorPtCount);
            glBindVertexArray(0);
        }

        // ---- ImGui overlay --------------------------------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ---- Cleanup ------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &emptyVAO);
    glDeleteVertexArrays(1, &boxVAO);
    glDeleteBuffers(1, &boxVBO);
    glDeleteVertexArrays(1, &g_meshVAO);
    glDeleteBuffers(1, &g_meshVBO);
    glDeleteBuffers(1, &g_meshEBO);
    glDeleteVertexArrays(1, &g_arrowVAO);
    glDeleteBuffers(1, &g_arrowVBO);
    glDeleteVertexArrays(1, &g_anchorVAO);
    glDeleteBuffers(1, &g_anchorVBO);
    glDeleteProgram(particleProg);
    glDeleteProgram(lineProg);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
