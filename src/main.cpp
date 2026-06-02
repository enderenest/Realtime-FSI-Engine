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

static bool loadOFF(const std::string& path, MeshData& out) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    std::string header; f >> header;
    if (header != "OFF") { std::cerr << "Not an OFF file: " << path << "\n"; return false; }

    int numV, numF, numE; f >> numV >> numF >> numE;
    out.verts.resize(numV);
    for (int i = 0; i < numV; ++i)
        f >> out.verts[i][0] >> out.verts[i][1] >> out.verts[i][2];

    out.faces.clear(); out.faces.reserve(numF);
    for (int i = 0; i < numF; ++i) {
        int n; f >> n;
        std::vector<int> poly(n);
        for (int j = 0; j < n; ++j) f >> poly[j];
        for (int j = 1; j + 1 < n; ++j)            // fan triangulation
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

        // ---- Mouse interaction (left click) ---------------------
        {
            bool interacting = false;
            PVec3 hitPos{ 0,0,0 };

            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !ImGui::GetIO().WantCaptureMouse) {
                double mx, my;
                glfwGetCursorPos(window, &mx, &my);

                float ndcX = 2.0f * (float)mx / (float)g_winW - 1.0f;
                float ndcY = 1.0f - 2.0f * (float)my / (float)g_winH;

                glm::mat4 V = viewMatrix();
                float aspect = (float)g_winW / std::max((float)g_winH, 1.f);
                glm::mat4 P = glm::perspective(glm::radians(45.f), aspect, 0.01f, 100.f);

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
        }

        // ---- ImGui frame ----------------------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("FPS: %d  |  Particles: %u", fps, cfg.fluid.particleCount);
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
        spawnDirty |= ImGui::SliderFloat3("spawnMin", &cfg.fluid.spawnMin.x, 0.0f, 6.f);
        spawnDirty |= ImGui::SliderFloat3("spawnMax", &cfg.fluid.spawnMax.x, 0.0f, 6.f);
        spawnDirty |= ImGui::SliderFloat3("initialVel", &cfg.fluid.initialVelocity.x, -5.f, 5.f);

        ImGui::Separator(); ImGui::Text("Bounds (AABB)");
        bool boundsDirty = false;
        boundsDirty |= ImGui::SliderFloat3("boundsMin", &cfg.fluid.boundsMin.x, 0.0f, 30.f);
        boundsDirty |= ImGui::SliderFloat3("boundsMax", &cfg.fluid.boundsMax.x, 0.f, 30.f);
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
        solverDirty |= ImGui::SliderFloat("interactionStrength", &cfg.fluid.interactionStrength, -30.0f, 30.0f);

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

        // ---- Simulation step (GPU only) -------------------------
        if (!paused || stepOnce) {
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
        glDrawArraysInstanced(GL_POINTS, 0, 1, cfg.fluid.particleCount);
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
    glDeleteProgram(particleProg);
    glDeleteProgram(lineProg);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
