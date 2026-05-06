#include "fluid/Kernels.h"
#include <algorithm> // std::max

// poly6 kernel and spiky kernel combination is very fast, no conditionals. 
// cubic kernel is better for robustness but has condtionals, so not good for GPU parallelism.
// We will stick to POLY6 + SPIKY combination for this project. 


F32 calcPoly6Kernel(const PVec3& r, const F32 h)
{
    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 315.0f / (64.0f * pi);

    const F32 h2 = h * h;
    const F32 r2 = norm2(r);

    if (r2 > h2) {
        return 0.0f;
    }

    const F32 h4 = h2 * h2;
    const F32 h9 = h4 * h4 * h;
    const F32 diff = h2 - r2;
    const F32 diff3 = diff * diff * diff;

    return (coeff / h9) * diff3;
}

PVec3 calcGradPoly6Kernel(const PVec3& r, const F32 h)
{
    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 945.0f / (32.0f * pi);

    const F32 h2 = h * h;
    const F32 r2 = norm2(r);

    if (r2 > h2) {
        return make_pvec3(0.0f, 0.0f, 0.0f);
    }

    const F32 h4 = h2 * h2;
    const F32 h9 = h4 * h4 * h;

    const F32 diff = h2 - r2;
    const F32 diff2 = diff * diff;

    // -r * (coeff / h^9) * (h^2 - r^2)^2
    return r * (-(coeff / h9) * diff2);
}

F32 calcSpikyPow2Kernel(const PVec3& r, const F32 h)
{
    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 15.0f /  (2 * pi);

    const F32 dist = norm(r);

    if (dist > h) {
        return 0.0f;
    }

	const F32 h2 = h * h;
    const F32 h3 = h2 * h;
    const F32 h5 = h2 * h3;
    const F32 diff = h - dist;
    const F32 diff2 = diff * diff;

    return (coeff / h5) * diff2;
}

PVec3 calcGradSpikyPow2Kernel(const PVec3& r, const F32 h)
{
    const F32 dist = norm(r);

    // Early exit for out-of-bounds AND division-by-zero protection
    if (dist > h || dist < 1e-5f) {
        return make_pvec3(0.0f, 0.0f, 0.0f);
    }

    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 15.0f / pi;

    // Optimized powers
    const F32 h2 = h * h;
    const F32 h3 = h2 * h;
    const F32 h5 = h2 * h3;

    const F32 diff = h - dist;

    // 1. Calculate the scalar derivative (matches your DerivativeSpikyPow2 logic)
    const F32 scalarDerivative = -diff * (coeff / h5);

    // 2. Multiply by the normalized direction vector (r / dist)
    return r * (scalarDerivative / dist);
}

F32 calcSpikyPow3Kernel(const PVec3& r, const F32 h)
{
    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 15.0f / pi;

    const F32 dist = norm(r);

    if (dist > h) {
        return 0.0f;
    }

    const F32 h3 = h * h * h;
    const F32 h6 = h3 * h3;
    const F32 diff = h - dist;
    const F32 diff3 = diff * diff * diff;

    return (coeff / h6) * diff3;
}

PVec3 calcGradSpikyPow3Kernel(const PVec3& r, const F32 h)
{
    const F32 dist = norm(r);

    // Early exit for out-of-bounds AND division-by-zero protection
    if (dist > h || dist < 1e-5f) {
        return make_pvec3(0.0f, 0.0f, 0.0f);
    }

    constexpr F32 pi = 3.14159265358979323846f;
    constexpr F32 coeff = 45.0f / pi; // Derivative of power 3 pulls down a 3 (15 * 3 = 45)

    // Optimized powers for h^6
    const F32 h2 = h * h;
    const F32 h3 = h2 * h;
    const F32 h6 = h3 * h3;

    const F32 diff = h - dist;
    const F32 diff2 = diff * diff; // This maps directly to your 'v * v'

    // 1. Calculate the scalar derivative (matches -v * v * K_SpikyPow3Grad)
    const F32 scalarDerivative = -diff2 * (coeff / h6);

    // 2. Multiply by the normalized direction vector (r / dist)
    return r * (scalarDerivative / dist);
}

F32 calcCubicKernel(const PVec3& r, const F32 h)
{
    const F32 dist = norm(r);
    const F32 q = dist / h;

    if (q > 1.0f) {
        return 0.0f;
    }

    constexpr F32 pi = 3.14159265358979323846f;
    const F32 h3 = h * h * h;
    const F32 k = 8.0f / (pi * h3);

    F32 res = 0.0f;

    if (q <= 0.5f) {
        const F32 q2 = q * q;
        const F32 q3 = q2 * q;
        res = k * (6.0f * q3 - 6.0f * q2 + 1.0f);
    }

    else {
        const F32 diff = 1.0f - q;
        res = k * 2.0f * (diff * diff * diff);
    }

    return res;
}

PVec3 calcGradCubicKernel(const PVec3& r, const F32 h)
{
    const F32 dist = norm(r);
    const F32 q = dist / h;

    if (q > 1.0f || dist <= 1e-12f) {
        return make_pvec3(0.0f, 0.0f, 0.0f);
    }

    constexpr F32 pi = 3.14159265358979323846f;
    const F32 h3 = h * h * h;
    const F32 l = 48.0f / (pi * h3);

    const PVec3 gradq = r * (1.0f / (dist * h));
    PVec3 res = make_pvec3(0.0f, 0.0f, 0.0f);

    if (q <= 0.5f) {
        res = gradq * (l * q * (3.0f * q - 2.0f));
    }
    else {
        const F32 diff = 1.0f - q;
        res = gradq * (-l * diff * diff);
    }

    return res;
}



F32 calcDensityKernel(const PVec3& r, const F32 h)
{
    return calcPoly6Kernel(r, h);
}

PVec3 calcDensityDerivative(const PVec3& r, const F32 h)
{
    return calcGradSpikyPow2Kernel(r, h);
}

PVec3 calcLambdaDerivative(const PVec3& r, const F32 h)
{
    return calcGradSpikyPow3Kernel(r, h);
}

F32 calcSCorrKernel(const PVec3& r, const F32 h)
{
    return calcPoly6Kernel(r, h);
}

F32 calcXSPHKernel(const PVec3& r, const F32 h)
{
    return calcPoly6Kernel(r, h);
}