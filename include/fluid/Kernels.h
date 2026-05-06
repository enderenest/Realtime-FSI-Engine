#ifndef KERNELS_H
#define KERNELS_H

#include "core/Types.h"

// We used the standard SPH kernels and extra cubic for density constraint
// See: https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics#Kernel_functions

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcPoly6Kernel(const PVec3& r, const F32 h);

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3   calcGradPoly6Kernel(const PVec3& r, const F32 h);



/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcSpikyPow2Kernel(const PVec3& r, const F32 h);

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3   calcGradSpikyPow2Kernel(const PVec3& r, const F32 h);



/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcSpikyPow3Kernel(const PVec3& r, const F32 h);

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3   calcGradSpikyPow3Kernel(const PVec3& r, const F32 h);



/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcCubicKernel(const PVec3& r, const F32 h);

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3	calcGradCubicKernel(const PVec3& r, const F32 h);



// I will use the functions below to clearly define which kernel is used for what
// They are just calling the functions above

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcDensityKernel(const PVec3& r, const F32 h);

/// <summary>
/// 
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3   calcDensityDerivative(const PVec3& r, const F32 h);

/// <summary>
///
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
PVec3	calcLambdaDerivative(const PVec3& r, const F32 h);

/// <summary>
///
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcSCorrKernel(const PVec3& r, const F32 h);

/// <summary>
///
/// </summary>
/// <param name="r: displacement vector between two particles"></param>
/// <param name="h: smoothing radius"></param>
/// <returns></returns>
F32		calcXSPHKernel(const PVec3& r, const F32 h);

#endif