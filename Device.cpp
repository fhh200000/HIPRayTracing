/* @file Device.cpp

    Implementations related to HIP device.
    SPDX-License-Identifier: WTFPL

*/

#include <Device.hpp>
#include <hip/hip_runtime.h>

status_t InitializeDevice()
{
    int device_count;
    auto hip_result = hipGetDeviceCount(&device_count);
    if (hip_result != hipSuccess) {
        FATAL("Cannot find any "
#ifdef __HIP_PLATFORM_AMD__
            "AMD "
#elif defined __HIP_PLATFORM_NVIDIA__
            "Nvidia "
#endif
            "device to open HIP!\n");
        return STATUS_NOT_SUPPORTED;
    }

    // For now, we only select 1 device with best performance.
    // HIP supports multi-devices, but we are not covering it.

    int iterator = 0;
    uint64_t current_rank_score, best_id=UINT64_MAX, best_rank_score=0;
    hipDeviceProp_t current_device_prop;

    VERBOSE("GPU scores:\n");
    while (iterator++ < device_count) {
        // Rank GPU by maxThreadsPerMultiProcessor * clockRate. Not caring VRAM.
        current_device_prop = {};
        hip_result = hipGetDeviceProperties(&current_device_prop, iterator-1);
        if (hip_result != hipSuccess) {
            WARNING("Cannot initialize Device #%d. Skip\n", iterator);
        }
        current_rank_score = (uint64_t)current_device_prop.maxThreadsPerMultiProcessor * current_device_prop.clockRate;
        VERBOSE("\t#%d %s:%llu\n", iterator, current_device_prop.name, current_rank_score);
        if (current_rank_score > best_rank_score) {
            best_rank_score = current_rank_score;
            best_id = iterator - 1;
        }
    }
    VERBOSE("\n");
    if (best_id == UINT64_MAX) {
        FATAL("No HIP devices can be retrieved\n");
        return STATUS_NOT_SUPPORTED;
    }

    // Best GPU chosen!
    hipGetDeviceProperties(&current_device_prop, best_id);
    hipSetDevice(best_id);

    INFO("Selected HIP device : %s(%s)\n", current_device_prop.name, current_device_prop.integrated?"iGPU":"dGPU");
    return STATUS_SUCCESS;
}


status_t DestroyDevice()
{
    return STATUS_SUCCESS;
}