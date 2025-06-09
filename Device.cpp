/* @file Device.cpp

    Implementations related to HIP device.
    SPDX-License-Identifier: WTFPL

*/

#include <Device.hpp>
#include <Window.hpp>
#include <Kernel.hpp>
#include <hip/hip_runtime.h>

static pixel_t *DeviceImageBuffer;

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
        VERBOSE("\t#%d %s:%lu\n", iterator, current_device_prop.name, current_rank_score);
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
    hip_result = hipGetDeviceProperties(&current_device_prop, best_id);
    hip_result = hipSetDevice(best_id);

    INFO("Selected HIP device : %s(%s)\n", current_device_prop.name, current_device_prop.integrated?"iGPU":"dGPU");

    // Allocate device-side memory.
    hip_result = hipMalloc(&DeviceImageBuffer, sizeof(pixel_t) * WINDOW_WIDTH * WINDOW_HEIGHT);
    if (hip_result != hipSuccess) {
        FATAL("Cannot allocate image buffer for device!\n");
        return STATUS_OUT_OF_RESOURCES;
    }

    // TODO: Generate world & do memcpy().

    return STATUS_SUCCESS;
}

status_t CalculateOneFrame()
{
    // Launch calculation from host.

    RayTracingKernel<<<dim3(WINDOW_WIDTH / KERNEL_SIZE, WINDOW_HEIGHT / KERNEL_SIZE),
        dim3(KERNEL_SIZE, KERNEL_SIZE),
        0,
        hipStreamDefault>>>(DeviceImageBuffer);

    auto hip_result = hipGetLastError();
    // Check if the kernel launch was successful.
    if (hip_result != HIP_SUCCESS) {
        ERROR("Cannot launch Kernel for calculation!\n");
        return STATUS_START_FAILED;
    }

    // Transfer the result back to the host.
    if (hipMemcpy(HostImageBuffer, DeviceImageBuffer, sizeof(pixel_t) * WINDOW_WIDTH * WINDOW_HEIGHT, hipMemcpyDeviceToHost) != HIP_SUCCESS) {
        ERROR("Cannot copy result back to the host!\n");
        return STATUS_OUT_OF_RESOURCES;
    }
    return STATUS_SUCCESS;
}

status_t DestroyDevice()
{
    // Free device-side memory.
    hipFree(&DeviceImageBuffer);
    return STATUS_SUCCESS;
}
