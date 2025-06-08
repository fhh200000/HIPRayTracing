/* @file Kernel.cpp

    Definitions related to HIP device.
    SPDX-License-Identifier: WTFPL

*/

#include <Kernel.hpp>
#include <Window.hpp>

__global__ void RayTracingKernel(pixel_t *output_buffer)
{
    unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * WINDOW_WIDTH;

    if (threadIdx.x && threadIdx.y) {
        output_buffer[index].R = 255.999 * (threadIdx.x + blockIdx.x * blockDim.x) / WINDOW_WIDTH;
        output_buffer[index].G = 255.999 * (threadIdx.y + blockIdx.y * blockDim.y) / WINDOW_HEIGHT;
        output_buffer[index].B = 0;
        output_buffer[index].A = 0;
    }
}


