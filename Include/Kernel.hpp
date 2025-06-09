/* @file Kernel.hpp

	Computing kernel that will be run on GPU.
	SPDX-License-Identifier: WTFPL

*/

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <Common.hpp>
#include <hip/hip_runtime.h>

#define KERNEL_SIZE 16	// Can be devided by both width and height

__global__ void RayTracingKernel(pixel_t *output_buffer);

#endif
