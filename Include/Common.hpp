/* @file Common.hpp

	Common values that can be shared program-wide.
	SPDX-License-Identifier: WTFPL

*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <cerrno>
#include <cstdio>

typedef enum {
	STATUS_SUCCESS,
	STATUS_NOT_SUPPORTED = -1,
	STATUS_OUT_OF_RESOURCES = -2,
	STATUS_START_FAILED = -3,
	STATUS_UNKNOWN_ERROR = -4
} status_t;

#pragma pack(push, 1)

typedef struct {
	uint8_t B;
	uint8_t G;
	uint8_t R;
	uint8_t A;
} pixel_t;

#pragma pack(pop)

#define LOGLEVEL_VERBOSE	0		// Debug information
#define LOGLEVEL_INFO		0x1		// Useful information
#define LOGLEVEL_WARNING	0x2		// Something might be wrong
#define LOGLEVEL_ERROR		0x4		// Something is definitely wrong, but can continue
#define LOGLEVEL_FATAL		0x8		// Program cannot continue		

#ifndef LOGLEVEL_ENABLED
#define LOGLEVEL_ENABLED	LOGLEVEL_WARNING	// Enable warning by default.
#endif

#define LOG(level, ...) \
do { \
	if (LOGLEVEL_##level >= LOGLEVEL_ENABLED) { \
			fprintf((LOGLEVEL_##level > LOGLEVEL_WARNING) ? stderr : stdout, __VA_ARGS__); \
	} \
} while (0);

#define VERBOSE(...)	LOG(VERBOSE, __VA_ARGS__)
#define INFO(...)		LOG(INFO, __VA_ARGS__)
#define WARNING(...)	LOG(WARNING, __VA_ARGS__)
#define ERROR(...)		LOG(ERROR, __VA_ARGS__)
#define FATAL(...)		LOG(FATAL, __VA_ARGS__)

#ifdef __INTELLISENSE__
#define __attribute__(x) // MSVC Intellisense do not know __attribute__
#endif

#ifndef __HIP__ // For clangd server
struct dim3;
extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;
#endif

#endif
