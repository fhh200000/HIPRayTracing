/* @file Common.hpp

	Definitions of window-related functions.
	SPDX-License-Identifier: WTFPL

*/

#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <Common.hpp>

#ifndef WINDOW_TITLE
#define WINDOW_TITLE L"HIP Ray Tracing Window"
#endif

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

// Callback functions of termination signal.
typedef void (*temination_signal_handler_t)();

// Exposed host-side image buffer.
extern pixel_t *HostImageBuffer;

status_t CreateMainWindow();
status_t DestroyMainWindow();
status_t EnterMainLoop(temination_signal_handler_t handler);

#endif
