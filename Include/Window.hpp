/* @file Common.hpp

	Definitions of window-related functions.
	SPDX-License-Identifier: WTFPL

*/

#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <Common.hpp>

#ifndef WINDOW_TITLE
#define WINDOW_TITLE "HIP Ray Tracing Window"
#endif

// Callback functions of termination signal.

typedef void (*temination_signal_handler_t)();

status_t CreateMainWindow(temination_signal_handler_t handler);
status_t DestroyMainWindow();

#endif