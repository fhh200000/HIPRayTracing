/* @file Win32Window.cpp

	Win32 API implementation of window-related functions.
	SPDX-License-Identifier: WTFPL

*/

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#undef ERROR	// Bug WA:"ERROR redefined"
#include <Window.hpp>

static HINSTANCE application_handle;
static temination_signal_handler_t terminating_handler;

status_t CreateMainWindow(temination_signal_handler_t handler)
{
	application_handle = GetModuleHandle(nullptr);
	if (!application_handle) {
		FATAL("Cannot get application handle!\n");
		return STATUS_NOT_SUPPORTED;
	}

	terminating_handler = handler;
	return STATUS_SUCCESS;
}

status_t DestroyMainWindow()
{
	return STATUS_SUCCESS;
}
