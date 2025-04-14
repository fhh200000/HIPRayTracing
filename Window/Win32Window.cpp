/* @file Win32Window.cpp

	Win32 API implementation of window-related functions.
	SPDX-License-Identifier: WTFPL

*/

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#undef ERROR	// Bug WA:"ERROR redefined"
#include <Window.hpp>

static HINSTANCE application_handle;
static ATOM registered_class;
static HWND hwnd;
static BOOL window_exiting;

static LRESULT CALLBACK MainWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg) {
    case WM_DESTROY: {
        PostQuitMessage(0);
        return 0;
    }
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

status_t CreateMainWindow()
{
	application_handle = GetModuleHandle(nullptr);
	if (!application_handle) {
		FATAL("Cannot get application handle!\n");
		return STATUS_NOT_SUPPORTED;
	}

    WNDCLASSEX wcx = {
        sizeof(wcx),                                            // cbSize
        0,                                                      // style
        MainWindowProc,                                         // lpfnWndProc
        0,                                                      // cbClsExtra
        0,                                                      // cbWndExtra
        application_handle,                                     // hInstance
        LoadIcon(NULL, IDI_APPLICATION),                        // hIcon
        LoadCursor(NULL,IDC_ARROW),                             // hCursor
        reinterpret_cast<HBRUSH>(GetStockObject(WHITE_BRUSH)),  // hbrBackground
        NULL,                                                   // lpszMenuName
        __TEXT("HipRayTracingClass"),                           // lpszClassName
        NULL                                                    // hIconSm
    };
    registered_class = RegisterClassEx(&wcx);
    if (!registered_class) {
        FATAL("Cannot register window class!\n");
        return STATUS_UNKNOWN_ERROR;
    }

    hwnd = CreateWindow (
        reinterpret_cast<LPCWSTR>(registered_class),
        WINDOW_TITLE,
        WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, 
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        WINDOW_WIDTH,
        WINDOW_HEIGHT, 
        (HWND)NULL,
        (HMENU)NULL,
        application_handle,
        (LPVOID)NULL
    );

    if (!hwnd) {
        FATAL("Cannot create window!\n");
        return STATUS_UNKNOWN_ERROR;
    }
    UpdateWindow(hwnd);
	return STATUS_SUCCESS;
}

status_t DestroyMainWindow()
{
    BOOL result;
    result = UnregisterClass(reinterpret_cast<LPCWSTR>(registered_class), application_handle);
	return STATUS_SUCCESS;
}

status_t EnterMainLoop(temination_signal_handler_t handler)
{
    MSG msg;
    BOOL bRet;

    ShowWindow(hwnd, SW_SHOWNORMAL);
    while (true)
    {
        bRet = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE);
        if (bRet == -1) {
            window_exiting = TRUE;
            goto loop_end;
        }
        if (msg.message == WM_QUIT) {
            window_exiting = TRUE;
            goto loop_end;
        }
        else
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (window_exiting != TRUE) {
            // TODO: Draw message
        }

    }
loop_end:
    handler();
    return STATUS_SUCCESS;
}