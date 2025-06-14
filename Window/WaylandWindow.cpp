/* @file WaylandWindow.cpp
 *
 *	Wayland API implementation of window-related functions.
 *	SPDX-License-Identifier: WTFPL
 *
 */
#include <Common.hpp>
#include <Window.hpp>
#include <Device.hpp>

#include <iostream>
#include <fstream>

pixel_t HostImageBufferData[WINDOW_WIDTH * WINDOW_HEIGHT];
pixel_t *HostImageBuffer = HostImageBufferData;

status_t CreateMainWindow()
{
    return STATUS_SUCCESS;
}
status_t DestroyMainWindow()
{
    return STATUS_SUCCESS;
}
status_t EnterMainLoop(temination_signal_handler_t handler)
{
    status_t status = CalculateOneFrame();
    if (status == STATUS_SUCCESS) {
        std::ofstream outfile("out.p3",std::ios::out);
        outfile<< "P3\n" << WINDOW_WIDTH << ' ' << WINDOW_HEIGHT << "\n" << WINDOW_COLOR_DEPTH-1 << "\n";
        for (size_t i=0;i<WINDOW_WIDTH*WINDOW_HEIGHT;i++) {
            outfile<< (int)HostImageBufferData[i].R << " " << (int)HostImageBufferData[i].G << " " << (int)HostImageBufferData[i].B << "\n";
        }
        outfile.close();
    }
    else {
        ERROR("CalculateOneFrame Error: %d\n",status);
    }
    return status;
}
