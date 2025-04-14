/* @file Main.cpp

	Main entry of the program.
	SPDX-License-Identifier: WTFPL

*/

#include <Device.hpp>
#include <Window.hpp>

static void teminatate()
{
	DestroyMainWindow();
}

int main()
{
	InitializeDevice();
	CreateMainWindow();
	EnterMainLoop(teminatate);
}