#pragma once

#include "GLFW/glfw3.h"

namespace DawnViewer {
    class Window {
    public:
        void create(const char* title, int width, int height);
        void shutdown();
    private:
        GLFWwindow* window;
        uint32_t width, height;
    };
}