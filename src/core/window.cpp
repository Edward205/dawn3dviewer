#include "core/window.hpp"
#include "GLFW/glfw3.h"

namespace DawnViewer {
    void Window::create(const char* title, int width, int height) {
        if (!glfwInit()) {
            return;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    };
    void Window::shutdown() {
        glfwDestroyWindow(window);
        //glfwTerminate(); 
    }

}