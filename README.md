# DawnViewer

This is a simple project that uses [Google's Dawn](https://dawn.googlesource.com/dawn) implementation of WebGPU in C++ to display a spinning 3D model. It also includes a basic ECS implementation (with [entt](https://github.com/skypjack/entt)) that can be used to add multiple models with different transforms. You can use this as the starting point for a game engine by extending it.

## Building
The program can be compiled on Windows, Mac OS, and Linux.
1. Clone the repository: `git clone https://github.com/Edward205/dawn3dviewer.git`
2. Get the submodules. **Do not recursively clone the submodules**, as the Dawn repository contains a ton of submodules which we don't need: `git submodule init`, `git submodule update`
3. Build with CMake: `cmake -B build`, `cmake --build build`

## Usage
Execute the program, with the OBJ format model you want to open as the first argument: `dawn3dviewer teapot.obj`.

On Windows, you may drag the model on top of the executable in Explorer to open it directly.

## Acknowledgments
- I learned Dawn from this tutorial: https://eliemichel.github.io/LearnWebGPU/index.html
  - I heavily modified its code and its proposed class structure because I disagree with using C-style functions in C++, and I wanted this project to have a larger scope than just a WebGPU tech demo
- https://dawn.googlesource.com/dawn
- https://github.com/skypjack/entt
- https://github.com/g-truc/glm.git
- https://github.com/gabime/spdlog.git
- https://github.com/tinyobjloader/tinyobjloader