# Replay

Replay is a Python-based framework designed to simulate the spread of infectious diseases over real-world human contact networks, using GPU acceleration for high performance at scale.

Built with CUDA 12.8 and Python 3.12.9, Replay is intended for researchers, epidemiologists, and public health professionals looking for a practical and scalable approach to modeling transmission dynamics using empirical data.

## Background

Understanding how diseases spread through dynamic human contact networks is essential for effective public health planning. Most traditional epidemic models assume simplified or static interaction patterns, which often don’t reflect the complexity of real-world contact behavior.

With growing access to mobility and proximity data, there’s a clear opportunity to build more accurate, data-driven models. However, working with time-resolved contact data can quickly become computationally expensive.

## What Replay Does

Replay addresses this by:

- Converting timestamped contact data into duration-weighted contact graphs  
- Using sparse matrix operations to compute exposure at the individual level  
- Running massive numbers of stochastic simulations in parallel via CUSPARSE  

Rather than approximating, Replay “replays” the actual contact patterns recorded in empirical datasets. This allows for realistic simulation of outbreaks, including localized spread and superspreader events, without hardcoded behavioral assumptions.

## Performance

Replay is optimized for speed and scale. On GPU, it’s capable of running thousands of Monte Carlo simulations per second, significantly outperforming CPU-based approaches, especially in high-density contact scenarios.

## Use Cases

- Public health planning  
- Hospital infection control  
- Real-time outbreak analysis  
- Research into contact-driven epidemic dynamics  

---

## 🔧 Build Walkthrough (Windows)

This section walks you through building Replay from source on **Windows** using **Visual Studio 2022**, **vcpkg**, and **Ninja**.

### ✅ Prerequisites

Make sure the following are installed and available in your system `PATH`:

- [x] [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)  
  – Include the **C++**, **CMake**, and **Windows SDK** components  
- [x] [CUDA Toolkit 12.8+](https://developer.nvidia.com/cuda-toolkit)  
- [x] Git  
- [x] Python 3.12  

---

### Step-by-Step Instructions

1. **Clone the repository and vcpkg submodule**  
   ```bat
   git clone https://github.com/your-username/replay.git
   cd replay
   git clone https://github.com/microsoft/vcpkg.git external/vcpkg
   cd external/vcpkg
   bootstrap-vcpkg.bat
   cd ../..

2. **Install dependencies with manifest mode**

    ```bat
    external\vcpkg\vcpkg install
    ```

3. **Create the build directory and configure with CMake**

    ```bat
    mkdir build
    cd build
    cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Debug ^
      -DCMAKE_TOOLCHAIN_FILE=../external/vcpkg/scripts/buildsystems/vcpkg.cmake
    ```

4. **Build the project**

    ```bat
    cmake --build .
    ```

    The executable will be generated in the tool/build directory as replay.exe.

---

### :repeat: Force a Clean Rebuild

```bat
rd /s /q build
mkdir build
cd build
cmake .. -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DCMAKE_TOOLCHAIN_FILE=../external/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

---

### :arrow_forward: Run the Executable

From the `build/` directory:

```bat
WIP
```

### For Maintainers

**How to update the vcpkg baseline**

```bat
git clone https://github.com/microsoft/vcpkg.git external/vcpkg
cd external/vcpkg
bootstrap-vcpkg.bat
vcpkg x-update-baseline
cd ../..
```

> This updates the `builtin-baseline` field in `vcpkg.json` to match the latest commit from the vcpkg registry.  
> Be sure to **commit the updated `vcpkg.json`** so others build with the same dependency versions.

